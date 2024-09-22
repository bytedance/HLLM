# Copyright (c) 2024 Bytedance Ltd. and/or its affiliate
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
import transformers
from transformers import AutoConfig, AutoModelForCausalLM
from logging import getLogger

from REC.utils.enum_type import InputType
from REC.model.basemodel import BaseModel, all_gather
from REC.model.HLLM.modeling_llama import LlamaForCausalLM
from REC.model.HLLM.modeling_bert import BertModel


class LLMIDRec(BaseModel):
    input_type = InputType.SEQ

    def __init__(self, config, dataload):
        super(LLMIDRec, self).__init__()
        self.logger = getLogger()

        self.user_pretrain_dir = config['user_pretrain_dir']
        self.gradient_checkpointing = config['gradient_checkpointing']
        self.use_ft_flash_attn = config['use_ft_flash_attn']
        self.logger.info(f"create user llm")
        self.user_llm = self.create_llm(self.user_pretrain_dir, config['user_llm_init'])

        self.item_num = dataload.item_num
        self.item_embedding = nn.Embedding(self.item_num, config['item_embed_dim'], padding_idx=0)
        self.item_id_proj_tower = nn.Identity() if config['item_embed_dim'] == self.user_llm.config.hidden_size else nn.Linear(config['item_embed_dim'], self.user_llm.config.hidden_size, bias=False)
        self.item_embedding.weight.data.normal_(mean=0.0, std=0.02)

        self.loss = config['loss']
        if self.loss == 'nce':
            if config['fix_temp']:
                self.logger.info(f"Fixed logit_scale 20")
                self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.05), requires_grad=False)
            else:
                self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
            self.nce_thres = config['nce_thres'] if config['nce_thres'] else 0.99
            self.num_negatives = config['num_negatives']
            self.logger.info(f"nce thres setting to {self.nce_thres}")
        else:
            raise NotImplementedError(f"Only nce is supported")

    def create_llm(self, pretrain_dir, init=True):
        self.logger.info(f"******* create LLM {pretrain_dir} *******")
        hf_config = AutoConfig.from_pretrained(pretrain_dir, trust_remote_code=True)
        self.logger.info(f"hf_config: {hf_config}")
        hf_config.gradient_checkpointing = self.gradient_checkpointing
        hf_config.use_cache = False
        hf_config.output_hidden_states = True
        hf_config.return_dict = True

        self.logger.info("xxxxx starting loading checkpoint")
        if isinstance(hf_config, transformers.LlamaConfig):
            hf_config.use_ft_flash_attn = self.use_ft_flash_attn
            self.logger.info(f'Using flash attention {hf_config.use_ft_flash_attn} for llama')
            self.logger.info(f'Init {init} for llama')
            if init:
                return LlamaForCausalLM.from_pretrained(pretrain_dir, config=hf_config)
            else:
                return LlamaForCausalLM(config=hf_config).bfloat16()
        elif isinstance(hf_config, transformers.BertConfig):
            hf_config.use_ft_flash_attn = self.use_ft_flash_attn
            self.logger.info(f'Using flash attention {hf_config.use_ft_flash_attn} for bert')
            self.logger.info(f'Init {init} for bert')
            if init:
                return BertModel.from_pretrained(pretrain_dir, config=hf_config)
            else:
                return BertModel(config=hf_config).bfloat16()
        else:
            return AutoModelForCausalLM.from_pretrained(
                self.local_dir, config=hf_config
            )

    def forward(self, interaction):
        items, neg_items, masked_index = interaction  # [batch, 2, seq_len]    #[batch, max_seq_len-1]
        if self.num_negatives:
            neg_items = torch.randint(
                low=1,
                high=self.item_num,
                size=(items.size(0), items.size(1) - 1, self.num_negatives),
                dtype=items.dtype,
                device=items.device,
            )

        pos_items_embs = self.item_id_proj_tower(self.item_embedding(items))  # [batch, 2, max_seq_len+1, dim]
        neg_items_embs = self.item_id_proj_tower(self.item_embedding(neg_items))  # [batch, 2, max_seq_len+1, dim]

        input_emb = pos_items_embs[:, :-1, :]  # [batch, max_seq_len, dim]
        target_pos_embs = pos_items_embs[:, 1:, :]  # [batch, max_seq_len, dim]
        neg_embedding_all = neg_items_embs  # [batch, max_seq_len, dim]
        output_embs = self.user_llm(inputs_embeds=input_emb, attention_mask=masked_index).hidden_states[-1]

        with torch.no_grad():
            self.logit_scale.clamp_(0, np.log(100))
        logit_scale = self.logit_scale.exp()
        output_embs = output_embs / output_embs.norm(dim=-1, keepdim=True)
        target_pos_embs = target_pos_embs / target_pos_embs.norm(dim=-1, keepdim=True)
        neg_embedding_all = neg_embedding_all / neg_embedding_all.norm(dim=-1, keepdim=True)
        pos_logits = F.cosine_similarity(output_embs, target_pos_embs, dim=-1).unsqueeze(-1)
        if self.num_negatives:
            neg_logits = F.cosine_similarity(output_embs.unsqueeze(2), neg_embedding_all, dim=-1)
            fix_logits = F.cosine_similarity(target_pos_embs.unsqueeze(2), neg_embedding_all, dim=-1)
        else:
            D = neg_embedding_all.size(-1)
            neg_embedding_all = all_gather(neg_embedding_all, sync_grads=True).reshape(-1, D)  # [num, dim]
            neg_embedding_all = neg_embedding_all.transpose(-1, -2)
            neg_logits = torch.matmul(output_embs, neg_embedding_all)
            fix_logits = torch.matmul(target_pos_embs, neg_embedding_all)
        neg_logits[fix_logits > self.nce_thres] = torch.finfo(neg_logits.dtype).min
        logits = torch.cat([pos_logits, neg_logits], dim=-1)
        logits = logits[masked_index.bool()] * logit_scale
        labels = torch.zeros(logits.size(0), device=logits.device, dtype=torch.int64)
        model_out = {}
        model_out['loss'] = F.cross_entropy(logits, labels)
        model_out['nce_samples'] = (logits > torch.finfo(logits.dtype).min/100).sum(dim=1).float().mean()
        for k in [1, 5, 10, 50, 100]:
            if k > logits.size(1):
                break
            indices = logits.topk(k, dim=1).indices
            model_out[f"nce_top{k}_acc"] = labels.view(-1, 1).eq(indices).any(dim=1).float().mean()
        return model_out

    @torch.no_grad()
    def predict(self, item_seq, time_seq, item_feature):

        item_emb = self.item_id_proj_tower(self.item_embedding(item_seq))
        attention_mask = (item_seq > 0).int()
        output_embs = self.user_llm(inputs_embeds=item_emb, attention_mask=attention_mask).hidden_states[-1]
        seq_output = output_embs[:, -1]
        seq_output = seq_output / seq_output.norm(dim=-1, keepdim=True)

        scores = torch.matmul(seq_output, item_feature.t())
        return scores

    @torch.no_grad()
    def compute_item_all(self):
        weight = self.item_id_proj_tower(self.item_embedding(torch.arange(self.item_num, device=self.item_embedding.weight.device)))
        return weight / weight.norm(dim=-1, keepdim=True)

    def get_attention_mask(self, item_seq, bidirectional=False):
        """Generate left-to-right uni-directional or bidirectional attention mask for multi-head attention."""
        attention_mask = (item_seq != 0)
        extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)  # torch.bool
        if not bidirectional:
            extended_attention_mask = torch.tril(extended_attention_mask.expand((-1, -1, item_seq.size(-1), -1)))
        extended_attention_mask = torch.where(extended_attention_mask, 0., -1e9)
        return extended_attention_mask
