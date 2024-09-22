# Copyright (c) 2024 westlake-repl
# Copyright (c) 2024 Bytedance Ltd. and/or its affiliate
# SPDX-License-Identifier: MIT
# This file has been modified by Junyi Chen.
#
# Original file was released under MIT, with the full license text
# available at https://choosealicense.com/licenses/mit/.
#
# This modified file is released under the same license.

import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
from logging import getLogger

from REC.model.layers import TransformerEncoder
from REC.utils.enum_type import InputType
from REC.model.basemodel import BaseModel, all_gather


class SASRec(BaseModel):
    input_type = InputType.SEQ

    def __init__(self, config, dataload):
        super(SASRec, self).__init__()
        self.logger = getLogger()
        # load parameters info
        self.n_layers = config['n_layers']
        self.n_heads = config['n_heads']
        self.hidden_size = config['embedding_size']  # same as embedding_size
        self.inner_size = config['inner_size']  # the dimensionality in feed-forward layer
        self.inner_size *= self.hidden_size
        self.hidden_dropout_prob = config['hidden_dropout_prob']
        self.attn_dropout_prob = config['attn_dropout_prob']
        self.hidden_act = config['hidden_act']
        self.layer_norm_eps = config['layer_norm_eps']

        self.initializer_range = config['initializer_range']
        self.max_seq_length = config['MAX_ITEM_LIST_LENGTH']
        self.item_num = dataload.item_num
        # define layers and loss
        self.item_embedding = nn.Embedding(self.item_num, self.hidden_size, padding_idx=0)
        self.position_embedding = nn.Embedding(self.max_seq_length, self.hidden_size)
        self.trm_encoder = TransformerEncoder(
            n_layers=self.n_layers,
            n_heads=self.n_heads,
            hidden_size=self.hidden_size,
            inner_size=self.inner_size,
            hidden_dropout_prob=self.hidden_dropout_prob,
            attn_dropout_prob=self.attn_dropout_prob,
            hidden_act=self.hidden_act,
            layer_norm_eps=self.layer_norm_eps
        )

        self.LayerNorm = nn.LayerNorm(self.hidden_size, eps=self.layer_norm_eps)
        self.dropout = nn.Dropout(self.hidden_dropout_prob)

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

        # parameters initialization
        self.apply(self._init_weights)

    def _init_weights(self, module):
        """ Initialize the weights """
        if isinstance(module, (nn.Linear, nn.Embedding)):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0, std=self.initializer_range)
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()

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

        pos_items_embs = self.item_embedding(items)  # [batch, 2, max_seq_len+1, dim]
        neg_items_embs = self.item_embedding(neg_items)  # [batch, 2, max_seq_len+1, dim]

        input_emb = pos_items_embs[:, :-1, :]  # [batch, max_seq_len, dim]
        target_pos_embs = pos_items_embs[:, 1:, :]  # [batch, max_seq_len, dim]
        neg_embedding_all = neg_items_embs  # [batch, max_seq_len, dim]

        position_ids = torch.arange(masked_index.size(1), dtype=torch.long, device=masked_index.device)
        position_ids = position_ids.unsqueeze(0).expand_as(masked_index)
        position_embedding = self.position_embedding(position_ids)
        input_emb = input_emb + position_embedding
        input_emb = self.LayerNorm(input_emb)
        input_emb = self.dropout(input_emb)

        extended_attention_mask = self.get_attention_mask(masked_index, bidirectional=False)

        output_embs = self.trm_encoder(input_emb, extended_attention_mask, output_all_encoded_layers=False)  # [batch, max_seq_len-1, dim]
        output_embs = output_embs[-1]

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

        position_ids = torch.arange(item_seq.size(1), dtype=torch.long, device=item_seq.device)
        position_ids = position_ids.unsqueeze(0).expand_as(item_seq)
        position_embedding = self.position_embedding(position_ids)

        item_emb = self.item_embedding(item_seq)
        input_emb = item_emb + position_embedding
        input_emb = self.LayerNorm(input_emb)
        input_emb = self.dropout(input_emb)

        extended_attention_mask = self.get_attention_mask(item_seq, bidirectional=False)

        output = self.trm_encoder(input_emb, extended_attention_mask, output_all_encoded_layers=False)
        output_embs = output[-1]
        seq_output = output_embs[:, -1]
        seq_output = seq_output / seq_output.norm(dim=-1, keepdim=True)

        scores = torch.matmul(seq_output, item_feature.t())
        return scores

    @torch.no_grad()
    def compute_item_all(self):
        weight = self.item_embedding.weight
        return weight / weight.norm(dim=-1, keepdim=True)

    def get_attention_mask(self, item_seq, bidirectional=False):
        """Generate left-to-right uni-directional or bidirectional attention mask for multi-head attention."""
        attention_mask = (item_seq != 0)
        extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)  # torch.bool
        if not bidirectional:
            extended_attention_mask = torch.tril(extended_attention_mask.expand((-1, -1, item_seq.size(-1), -1)))
        extended_attention_mask = torch.where(extended_attention_mask, 0., -1e9)
        return extended_attention_mask
