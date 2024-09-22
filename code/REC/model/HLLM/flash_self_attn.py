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

from typing import Optional

import torch
import torch.nn.functional as F
from flash_attn.flash_attn_interface import (
    flash_attn_qkvpacked_func,
    flash_attn_varlen_qkvpacked_func,
)


def flash_self_attention(
    qkv: torch.Tensor,
    causal: bool = False,
    cu_seqlens: Optional[torch.Tensor] = None,
    max_seqlen: Optional[int] = None,
    softmax_scale: Optional[float] = None,
    attention_dropout: float = 0.0,
    training: bool = False,
):
    """Implements the multihead softmax attention.
    Modified from https://github.com/Dao-AILab/flash-attention/blob/v2.0.4/flash_attn/modules/mha.py#L35-L84
    Arguments
    ---------
        qkv: The tensor containing the query, key, and value.
            If cu_seqlens is None and max_seqlen is None, then qkv has shape (B, S, 3, H, D).
            If cu_seqlens is not None and max_seqlen is not None, then qkv has shape
            (total, 3, H, D), where total is the sum of the sequence lengths in the batch.
        causal: if passed, will override self.causal
        cu_seqlens: (batch_size + 1,), dtype torch.int32. The cumulative sequence lengths
            of the sequences in the batch, used to index into qkv.
        max_seqlen: int. Maximum sequence length in the batch.
    Returns:
    --------
        out: (total, H, D) if cu_seqlens is not None and max_seqlen is not None,
            else (B, S, H, D).
    """
    assert qkv.dtype in [torch.float16, torch.bfloat16]
    assert qkv.is_cuda
    unpadded = cu_seqlens is not None
    if unpadded:
        assert cu_seqlens.dtype == torch.int32
        assert max_seqlen is not None
        assert isinstance(max_seqlen, int)
        return flash_attn_varlen_qkvpacked_func(
            qkv,
            cu_seqlens,
            max_seqlen,
            attention_dropout if training else 0.0,
            softmax_scale=softmax_scale,
            causal=causal,
        )
    else:
        return flash_attn_qkvpacked_func(
            qkv,
            attention_dropout if training else 0.0,
            softmax_scale=softmax_scale,
            causal=causal,
        )


def compute_flash_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
    cu_input_lens: Optional[torch.Tensor] = None,
    causal: bool = True,
    training: bool = False,
    attention_dropout: float = 0.0,
):
    """Modified from https://github.com/LAION-AI/Open-Assistant/blob/main/model/model_training/models/patching_utils.py"""
    # q, k, v: [bs, seq_len, num_attention_heads, attn_head_size]
    # attention_mask (float): [bs, seq_len]
    batch_size, max_len = q.size(0), q.size(1)

    qkv = torch.stack([q, k, v], dim=2)  # [bs, seq_len, 3, num_attention_heads, attn_head_size]

    if cu_input_lens is not None:
        qkv.squeeze_(0)
        cu_seqlens = F.pad(cu_input_lens.cumsum(dim=0, dtype=torch.int32), (1, 0))
        max_seqlen = cu_input_lens.max().item()
        out = flash_self_attention(
            qkv,
            causal=causal,
            cu_seqlens=cu_seqlens,
            max_seqlen=max_seqlen,
            training=training,
            attention_dropout=attention_dropout,
        )
        return out
    elif attention_mask is None:
        return flash_self_attention(qkv, causal=causal, training=training, attention_dropout=attention_dropout)
    else:
        # Limitation: non-contiguous attention mask will not be handled correctly
        # model will be able to pay attention between the first and last non-masked token, i.e. left- and right-side padding is supported.
        cur_mask = attention_mask >= 0
        csums = cur_mask.cumsum(dim=1, dtype=torch.int32)
        ends = csums.argmax(dim=1) + 1
        starts = ends - csums.max(dim=1).values
        seqlens = ends - starts

        # qkv = torch.cat([qkv[i, starts[i] : ends[i]] for i in range(batch_size)], dim=0)
        qkv = qkv.view(batch_size*max_len, *qkv.size()[2:])
        cur_mask = cur_mask.flatten().nonzero().squeeze()
        qkv = qkv[cur_mask]
        cu_seqlens = F.pad(seqlens.cumsum(dim=0, dtype=torch.int32), (1, 0))
        max_seqlen = seqlens.max().item()

        out = flash_self_attention(
            qkv,
            causal=causal,
            cu_seqlens=cu_seqlens,
            max_seqlen=max_seqlen,
            training=training,
            attention_dropout=attention_dropout
        )
        # out: [num_unmasked_tokens, num_attention_heads, attn_head_size]

        seqs = [out[start:end] for start, end in zip(cu_seqlens[:-1], cu_seqlens[1:])]
        # stack and pad sequences together
        padded_seqs = [
            F.pad(
                seqs[i],
                (0, 0) * (seqs[i].dim() - 1) + (starts[i], max_len - ends[i]),
                value=0.0,
            )
            for i in range(batch_size)
        ]
        out = torch.stack(padded_seqs)
        return out
