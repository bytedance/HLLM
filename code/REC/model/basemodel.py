# Copyright (c) 2024 westlake-repl
# SPDX-License-Identifier: MIT

import numpy as np
import torch
import torch.nn as nn

from REC.utils import set_color


def all_gather(data,
               group=None,
               sync_grads=False):
    group = group if group is not None else torch.distributed.group.WORLD
    if torch.distributed.get_world_size() > 1:
        from torch.distributed import nn
        if sync_grads:
            return torch.stack(nn.functional.all_gather(data, group=group), dim=0)
        with torch.no_grad():
            return torch.stack(nn.functional.all_gather(data, group=group), dim=0)
    else:
        return data.unsqueeze(0)


def l2_norm(x, eps=1e-6):
    x = x / torch.clamp(
        torch.linalg.norm(x, ord=2, dim=-1, keepdim=True),
        min=eps,
    )
    return x


class BaseModel(nn.Module):

    def __init__(self):
        super(BaseModel, self).__init__()

    def load_weights(self, path):
        checkpoint = torch.load(path, map_location='cpu')
        pretrained_dicts = checkpoint['state_dict']
        self.load_state_dict({k.replace('item_embedding.rec_fc', 'visual_encoder.item_encoder.fc'): v for k, v in pretrained_dicts.items()}, strict=False)

    def __str__(self):
        """
        Model prints with number of trainable parameters
        """
        model_parameters = filter(lambda p: p.requires_grad, self.parameters())
        params = sum([np.prod(p.size()) for p in model_parameters])
        return super().__str__() + set_color('\nTrainable parameters', 'blue') + f': {params}'
