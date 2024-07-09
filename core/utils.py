"""
Date: 2024-03-24 16:43:34
LastEditors: Jiaqi Gu && jiaqigu@asu.edu
LastEditTime: 2024-03-24 16:43:34
FilePath: /SparseTeMPO/core/models/utils.py
"""

from torch import nn
from torch.nn.modules.batchnorm import _BatchNorm

def hidden_register_hook(m, input, output):
    m._recorded_hidden = output


def register_hidden_hooks(model):
    for name, m in model.named_modules():
        if isinstance(m, (nn.ReLU, nn.GELU, nn.Hardswish, nn.ReLU6)):
            m.register_forward_hook(hidden_register_hook)


def get_parameter_group(model, weight_decay=0.0):
    """set weigh_decay to Normalization layers to 0"""
    all_parameters = set(model.parameters())
    group_no_decay = set()
    group_decay_name = []
    for name, m in model.named_modules():
        if isinstance(m, _BatchNorm):
            if m.weight is not None:
                group_no_decay.add(m.weight)
            if m.bias is not None:
                group_no_decay.add(m.bias)
            group_decay_name.append(name)
        if hasattr(m, "alpha") and m.alpha is not None:
            group_no_decay.add(m.alpha)
            group_decay_name.append(name)
        if hasattr(m, "zero_point") and m.zero_point is not None:
            group_no_decay.add(m.zero_point)
            group_decay_name.append(name)
    print(f"set weight_decay to 0 for {group_decay_name}")
    group_decay = all_parameters - group_no_decay
    

    return [
        {"params": list(group_no_decay), "weight_decay": 0.0},
        {"params": list(group_decay), "weight_decay": weight_decay},
    ]