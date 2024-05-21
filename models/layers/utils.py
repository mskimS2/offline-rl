from torch import nn
from torch.nn import Module
from typing import Union


def get_activation(module: Union[str, nn.Module], *args, **kwargs) -> nn.Module:
    if isinstance(module, (Module, nn.Module)):
        return module
    elif hasattr(nn, module):
        return getattr(nn, module)

    return getattr(nn, module)