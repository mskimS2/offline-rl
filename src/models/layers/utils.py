from torch import nn
from torch.nn import Module
from typing import Union


def get_activation(module: Union[str, nn.Module], *args, **kwargs) -> nn.Module:
    if isinstance(module, str):
        if hasattr(nn, module):
            return getattr(nn, module)(*args, **kwargs)
        raise ValueError(f"Unsupported activation function: {module}")
    
    elif isinstance(module, type) and issubclass(module, nn.Module):
        return module(*args, **kwargs)
    
    elif isinstance(module, nn.Module):
        return module

    raise ValueError(f"Unsupported activation type: {module}")
