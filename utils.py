import scipy
import numpy as np
from torch import nn


def combined_shape(length: int, shape: int = None):
    if shape is None:
        return (length,)
    return (length, shape) if np.isscalar(shape) else (length, *shape)


def count_vars(m: nn.Module):
    return sum([np.prod(p.shape) for p in m.parameters()])


def discount_cumsum(x, gamma: float):
    return scipy.signal.lfilter([1], [1, float(-gamma)], x[::-1], axis=0)[::-1]


def soft_update(trg: nn.Module, src: nn.Module, tau: float) -> None:
    for tp, sp in zip(trg.parameters(), src.parameters()):
        tp.data.copy_((1 - tau) * tp.data + tau * sp.data)
