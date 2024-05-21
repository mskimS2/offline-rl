from torch import nn
from typing import List
from utils import get_activation


class MLP(nn.Sequential):
    def __init__(
        self,
        hidden_dim: List[int],
        hidden_act: nn.Module,
        out_act: nn.Module = nn.Identity(),
    ):
        super(MLP, self).__init__()

        for i, (in_dim, out_dim) in enumerate(zip(hidden_dim[:-1], hidden_dim[1:])):
            act = hidden_act if i < len(hidden_dim) - 2 else out_act
            self.add_module(f"linear_{i}", nn.Linear(in_dim, out_dim))
            self.add_module(f"activation_{i}", get_activation(act))

    def __repr__(self):
        return f"Num Params: {sum(p.numel() for p in self.parameters())}"