import torch
from torch import nn
from typing import List


class MLP(nn.Module):
    def __init__(
        self,
        dim_list: List[int],
        activation: nn.Module,
        output_activation=nn.Identity,
    ):
        super(MLP, self).__init__()

        layers = []
        for j in range(len(dim_list) - 1):
            act = activation if j < len(dim_list) - 2 else output_activation
            layers += [nn.Linear(dim_list[j], dim_list[j + 1]), act()]

        self.mlp = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.mlp(x)
