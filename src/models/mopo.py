import torch
import torch.nn.functional as F
from torch import nn
from models.layers.mlp import MLP


class Swish(nn.Module):
    def __init__(self):
        super(Swish, self).__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.sigmoid(x)


class EnsembleModel(nn.Module):
    def __init__(
        self,
        obs_dim,
        act_dim,
        hidden_sizes,
        activation=Swish,
        output_activation=nn.Identity,
        reward_dim=1,
        ensemble_size=7,
        num_elite=5,
        decay_weights=None,
        device=torch.device("cpu"),
    ):
        super(EnsembleModel, self).__init__()

        self.out_dim = obs_dim + reward_dim

        self.ensemble_models = [
            MLP([obs_dim + act_dim] + list(hidden_sizes) + [self.out_dim * 2], activation, output_activation)
            for _ in range(ensemble_size)
        ]
        for i in range(ensemble_size):
            self.add_module("model_{}".format(i), self.ensemble_models[i])

        self.obs_dim = obs_dim
        self.action_dim = act_dim
        self.num_elite = num_elite
        self.ensemble_size = ensemble_size
        self.decay_weights = decay_weights
        self.elite_model_idxes = torch.tensor([i for i in range(num_elite)])
        self.max_logvar = nn.Parameter((torch.ones((1, self.out_dim)).float() / 2).to(device), requires_grad=True)
        self.min_logvar = nn.Parameter((-torch.ones((1, self.out_dim)).float() * 10).to(device), requires_grad=True)
        self.register_parameter("max_logvar", self.max_logvar)
        self.register_parameter("min_logvar", self.min_logvar)
        self.device = device

    def predict(self, input):
        # convert input to tensors
        if type(input) != torch.Tensor:
            if len(input.shape) == 1:
                input = torch.FloatTensor([input]).to(self.device)
            else:
                input = torch.FloatTensor(input).to(self.device)

        # predict
        if len(input.shape) == 3:
            model_outputs = [net(ip) for ip, net in zip(torch.unbind(input), self.ensemble_models)]
        elif len(input.shape) == 2:
            model_outputs = [net(input) for net in self.ensemble_models]
        predictions = torch.stack(model_outputs)

        mean = predictions[:, :, : self.out_dim]
        logvar = predictions[:, :, self.out_dim :]
        logvar = self.max_logvar - F.softplus(self.max_logvar - logvar)
        logvar = self.min_logvar + F.softplus(logvar - self.min_logvar)

        return mean, logvar

    def get_decay_loss(self):
        decay_losses = []
        for model_net in self.ensemble_models:
            curr_net_decay_losses = [
                decay_weight * torch.sum(torch.square(weight))
                for decay_weight, weight in zip(self.decay_weights, model_net.weights)
            ]
            decay_losses.append(torch.sum(torch.stack(curr_net_decay_losses)))
        return torch.sum(torch.stack(decay_losses))
