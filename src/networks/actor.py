import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.distributions import Normal
import math
from typing import Sequence

class GaussianActor(nn.Module):

    LOG_STD_MIN: float= -20.0
    LOG_STD_MAX: float= 2.0

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dims: Sequence[int]= (256, 256),
        action_scale: float | Tensor= 1.0,
        action_bias: float | Tensor= 0.0,
    ) -> None:
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim

        layers: list[nn.Module] = []
        last = state_dim
        for h in hidden_dims:
            layers.append(nn.Linear(last, h))
            layers.append(nn.ReLU(inplace=True))
            last = h
        self.trunk = nn.Sequential(*layers)

        self.mean_head =nn.Linear(last, action_dim)
        self.log_std_head = nn.Linear(last, action_dim)

        for head in (self.mean_head, self.log_std_head):
            nn.init.uniform_(head.weight, -3e-3,3e-3)
            nn.init.uniform_(head.bias,-3e-3, 3e-3)

        scale = torch.as_tensor(action_scale, dtype=torch.float32)
        bias = torch.as_tensor(action_bias, dtype=torch.float32)
        if scale.ndim == 0:
            scale = scale.expand(action_dim).clone()
        if bias.ndim == 0:
            bias = bias.expand(action_dim).clone()
        self.register_buffer("action_scale", scale)
        self.register_buffer("action_bias", bias)

    def _forward_dist(self, state: Tensor) -> tuple[Tensor, Tensor]:
        h = self.trunk(state)
        mean = self.mean_head(h)
        log_std = self.log_std_head(h).clamp(self.LOG_STD_MIN, self.LOG_STD_MAX)
        return mean, log_std

   
    def forward(
        self, state: Tensor, *, deterministic: bool = False
    ) -> tuple[Tensor, Tensor]:
        
        mean, log_std = self._forward_dist(state)

        if deterministic:
            u = mean
            log_prob = torch.zeros(state.shape[0], 1, device=state.device)
        else:
            std = log_std.exp()
            normal = Normal(mean, std)
           
            u = normal.rsample()

            log_prob_u = normal.log_prob(u).sum(dim=-1, keepdim=True)

            squash_correction = (
                2.0 * (math.log(2.0) - u - F.softplus(-2.0 * u))
            ).sum(dim=-1, keepdim=True)

            log_prob = log_prob_u - squash_correction

        squashed = torch.tanh(u)
        action = self.action_scale * squashed + self.action_bias
        return action, log_prob

    @torch.no_grad()
    def act(self, state: Tensor, *, deterministic: bool = False) -> Tensor:
        was_training = self.training
        self.eval()
        action, _ = self.forward(state, deterministic=deterministic)
        if was_training:
            self.train()
        return action
