import torch
import torch.nn as nn
from torch import Tensor
import copy
from typing import Iterator, Sequence


class QNetwork(nn.Module):

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dims: Sequence[int] = (256,256),
    ) -> None:
        super().__init__()

        layers: list[nn.Module] = []
        last = state_dim + action_dim
        for h in hidden_dims:
            layers.append(nn.Linear(last, h))
            layers.append(nn.ReLU(inplace=True))
            last = h
        layers.append(nn.Linear(last, 1))
        self.net = nn.Sequential(*layers)

        final = self.net[-1]
        nn.init.uniform_(final.weight, -3e-3, 3e-3)
        nn.init.uniform_(final.bias, -3e-3, 3e-3)

    def forward(self, state: Tensor, action: Tensor) -> Tensor:
        x = torch.cat([state, action], dim=-1)
        return self.net(x)

class TwinCritic(nn.Module):

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dims: Sequence[int] = (256,256),
    ) -> None:
        super().__init__()

        self.q1 = QNetwork(state_dim, action_dim, hidden_dims)
        self.q2 = QNetwork(state_dim, action_dim, hidden_dims)

        self.q1_target = copy.deepcopy(self.q1)
        self.q2_target = copy.deepcopy(self.q2)

        for p in self.q1_target.parameters():
            p.requires_grad_(False)
        for p in self.q2_target.parameters():
            p.requires_grad_(False)

    def forward(self,state: Tensor, action: Tensor) -> tuple[Tensor, Tensor]:
        return self.q1(state,action), self.q2(state,action)

    def q_min(self, state: Tensor, action: Tensor) -> Tensor:

    @torch.no_grad()
    def q_target_min(self, next_state: Tensor, next_action: Tensor) -> Tensor:
        q1_t= self.q1_target(next_state, next_action)
        q2_t= self.q2_target(next_state, next_action)
        return torch.min(q1_t, q2_t)

    @torch.no_grad()
    def soft_update(self, tau: float) -> None:
        for online, target in (
            (self.q1, self.q1_target),
            (self.q2, self.q2_target),
        ):
            for p_online, p_target in zip(
                online.parameters(),target.parameters()
            ):
                p_target.data.mul_(1.0 - tau).add_(p_online.data,alpha=tau)

    @torch.no_grad()
    def hard_update(self) -> None:
        self.q1_target.load_state_dict(self.q1.state_dict())
        self.q2_target.load_state_dict(self.q2.state_dict())

   
    def online_parameters(self) -> Iterator[nn.Parameter]:
        yield from self.q1.parameters()
        yield from self.q2.parameters()
