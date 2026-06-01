import torch
from torch import Tensor
from typing import Sequence
from src.networks.discriminator import Discriminator



class GAILTrainer:

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dims: Sequence[int]= (256, 256),
        lr: float = 3e-4,
        r1_coeff: float= 10.0,
        use_tanh: bool= True,
        device: str | torch.device = "cpu",
    ) 

        self.device = torch.device(device)
        self.r1_coeff= float(r1_coeff)

        self.discriminator= Discriminator(
            state_dim=state_dim,
            action_dim=action_dim,
            hidden_dims=hidden_dims,
            use_tanh=use_tanh,
        ).to(self.device)

        self.opt = torch.optim.Adam(self.discriminator.parameters(), lr=lr)
        self._update_count = 0

   
    def compute_reward(self, state: Tensor, action: Tensor) -> Tensor:
        return self.discriminator.reward(state, action)

    
    def update(
        self,
        expert_batch: dict[str, Tensor],
        agent_batch: dict[str, Tensor],
    ) -> dict[str, float]:
        diag = self.discriminator.loss(
            expert_state=expert_batch["state"],
            expert_action=expert_batch["action"],
            agent_state=agent_batch["state"],
            agent_action=agent_batch["action"],
        )
        bce_loss = diag["loss"]

        
        r1 = self.discriminator.r1_penalty(
            expert_state=expert_batch["state"],
            expert_action=expert_batch["action"],
            coeff=self.r1_coeff,
        )

        total_loss = bce_loss + r1

        self.opt.zero_grad(set_to_none=True)
        total_loss.backward()

        grad_norm =self._compute_grad_norm()

        self.opt.step()
        self._update_count += 1

        return {
            "bce_loss": float(bce_loss.detach()),
            "r1_penalty": float(r1.detach()),
            "total_loss": float(total_loss.detach()),
            "loss_expert":float(diag["loss_expert"]),
            "loss_agent": float(diag["loss_agent"]),
            "d_expert":float(diag["d_expert"]),
            "d_agent":float(diag["d_agent"]),
            "acc": float(diag["acc"]),
            "grad_norm":   grad_norm,
        }

    def _compute_grad_norm(self) -> float:
        total_sq = 0.0
        for p in self.discriminator.parameters():
            if p.grad is not None:
                total_sq += p.grad.detach().pow(2).sum().item()
        return total_sq ** 0.5

    @property
    def update_count(self) -> int:
        return self._update_count

    def state_dict(self) -> dict:
        return {
            "discriminator": self.discriminator.state_dict(),
            "opt":           self.opt.state_dict(),
            "update_count":  self._update_count,
        }

    def load_state_dict(self, sd: dict) -> None:
        self.discriminator.load_state_dict(sd["discriminator"])
        self.opt.load_state_dict(sd["opt"])
        self._update_count= int(sd.get("update_count", 0))
