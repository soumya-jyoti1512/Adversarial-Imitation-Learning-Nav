import torch
from torch import Tensor


class HybridReward:
    def __init__(
        self,
        lambda_goal: float = 0.3,
        lambda_collision: float = 0.5,
        collision_penalty: float = 1.0,
        collision_threshold: float = 0.2,
        lidar_slice: slice= slice(0, 20),
        goal_slice: slice= slice(20, 22),
    )

        self.lambda_goal = float(lambda_goal)
        self.lambda_collision= float(lambda_collision)
        self.collision_penalty= float(collision_penalty)
        self.collision_threshold = float(collision_threshold)
        self.lidar_slice = lidar_slice
        self.goal_slice= goal_slice

    
    def r_goal(self, state: Tensor, next_state: Tensor) -> Tensor:
        d_curr = torch.linalg.vector_norm(
            state[..., self.goal_slice], dim=-1, keepdim=True
        )
        d_next = torch.linalg.vector_norm(
            next_state[..., self.goal_slice], dim=-1, keepdim=True
        )
        return d_curr - d_next

    def r_collision(self, next_state: Tensor) -> Tensor:
        lidar = next_state[...,self.lidar_slice]
        min_dist = lidar.min(dim=-1,keepdim=True).values
        unsafe = (min_dist < self.collision_threshold).to(next_state.dtype)
        return -self.collision_penalty * unsafe

    
    def compute(
        self,
        state: Tensor,
        next_state: Tensor,
        r_gail: Tensor,
    ) -> dict[str, Tensor]:
        r_goal_t = self.r_goal(state, next_state)
        r_coll_t = self.r_collision(next_state)
        r_total = (
            r_gail
            + self.lambda_goal * r_goal_t
            + self.lambda_collision * r_coll_t
        )
        return {
            "r_gail": r_gail,
            "r_goal": r_goal_t,
            "r_collision": r_coll_t,
            "r_total": r_total,
        }

    def __repr__(self) -> str:
        return (
            f"HybridReward(λ_goal={self.lambda_goal}, "
            f"λ_collision={self.lambda_collision}, "
            f"C={self.collision_penalty}, "
            f"ε={self.collision_threshold} m)"
        )
