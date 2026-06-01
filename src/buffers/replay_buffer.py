import numpy as np
import torch
from torch import Tensor
from pathlib import Path
from typing import TypedDict

class TransitionBatch:
    state:      Tensor  
    action:     Tensor 
    next_state: Tensor  
    done:       Tensor  


def _to_np(arr, expected_dim: int, name: str) -> np.ndarray:
    if isinstance(arr, torch.Tensor):
        arr = arr.detach().cpu().numpy()
    arr = np.asarray(arr, dtype=np.float32).reshape(-1)
    if arr.shape[0] != expected_dim:
        raise ValueError(
            f"{name} has size {arr.shape[0]}, expected {expected_dim}."
        )
    return arr


class ReplayBuffer:

    def __init__(
        self,
        capacity: int,
        state_dim: int,
        action_dim: int,
        device: str | torch.device= "cpu",
        seed: int | None = None,
    ) 

        self.capacity= int(capacity)
        self.state_dim= int(state_dim)
        self.action_dim= int(action_dim)
        self.device= torch.device(device)

        self._states= np.empty((capacity, state_dim),  dtype=np.float32)
        self._actions= np.empty((capacity, action_dim), dtype=np.float32)
        self._next_states= np.empty((capacity, state_dim), dtype=np.float32)
        self._dones= np.empty((capacity, 1), dtype=np.float32)

        self._pos= 0    
        self._size= 0    
        self._rng= np.random.default_rng(seed)

    
    def add(
        self,
        state,
        action,
        next_state,
        done: bool,
    ) -> None:
        self._states[self._pos]= _to_np(state,      self.state_dim,  "state")
        self._actions[self._pos]= _to_np(action,     self.action_dim, "action")
        self._next_states[self._pos]= _to_np(next_state, self.state_dim,  "next_state")
        self._dones[self._pos, 0]= float(bool(done))

        self._pos= (self._pos + 1) % self.capacity
        self._size = min(self._size + 1, self.capacity)

   
    def sample(self, batch_size: int) -> TransitionBatch:

        idx = self._rng.integers(0, self._size, size=batch_size)

        return TransitionBatch(
            state= torch.from_numpy(self._states[idx]).to(self.device),
            action= torch.from_numpy(self._actions[idx]).to(self.device),
            next_state= torch.from_numpy(self._next_states[idx]).to(self.device),
            done= torch.from_numpy(self._dones[idx]).to(self.device),
        )

    def __len__(self) -> int:
        return self._size

    @property
    def is_full(self) -> bool:
        return self._size == self.capacity

    def save(self, path: str | Path) -> None:
        np.savez(
            Path(path),
            states=self._states,
            actions=self._actions,
            next_states=self._next_states,
            dones=self._dones,
            pos=np.array(self._pos, dtype=np.int64),
            size=np.array(self._size, dtype=np.int64),
        )

    @classmethod
    def load(
        cls,
        path: str | Path,
        device: str | torch.device= "cpu",
        seed: int | None = None,
    ) -> "ReplayBuffer":
        data = np.load(Path(path))
        capacity, state_dim = data["states"].shape
        _, action_dim = data["actions"].shape

        buf = cls(
            capacity=int(capacity),
            state_dim=int(state_dim),
            action_dim=int(action_dim),
            device=device,
            seed=seed,
        )
        buf._states= data["states"].copy()
        buf._actions= data["actions"].copy()
        buf._next_states= data["next_states"].copy()
        buf._dones= data["dones"].copy()
        buf._pos= int(data["pos"])
        buf._size= int(data["size"])
        return buf
