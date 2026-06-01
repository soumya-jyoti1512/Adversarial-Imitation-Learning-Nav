from pathlib import Path
from typing import TypedDict
import h5py
import numpy as np
import torch
from torch import Tensor


EXPECTED_FORMAT_VERSION = "1.0"


class ExpertBatch(TypedDict):
    state: Tensor  
    action: Tensor  
    next_state: Tensor  
    done: Tensor  


def _decode_attr(value):
    if isinstance(value, bytes):
        return value.decode("utf-8")
    return value


class ExpertBuffer:

    def __init__(
        self,
        path: str | Path,
        device: str | torch.device="cpu",
        seed: int | None= None,
        strict: bool= True,
    ) 

        with h5py.File(path, "r") as f:
            required = {"states", "actions", "next_states","dones"}
            missing =required - set(f.keys())
            if strict:
                version = _decode_attr(f.attrs.get("format_version", ""))
                

            self._states= f["states"][...].astype(np.float32, copy=False)
            self._actions= f["actions"][...].astype(np.float32, copy=False)
            self._next_states= f["next_states"][...].astype(np.float32, copy=False)
            dones= f["dones"][...].astype(np.float32, copy=False)
            if dones.ndim == 1:
                dones = dones[:, None]
            self._dones = dones

            if "episode_starts" in f:
                self._episode_starts = f["episode_starts"][...].astype(np.int64)
            else:
                self._episode_starts = None


        N, state_dim = self._states.shape
        _, action_dim = self._actions.shape

        self.path= path
        self.device= torch.device(device)
        self.state_dim = int(state_dim)
        self.action_dim= int(action_dim)
        self._rng = np.random.default_rng(seed)

    def sample(self, batch_size: int) -> ExpertBatch:
        idx = self._rng.integers(0, len(self), size=batch_size)
        return ExpertBatch(
            state= torch.from_numpy(self._states[idx]).to(self.device),
            action=torch.from_numpy(self._actions[idx]).to(self.device),
            next_state= torch.from_numpy(self._next_states[idx]).to(self.device),
            done=torch.from_numpy(self._dones[idx]).to(self.device),
        )

    def __len__(self) -> int:
        return self._states.shape[0]

    @property
    def num_episodes(self) -> int:
        if self._episode_starts is None:
            return 0
        return int(self._episode_starts.shape[0])

    @property
    def avg_episode_length(self) -> float:
        if self.num_episodes==0:
            return float("nan")
        return len(self) / self.num_episodes

    @staticmethod
    def write_hdf5(
        path: str | Path,
        states: np.ndarray,
        actions: np.ndarray,
        next_states: np.ndarray,
        dones: np.ndarray,
        episode_starts: np.ndarray | None = None,
    ) -> None:
        states= np.asarray(states, dtype=np.float32)
        actions= np.asarray(actions, dtype=np.float32)
        next_states= np.asarray(next_states, dtype=np.float32)
        dones= np.asarray(dones, dtype=np.float32).reshape(-1, 1)


        N, state_dim = states.shape
        action_dim = actions.shape[1]

        num_episodes = 0
        if episode_starts is not None:
            episode_starts = np.asarray(episode_starts, dtype=np.int64)
            num_episodes = int(episode_starts.shape[0])

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with h5py.File(path, "w") as f:
            f.create_dataset("states", data=states, compression="gzip")
            f.create_dataset("actions", data=actions, compression="gzip")
            f.create_dataset("next_states", data=next_states, compression="gzip")
            f.create_dataset("dones", data=dones, compression="gzip")
            if episode_starts is not None:
                f.create_dataset("episode_starts", data=episode_starts)
            f.attrs["state_dim"]= int(state_dim)
            f.attrs["action_dim"]= int(action_dim)
            f.attrs["num_episodes"]= num_episodes
            f.attrs["format_version"]= EXPECTED_FORMAT_VERSION
