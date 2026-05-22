from __future__ import annotations

from typing import Any, Optional

import numpy as np
import gymnasium as gym
from gymnasium import spaces


class ToyNavEnv(gym.Env):

    metadata = {"render_modes": ["rgb_array"]}

    def __init__(
        self,
        env_size: float = 5.4,
        num_obstacles: int = 5,
        obstacle_size_range: tuple[float, float] = (0.3, 0.6),
        robot_radius: float = 0.15,
        lidar_max_range: float = 3.0,
        lidar_num_beams: int = 20,
        goal_threshold: float = 0.3,
        v_max: float = 1.0,
        omega_max: float = 1.5,
        dt: float = 0.1,
        max_steps: int = 300,
        min_spawn_goal_dist: float = 2.0,
        seed: Optional[int] = None,
    ) -> None:
        super().__init__()

        # Geometry
        self.env_size = float(env_size)
        self.half = self.env_size / 2.0
        self.num_obstacles = int(num_obstacles)
        self.obs_size_lo, self.obs_size_hi = obstacle_size_range
        self.robot_radius = float(robot_radius)

        #LiDAR
        self.lidar_max_range = float(lidar_max_range)
        self.lidar_num_beams = int(lidar_num_beams)
        self.beam_angles = np.linspace(
            0.0, 2.0 * np.pi, self.lidar_num_beams, endpoint=False,
            dtype=np.float32,
        )

        #Dynamics
        self.v_max = float(v_max)
        self.omega_max = float(omega_max)
        self.dt = float(dt)
        self.max_steps = int(max_steps)
        self.goal_threshold = float(goal_threshold)
        self.min_spawn_goal_dist = float(min_spawn_goal_dist)

        #Gym spaces
        action_low = np.array([-v_max, -v_max, -omega_max], dtype=np.float32)
        action_high = -action_low
        self.action_space = spaces.Box(
            low=action_low, high=action_high, dtype=np.float32
        )
        obs_low = np.concatenate([
            np.zeros(self.lidar_num_beams, dtype=np.float32),
            np.full(2, -self.env_size, dtype=np.float32),
        ])
        obs_high = np.concatenate([
            np.full(self.lidar_num_beams, self.lidar_max_range, dtype=np.float32),
            np.full(2,  self.env_size, dtype=np.float32),
        ])
        self.observation_space = spaces.Box(
            low=obs_low, high=obs_high, dtype=np.float32
        )

        #RNG
        self._rng = np.random.default_rng(seed)

        # Per-episode state
        self.robot_pos = np.zeros(2, dtype=np.float32)
        self.robot_heading = 0.0
        self.goal = np.zeros(2, dtype=np.float32)
        self.obstacles = np.zeros((0, 4), dtype=np.float32)  # (cx, cy, hw, hh)
        self.step_count = 0

    # Episode setup
    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[dict] = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        if seed is not None:
            self._rng = np.random.default_rng(seed)

        self.obstacles = self._sample_obstacles()
        self.robot_pos = self._sample_free_position()
        self.robot_heading = float(self._rng.uniform(-np.pi, np.pi))
       
        for _ in range(50):
            candidate = self._sample_free_position()
            if np.linalg.norm(candidate - self.robot_pos) >= self.min_spawn_goal_dist:
                self.goal = candidate
                break
        else:
            self.goal = candidate

        self.step_count = 0
        return self._build_obs(), self._info()

    def _sample_obstacles(self) -> np.ndarray:
        obstacles = []
        attempts = 0
        spawn_margin = self.robot_radius + 0.05
        while len(obstacles) < self.num_obstacles and attempts < 500:
            attempts += 1
            side = float(self._rng.uniform(self.obs_size_lo, self.obs_size_hi))
            hw = hh = side / 2.0
            cx = float(self._rng.uniform(-self.half + hw + spawn_margin,
                                          self.half - hw - spawn_margin))
            cy = float(self._rng.uniform(-self.half + hh + spawn_margin,
                                          self.half - hh - spawn_margin))
            candidate = np.array([cx, cy, hw, hh], dtype=np.float32)
            if any(self._aabb_overlap(candidate, o, 0.1) for o in obstacles):
                continue
            obstacles.append(candidate)

        wt = 0.05 
        walls = [
            (-self.half - wt,  0.0, wt, self.half + 2 * wt),  # left
            ( self.half + wt,  0.0, wt, self.half + 2 * wt),  # right
            ( 0.0, -self.half - wt, self.half + 2 * wt, wt),  # bottom
            ( 0.0,  self.half + wt, self.half + 2 * wt, wt),  # top
        ]
        all_boxes = obstacles + [np.array(w, dtype=np.float32) for w in walls]
        return np.stack(all_boxes, axis=0)

    @staticmethod
    def _aabb_overlap(a: np.ndarray, b: np.ndarray, gap: float = 0.0) -> bool:
        """Return True if two (cx, cy, hw, hh) boxes overlap, inflated by gap."""
        return (
            abs(a[0] - b[0]) < a[2] + b[2] + gap
            and abs(a[1] - b[1]) < a[3] + b[3] + gap
        )

    def _sample_free_position(self) -> np.ndarray:
        """Rejection-sample a (x, y) at least robot_radius away from any box."""
        for _ in range(200):
            pos = self._rng.uniform(
                low=-self.half + self.robot_radius + 0.1,
                high= self.half - self.robot_radius - 0.1,
                size=2,
            ).astype(np.float32)
            if self._min_clearance(pos) > self.robot_radius + 0.05:
                return pos
        return np.zeros(2, dtype=np.float32)

    # Step
    def step(
        self, action: np.ndarray
    ) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        a = np.asarray(action, dtype=np.float32).reshape(-1)
        if a.shape != (3,):
            raise ValueError(f"action shape {a.shape}, expected (3,)")
        a = np.clip(a, self.action_space.low, self.action_space.high)
        vx_body, vy_body, omega = float(a[0]), float(a[1]), float(a[2])

        # Forward dynamics 
        c, s = np.cos(self.robot_heading), np.sin(self.robot_heading)
        vx_world = vx_body * c - vy_body * s
        vy_world = vx_body * s + vy_body * c
        self.robot_pos = self.robot_pos + np.array(
            [vx_world * self.dt, vy_world * self.dt], dtype=np.float32
        )
        self.robot_heading = self._wrap_angle(self.robot_heading + omega * self.dt)
        self.step_count += 1

        #Termination logic
        dist_to_goal = float(np.linalg.norm(self.goal - self.robot_pos))
        reached_goal = dist_to_goal < self.goal_threshold
        collided = self._min_clearance(self.robot_pos) < self.robot_radius
        terminated = bool(reached_goal or collided)
        truncated = (not terminated) and self.step_count >= self.max_steps

   
        r_env = 0.0
        if reached_goal:
            r_env += 5.0
        if collided:
            r_env -= 5.0

        return (
            self._build_obs(),
            r_env,
            terminated,
            truncated,
            self._info(
                dist_to_goal=dist_to_goal,
                reached_goal=reached_goal,
                collided=collided,
            ),
        )

    # Observation construction
    def _build_obs(self) -> np.ndarray:
        lidar = self._lidar_scan(self.robot_pos, self.robot_heading)
        delta_world = self.goal - self.robot_pos
        c, s = np.cos(self.robot_heading), np.sin(self.robot_heading)
        dx_robot =  delta_world[0] * c + delta_world[1] * s
        dy_robot = -delta_world[0] * s + delta_world[1] * c
        return np.concatenate([
            lidar,
            np.array([dx_robot, dy_robot], dtype=np.float32),
        ]).astype(np.float32)

    def _lidar_scan(self, origin: np.ndarray, heading: float) -> np.ndarray:
        world_angles = self.beam_angles + heading
        dirs = np.stack(
            [np.cos(world_angles), np.sin(world_angles)], axis=-1
        ).astype(np.float32) 

        # AABB extents
        xmin = self.obstacles[:, 0] - self.obstacles[:, 2]   # (N_obs,)
        xmax = self.obstacles[:, 0] + self.obstacles[:, 2]
        ymin = self.obstacles[:, 1] - self.obstacles[:, 3]
        ymax = self.obstacles[:, 1] + self.obstacles[:, 3]

        eps = 1e-12
        dx = dirs[:, 0:1]  # (N_beams, 1)
        dy = dirs[:, 1:2]
        inv_dx = np.where(np.abs(dx) > eps, 1.0 / np.where(dx == 0, eps, dx),
                          np.sign(dx + eps) * 1e12)
        inv_dy = np.where(np.abs(dy) > eps, 1.0 / np.where(dy == 0, eps, dy),
                          np.sign(dy + eps) * 1e12)

        t1 = (xmin[None, :] - origin[0]) * inv_dx  # (N_beams, N_obs)
        t2 = (xmax[None, :] - origin[0]) * inv_dx
        t3 = (ymin[None, :] - origin[1]) * inv_dy
        t4 = (ymax[None, :] - origin[1]) * inv_dy

        t_near = np.maximum(np.minimum(t1, t2), np.minimum(t3, t4))
        t_far  = np.minimum(np.maximum(t1, t2), np.maximum(t3, t4))

        valid = (t_far >= t_near) & (t_far >= 0.0)
        hit_dist = np.where(valid, np.maximum(t_near, 0.0), np.inf)

        min_dist = hit_dist.min(axis=1)
        return np.minimum(min_dist, self.lidar_max_range).astype(np.float32)

    def _min_clearance(self, pos: np.ndarray) -> float:
        dx = np.maximum(0.0, np.abs(pos[0] - self.obstacles[:, 0]) - self.obstacles[:, 2])
        dy = np.maximum(0.0, np.abs(pos[1] - self.obstacles[:, 1]) - self.obstacles[:, 3])
        return float(np.min(np.hypot(dx, dy)))

    @staticmethod
    def _wrap_angle(theta: float) -> float:
        return float((theta + np.pi) % (2.0 * np.pi) - np.pi)

    def _info(self, **extras) -> dict[str, Any]:
        return {
            "robot_pos": self.robot_pos.copy(),
            "robot_heading": self.robot_heading,
            "goal": self.goal.copy(),
            "step_count": self.step_count,
            **extras,
        }



if __name__ == "__main__":
    import math

    env = ToyNavEnv(seed=0)
    assert env.observation_space.shape == (22,)
    assert env.action_space.shape == (3,)
    obs, info = env.reset(seed=0)
    assert obs.shape == (22,) and obs.dtype == np.float32
    assert env.observation_space.contains(obs), (
        "initial obs out of declared space"
    )
    print(f"reset: obs.shape={obs.shape}, lidar in "
          f"[{obs[:20].min():.2f}, {obs[:20].max():.2f}] m, "
          f"|Δ|={np.linalg.norm(obs[20:]):.2f} m")
    assert (obs[:20] > 0).all(), "LiDAR returned 0 at spawn — collision at init"
    assert (obs[:20] <= env.lidar_max_range + 1e-6).all()

    try:
        env.step(np.zeros(2))
    except ValueError as e:
        print(f"wrong action shape caught: {e}")

    rng = np.random.default_rng(0)
    successes = collisions = timeouts = 0
    total_steps = 0
    for ep in range(20):
        obs, _ = env.reset()
        for t in range(env.max_steps + 1):
            a = rng.uniform(env.action_space.low, env.action_space.high)
            obs, r, terminated, truncated, info = env.step(a)
            total_steps += 1
            assert not (terminated and truncated), (
                "terminated and truncated must be mutually exclusive"
            )
            if terminated:
                if info["reached_goal"]:
                    successes += 1
                if info["collided"]:
                    collisions += 1
                break
            if truncated:
                timeouts += 1
                break
    print(f"20 random episodes ({total_steps} steps): "
          f"goal={successes}  collision={collisions}  timeout={timeouts}")

    env2 = ToyNavEnv(seed=42)
    obs_a, _ = env2.reset(seed=42)
    obs_b, _ = env2.reset(seed=42)
    assert np.allclose(obs_a, obs_b)
    print("reset(seed=42) is deterministic.")

    env3 = ToyNavEnv(seed=7, num_obstacles=0)
    env3.reset(seed=7)
    env3.robot_pos = np.array([env3.half - 0.3, 0.0], dtype=np.float32)
    env3.robot_heading = 0.0 
    obs_close = env3._build_obs()
    print(f"beam 0 from x=+2.4 facing right: {obs_close[0]:.3f} m  (expect ~0.30)")
    assert 0.20 < obs_close[0] < 0.40

    print("\n--- end-to-end pipeline smoke test ---")
    import torch
    from src.algorithms.sac import SACAgent
    from src.algorithms.gail import GAILTrainer
    from src.rewards.hybrid_reward import HybridReward
    from src.buffers.replay_buffer import ReplayBuffer
    from src.buffers.expert_buffer import ExpertBuffer

    torch.manual_seed(0)
    env = ToyNavEnv(seed=0)
    state_dim, action_dim = 22, 3

    sac = SACAgent(
        state_dim=state_dim,
        action_dim=action_dim,
        action_scale=torch.tensor([env.v_max, env.v_max, env.omega_max]),
        device="cpu",
    )
    gail = GAILTrainer(state_dim=state_dim, action_dim=action_dim, device="cpu")
    hybrid = HybridReward()
    replay = ReplayBuffer(
        capacity=10_000, state_dim=state_dim, action_dim=action_dim,
        device="cpu", seed=0,
    )

    import tempfile, os
    states, actions, next_states, dones, starts = [], [], [], [], []
    cur_start = 0
    for _ in range(3):
        o, _ = env.reset()
        starts.append(cur_start)
        for _ in range(50):
            a = env.action_space.sample()
            o_next, _, term, trunc, _ = env.step(a)
            states.append(o); actions.append(a); next_states.append(o_next)
            dones.append([float(term)])
            cur_start += 1
            o = o_next
            if term or trunc:
                break
    with tempfile.NamedTemporaryFile(suffix=".h5", delete=False) as f:
        expert_path = f.name
    ExpertBuffer.write_hdf5(
        expert_path,
        np.array(states), np.array(actions), np.array(next_states),
        np.array(dones), np.array(starts, dtype=np.int64),
    )
    expert = ExpertBuffer(expert_path, seed=0)

    obs, _ = env.reset()
    n_steps, batch_size, updates = 0, 64, 0
    for step in range(200):
        a = sac.act(obs)
        obs_next, _, terminated, truncated, info = env.step(a)
        replay.add(obs, a, obs_next, terminated)
        obs = obs_next if not (terminated or truncated) else env.reset()[0]
        n_steps += 1

        if len(replay) >= batch_size:
            agent_batch = replay.sample(batch_size)
            expert_batch = expert.sample(batch_size)
            gail_m = gail.update(expert_batch, agent_batch)
            r_gail = gail.compute_reward(
                agent_batch["state"], agent_batch["action"]
            )
            r_total = hybrid.compute(
                agent_batch["state"], agent_batch["next_state"], r_gail
            )["r_total"]
            sac_m = sac.update(agent_batch, r_total)
            updates += 1
            for k, v in {**gail_m, **sac_m}.items():
                assert math.isfinite(v), (
                    f"{k} non-finite at update {updates}: {v}"
                )
    print(f"ran {n_steps} env steps and {updates} joint SAC+GAIL updates")
    print(f"final: r_total mean={sac_m['r_mean']:+.3f}  "
          f"actor_loss={sac_m['actor_loss']:+.3f}  "
          f"D(exp)={gail_m['d_expert']:.3f}  D(agn)={gail_m['d_agent']:.3f}")

    os.remove(expert_path)
    print("\nall checks pass — full pipeline runs end-to-end.")
