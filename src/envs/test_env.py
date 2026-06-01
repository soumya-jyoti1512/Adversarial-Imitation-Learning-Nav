import numpy as np
import gymnasium as gym
from gymnasium import spaces
from typing import Any, Optional

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

        self.env_size = float(env_size)
        self.half = self.env_size / 2.0
        self.num_obstacles= int(num_obstacles)
        self.obs_size_lo, self.obs_size_hi = obstacle_size_range
        self.robot_radius= float(robot_radius)

       
        self.lidar_max_range= float(lidar_max_range)
        self.lidar_num_beams= int(lidar_num_beams)
        self.beam_angles = np.linspace(
            0.0, 2.0 * np.pi, self.lidar_num_beams, endpoint=False,
            dtype=np.float32,
        )

        self.v_max= float(v_max)
        self.omega_max= float(omega_max)
        self.dt = float(dt)
        self.max_steps= int(max_steps)
        self.goal_threshold= float(goal_threshold)
        self.min_spawn_goal_dist= float(min_spawn_goal_dist)

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

        self._rng = np.random.default_rng(seed)

        self.robot_pos = np.zeros(2, dtype=np.float32)
        self.robot_heading = 0.0
        self.goal = np.zeros(2, dtype=np.float32)
        self.obstacles = np.zeros((0, 4), dtype=np.float32) 
        self.step_count = 0

   
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
            if np.linalg.norm(candidate - self.robot_pos)>= self.min_spawn_goal_dist:
                self.goal = candidate
                break
        else:
            self.goal= candidate

        self.step_count =0
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
            (-self.half - wt,  0.0, wt, self.half + 2 * wt), 
            ( self.half + wt,  0.0, wt, self.half + 2 * wt),  
            ( 0.0, -self.half - wt, self.half + 2 * wt, wt),  
            ( 0.0,  self.half + wt, self.half + 2 * wt, wt),  
        ]
        all_boxes = obstacles + [np.array(w, dtype=np.float32) for w in walls]
        return np.stack(all_boxes, axis=0)

    @staticmethod
    def _aabb_overlap(a: np.ndarray, b: np.ndarray, gap: float = 0.0) -> bool:
        return (
            abs(a[0] - b[0]) < a[2] + b[2] + gap
            and abs(a[1] - b[1]) < a[3] + b[3] + gap
        )

    def _sample_free_position(self) -> np.ndarray:
        for _ in range(200):
            pos = self._rng.uniform(
                low=-self.half + self.robot_radius + 0.1,
                high= self.half - self.robot_radius - 0.1,
                size=2,
            ).astype(np.float32)
            if self._min_clearance(pos) > self.robot_radius + 0.05:
                return pos
        return np.zeros(2, dtype=np.float32)

    def step(
        self, action: np.ndarray
    ) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        a = np.asarray(action, dtype=np.float32).reshape(-1)
        a = np.clip(a, self.action_space.low, self.action_space.high)
        vx_body, vy_body, omega = float(a[0]), float(a[1]), float(a[2])

        c, s = np.cos(self.robot_heading), np.sin(self.robot_heading)
        vx_world = vx_body * c - vy_body * s
        vy_world = vx_body * s + vy_body * c
        self.robot_pos = self.robot_pos + np.array(
            [vx_world * self.dt, vy_world * self.dt], dtype=np.float32
        )
        self.robot_heading = self._wrap_angle(self.robot_heading + omega * self.dt)
        self.step_count += 1

        dist_to_goal = float(np.linalg.norm(self.goal - self.robot_pos))
        reached_goal = dist_to_goal < self.goal_threshold
        collided = self._min_clearance(self.robot_pos)< self.robot_radius
        terminated = bool(reached_goal or collided)
        truncated = (not terminated) and self.step_count>= self.max_steps

   
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

      
        xmin = self.obstacles[:, 0] - self.obstacles[:, 2]   
        xmax = self.obstacles[:, 0] + self.obstacles[:, 2]
        ymin = self.obstacles[:, 1] - self.obstacles[:, 3]
        ymax = self.obstacles[:, 1] + self.obstacles[:, 3]

        eps = 1e-12
        dx = dirs[:, 0:1]  
        dy = dirs[:, 1:2]
        inv_dx = np.where(np.abs(dx) > eps, 1.0 / np.where(dx == 0, eps, dx),
                          np.sign(dx + eps) * 1e12)
        inv_dy = np.where(np.abs(dy) > eps, 1.0 / np.where(dy == 0, eps, dy),
                          np.sign(dy + eps) * 1e12)

        t1 = (xmin[None, :] - origin[0]) * inv_dx  
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
