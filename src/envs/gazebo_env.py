from __future__ import annotations

import math
import threading
import time
from typing import Any, Optional

import numpy as np
import gymnasium as gym
from gymnasium import spaces

try:
    import rclpy
    from rclpy.node import Node
    from rclpy.qos import (
        QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy,
    )
    from rclpy.executors import SingleThreadedExecutor

    from geometry_msgs.msg import Twist
    from sensor_msgs.msg import LaserScan
    from nav_msgs.msg import Odometry
    from ros_gz_interfaces.srv import SetEntityPose
    from ros_gz_interfaces.msg import Entity

    HAS_ROS = True
except ImportError:
    HAS_ROS = False


# Helpers
def _quat_to_yaw(qx: float, qy: float, qz: float, qw: float) -> float:
    """Convert a quaternion to yaw (rotation about z)."""
    siny_cosp = 2.0 * (qw * qz + qx * qy)
    cosy_cosp = 1.0 - 2.0 * (qy * qy + qz * qz)
    return float(math.atan2(siny_cosp, cosy_cosp))


def _yaw_to_quat(yaw: float) -> tuple[float, float, float, float]:
    """Convert a yaw to a quaternion (qx, qy, qz, qw)."""
    return (0.0, 0.0, float(math.sin(yaw / 2.0)), float(math.cos(yaw / 2.0)))


# Env
class GazeboEnv(gym.Env):

    metadata = {"render_modes": []}

    def __init__(
        self,
        # Geometry / sensor
        env_size: float = 5.4,
        lidar_num_beams: int = 20,
        lidar_max_range: float = 3.0,
        goal_threshold: float = 0.3,
        collision_terminal_distance: float = 0.15,
        #Dynamics
        v_max: float = 1.0,
        omega_max: float = 1.5,
        dt: float = 0.1,
        max_steps: int = 300,
        min_spawn_goal_dist: float = 2.0,
       #writer
        node_name: str = "gail_sac_env",
        cmd_vel_topic: str = "/cmd_vel",
        scan_topic: str = "/scan",
        odom_topic: str = "/odom",
        world_name: str = "default",
        robot_name: str = "robot",
        message_wait_timeout: float = 5.0,
        seed: Optional[int] = None,
    ) -> None:
        super().__init__()
        if not HAS_ROS:
            raise ImportError(
                "ROS 2 packages are not importable. Source your ROS 2 Jazzy "
                "setup first:\n"
                "    source /opt/ros/jazzy/setup.bash\n"
                "If you're developing without ROS installed, use ToyNavEnv "
                "instead both share the same gym API."
            )

        #Store config
        self.env_size = float(env_size)
        self.half = self.env_size / 2.0
        self.lidar_num_beams = int(lidar_num_beams)
        self.lidar_max_range = float(lidar_max_range)
        self.goal_threshold = float(goal_threshold)
        self.collision_terminal_distance = float(collision_terminal_distance)
        self.v_max = float(v_max)
        self.omega_max = float(omega_max)
        self.dt = float(dt)
        self.max_steps = int(max_steps)
        self.min_spawn_goal_dist = float(min_spawn_goal_dist)
        self.world_name = str(world_name)
        self.robot_name = str(robot_name)
        self.message_wait_timeout = float(message_wait_timeout)

        #Gym spaces
        action_low = np.array([-v_max, -v_max, -omega_max], dtype=np.float32)
        self.action_space = spaces.Box(
            low=action_low, high=-action_low, dtype=np.float32
        )
        obs_low = np.concatenate([
            np.zeros(self.lidar_num_beams, dtype=np.float32),
            np.full(2, -self.env_size, dtype=np.float32),
        ])
        obs_high = np.concatenate([
            np.full(self.lidar_num_beams, self.lidar_max_range, dtype=np.float32),
            np.full(2, self.env_size, dtype=np.float32),
        ])
        self.observation_space = spaces.Box(
            low=obs_low, high=obs_high, dtype=np.float32
        )

        # Per-episode state
        self._rng = np.random.default_rng(seed)
        self.goal = np.zeros(2, dtype=np.float32)
        self.step_count = 0

        # ROS infrastructure
        self._owns_rclpy = not rclpy.ok()
        if self._owns_rclpy:
            rclpy.init()

        self._node = Node(node_name)
        self._scan_lock = threading.Lock()
        self._odom_lock = threading.Lock()
        self._latest_scan: Optional[LaserScan] = None
        self._latest_odom: Optional[Odometry] = None
        self._beam_idx: Optional[np.ndarray] = None

        sensor_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
            durability=DurabilityPolicy.VOLATILE,
        )
        reliable_qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            history=HistoryPolicy.KEEP_LAST,
            depth=10,
        )

        self._cmd_pub = self._node.create_publisher(
            Twist, cmd_vel_topic, reliable_qos
        )
        self._scan_sub = self._node.create_subscription(
            LaserScan, scan_topic, self._on_scan, sensor_qos
        )
        self._odom_sub = self._node.create_subscription(
            Odometry, odom_topic, self._on_odom, sensor_qos
        )
        self._set_pose_client = self._node.create_client(
            SetEntityPose, f"/world/{self.world_name}/set_pose"
        )

        self._executor = SingleThreadedExecutor()
        self._executor.add_node(self._node)
        self._spin_thread = threading.Thread(
            target=self._executor.spin, daemon=True, name="ros2-spin"
        )
        self._spin_thread.start()
        self._closed = False

    # ROS callbacks
    def _on_scan(self, msg: "LaserScan") -> None:
        with self._scan_lock:
            self._latest_scan = msg
            if self._beam_idx is None:
                self._beam_idx = self._compute_beam_indices(msg)

    def _on_odom(self, msg: "Odometry") -> None:
        with self._odom_lock:
            self._latest_odom = msg

    def _compute_beam_indices(self, msg: "LaserScan") -> np.ndarray:
        target = np.linspace(
            0.0, 2.0 * np.pi, self.lidar_num_beams, endpoint=False
        )
        target = (target + np.pi) % (2.0 * np.pi) - np.pi
        n_raw = len(msg.ranges)
        idx = np.round((target - msg.angle_min) / msg.angle_increment).astype(int)
        return np.clip(idx, 0, n_raw - 1)

    # Reset
    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[dict] = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        if seed is not None:
            self._rng = np.random.default_rng(seed)

        # spawn pose and goal in env bounds
        spawn_xy = self._sample_bounded_xy()
        spawn_yaw = float(self._rng.uniform(-math.pi, math.pi))
        for _ in range(50):
            goal_xy = self._sample_bounded_xy()
            if np.linalg.norm(goal_xy - spawn_xy) >= self.min_spawn_goal_dist:
                break
        self.goal = goal_xy.astype(np.float32)

        # Teleport robot via SetEntityPose
        self._teleport_robot(spawn_xy, spawn_yaw)

        self.step_count = 0
        t_request = self._node.get_clock().now().nanoseconds
        self._wait_for_fresh_messages(reference_ns=t_request)
        return self._build_obs(), self._info()

    def _sample_bounded_xy(self) -> np.ndarray:
        margin = self.collision_terminal_distance + 0.2
        return self._rng.uniform(
            low=-self.half + margin,
            high= self.half - margin,
            size=2,
        )

    def _teleport_robot(self, xy: np.ndarray, yaw: float) -> None:
        if not self._set_pose_client.wait_for_service(timeout_sec=2.0):
            raise RuntimeError(
                f"set_pose service /world/{self.world_name}/set_pose not "
                f"available. Is gz-sim running with the right world name? "
                f"Check `ros2 service list | grep set_pose`."
            )
        req = SetEntityPose.Request()
        req.entity = Entity(name=self.robot_name, type=Entity.MODEL)
        req.pose.position.x = float(xy[0])
        req.pose.position.y = float(xy[1])
        req.pose.position.z = 0.0
        qx, qy, qz, qw = _yaw_to_quat(yaw)
        req.pose.orientation.x = qx
        req.pose.orientation.y = qy
        req.pose.orientation.z = qz
        req.pose.orientation.w = qw

        future = self._set_pose_client.call_async(req)
        t0 = time.time()
        while not future.done():
            if time.time() - t0 > 2.0:
                raise RuntimeError("set_pose service call timed out")
            time.sleep(0.005)
        if not future.result().success:
            raise RuntimeError(
                f"set_pose service returned success=False. Verify the "
                f"robot model name ('{self.robot_name}') is correct."
            )

  
    # Step
    def step(
        self, action: np.ndarray
    ) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        a = np.asarray(action, dtype=np.float32).reshape(-1)
        if a.shape != (3,):
            raise ValueError(f"action shape {a.shape}, expected (3,)")
        a = np.clip(a, self.action_space.low, self.action_space.high)

        # ---- Publish cmd_vel ---------------------------------------------
        cmd = Twist()
        cmd.linear.x = float(a[0])
        cmd.linear.y = float(a[1])
        cmd.angular.z = float(a[2])
        self._cmd_pub.publish(cmd)

        time.sleep(self.dt)
        self.step_count += 1

        obs = self._build_obs()
        lidar = obs[:self.lidar_num_beams]

        dist_to_goal = float(np.linalg.norm(obs[self.lidar_num_beams:]))
        reached_goal = dist_to_goal < self.goal_threshold
        collided = float(lidar.min()) < self.collision_terminal_distance
        terminated = bool(reached_goal or collided)
        truncated = (not terminated) and self.step_count >= self.max_steps

        r_env = (5.0 if reached_goal else 0.0) + (-5.0 if collided else 0.0)

        return obs, r_env, terminated, truncated, self._info(
            dist_to_goal=dist_to_goal,
            reached_goal=reached_goal,
            collided=collided,
        )

    # Observation construction
    def _build_obs(self) -> np.ndarray:
        with self._scan_lock:
            scan = self._latest_scan
            beam_idx = self._beam_idx
        with self._odom_lock:
            odom = self._latest_odom
        if scan is None or beam_idx is None or odom is None:
            raise RuntimeError(
                "Building observation before /scan or /odom has arrived. "
                "Did reset() complete successfully?"
            )

        ranges = np.asarray(scan.ranges, dtype=np.float32)[beam_idx]
        ranges = np.where(np.isfinite(ranges), ranges, self.lidar_max_range)
        ranges = np.clip(ranges, 0.0, self.lidar_max_range)

        x = float(odom.pose.pose.position.x)
        y = float(odom.pose.pose.position.y)
        q = odom.pose.pose.orientation
        yaw = _quat_to_yaw(q.x, q.y, q.z, q.w)

        dx_w = self.goal[0] - x
        dy_w = self.goal[1] - y
        c, s = math.cos(yaw), math.sin(yaw)
        dx_robot =  dx_w * c + dy_w * s
        dy_robot = -dx_w * s + dy_w * c

        return np.concatenate([
            ranges,
            np.array([dx_robot, dy_robot], dtype=np.float32),
        ]).astype(np.float32)

    def _wait_for_fresh_messages(self, reference_ns: int) -> None:
        """Block until /scan AND /odom timestamps are newer than reference."""
        deadline = time.time() + self.message_wait_timeout
        while time.time() < deadline:
            with self._scan_lock:
                scan = self._latest_scan
            with self._odom_lock:
                odom = self._latest_odom
            if scan is not None and odom is not None:
                scan_ns = scan.header.stamp.sec * 1_000_000_000 + scan.header.stamp.nanosec
                odom_ns = odom.header.stamp.sec * 1_000_000_000 + odom.header.stamp.nanosec
                if scan_ns >= reference_ns and odom_ns >= reference_ns:
                    return
            time.sleep(0.01)
        raise RuntimeError(
            f"Timed out waiting for fresh /scan and /odom after reset. "
            f"Check that the bridge is alive: `ros2 topic hz /scan` and "
            f"`ros2 topic hz /odom`."
        )

    def _info(self, **extras) -> dict[str, Any]:
        return {
            "goal": self.goal.copy(),
            "step_count": self.step_count,
            **extras,
        }

    # Teardown
    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        try:
            self._cmd_pub.publish(Twist())
        except Exception:
            pass
        self._executor.shutdown()
        self._spin_thread.join(timeout=2.0)
        self._node.destroy_node()
        if self._owns_rclpy and rclpy.ok():
            rclpy.shutdown()

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass
