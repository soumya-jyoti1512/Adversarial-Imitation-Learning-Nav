#!/usr/bin/env python3

import argparse
import os
import sys
import time
import warnings
from pathlib import Path
from typing import Optional
import numpy as np

_FILE_ROOT = Path(__file__).resolve().parent.parent
if str(_FILE_ROOT) not in sys.path:
    sys.path.insert(0, str(_FILE_ROOT))

from src.envs.test_env import ToyNavEnv  
from src.buffers.expert_buffer import ExpertBuffer  


try:
    import pygame
    HAS_PYGAME= True
except ImportError:
    HAS_PYGAME= False


class InputDevice:

    def poll(self, obs: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def should_quit(self) -> bool:
        return False

    def close(self) -> None:
        pass



class ScriptedInput(InputDevice):

    def __init__(self, v_max: float, omega_max: float, speed_frac: float= 0.7):
        self.v_max= float(v_max)
        self.omega_max= float(omega_max)
        self.speed_frac= float(speed_frac)
        self._quit= False

    def poll(self, obs: np.ndarray) -> np.ndarray:
        dx, dy = float(obs[-2]), float(obs[-1])
        dist= float(np.hypot(dx, dy)) + 1e-6
        return np.array([
            (dx / dist) * self.v_max * self.speed_frac,
            (dy / dist) * self.v_max * self.speed_frac,
            0.0,
        ], dtype=np.float32)


class JoystickInput:

    def __init__(
        self,
        v_max: float,
        omega_max: float,
        deadzone: float = 0.10,
        axis_left_x: int= 0,
        axis_left_y: int= 1,
        axis_right_x: int = 3,   
    ):
        
        self.v_max = float(v_max)
        self.omega_max = float(omega_max)
        self.deadzone = float(deadzone)
        self.axis_left_x = int(axis_left_x)
        self.axis_left_y = int(axis_left_y)
        self.axis_right_x = int(axis_right_x)

        pygame.init()
        pygame.joystick.init()
        self.joy = pygame.joystick.Joystick(0)
        self.joy.init()

        self._quit = False

    def _apply_deadzone(self, x: float) -> float:
        return 0.0 if abs(x) < self.deadzone else x

    def poll(self, obs: np.ndarray) -> np.ndarray:
        for event in pygame.event.get():
            if event.type== pygame.QUIT:
                self._quit= True
            elif event.type== pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                self._quit = True

       
        lx = self._apply_deadzone(self.joy.get_axis(self.axis_left_x))
        ly = self._apply_deadzone(self.joy.get_axis(self.axis_left_y))
        rx = self._apply_deadzone(self.joy.get_axis(self.axis_right_x))

        vx = -ly * self.v_max         
        vy = -lx * self.v_max         
        omega = -rx * self.omega_max  
        return np.array([vx, vy, omega], dtype=np.float32)

    def should_quit(self) -> bool:
        return self._quit

    def close(self) -> None:
        try:
            pygame.joystick.quit()
            pygame.quit()
        except Exception:
            pass


class KeyboardInput: 

    def __init__(self, v_max: float, omega_max: float):

        self.v_max = float(v_max)
        self.omega_max = float(omega_max)
        pygame.init()
        try:
            self.surface = pygame.display.set_mode((360, 140))
        except pygame.error as e:
            from e
        self._quit= False
        self._font= pygame.font.Font(None, 22)

    def _draw_status(self, ep: int, total: int, ep_step: int) -> None:
        self.surface.fill((20, 20, 30))
        for i, line in enumerate(lines):
            txt= self._font.render(line, True, (220, 220, 240))
            self.surface.blit(txt, (10, 8+i * 26))
        pygame.display.flip()

    def poll(self, obs: np.ndarray) -> np.ndarray:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self._quit = True
            elif event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                self._quit = True
        k = pygame.key.get_pressed()
        vx = (int(k[pygame.K_w]) - int(k[pygame.K_s])) * self.v_max
        vy = (int(k[pygame.K_a]) - int(k[pygame.K_d])) * self.v_max
        omega = (int(k[pygame.K_q]) - int(k[pygame.K_e])) * self.omega_max
        return np.array([vx, vy, omega], dtype=np.float32)

    def should_quit(self) -> bool:
        return self._quit

    def close(self) -> None:
        try:
            pygame.quit()
        except Exception:
            pass




def make_env(backend: str, seed: Optional[int]):
    if backend == "toy":
        return ToyNavEnv(seed=seed)
    elif backend == "gazebo":
        from src.envs.gazebo_env import GazeboEnv
        return GazeboEnv(seed=seed)
    raise ValueError(f"unknown backend: {backend!r}")



def record_episodes(
    env,
    device: InputDevice,
    num_episodes: int,
    auto_reject_failures: bool,
    realtime: bool,
    seed: Optional[int]=None,
):
   
    states:      list[np.ndarray] = []
    actions:     list[np.ndarray] = []
    next_states: list[np.ndarray] = []
    dones:       list[list[float]] = []
    episode_starts: list[int] = []

    kept = 0
    total_attempts = 0
    rng = np.random.default_rng(seed)

    while kept < num_episodes:
        if device.should_quit():
            print("[recorder] quit requested by user.")
            break

        ep_seed = int(rng.integers(0, 2**31 - 1))
        obs, _ = env.reset(seed=ep_seed)
        total_attempts += 1

        ep_s, ep_a, ep_sn, ep_d = [], [], [], []
        t0 = time.time()
        outcome = "timeout"  

        while True:
            action= device.poll(obs)
            action= np.clip(
                action, env.action_space.low, env.action_space.high
            ).astype(np.float32)

            obs_next, _, terminated, truncated, info = env.step(action)
            ep_s.append(obs.copy())
            ep_a.append(action.copy())
            ep_sn.append(obs_next.copy())
            ep_d.append([float(terminated)])
            obs = obs_next

            if device.should_quit():
                outcome="aborted"
                break
            if terminated:
                outcome="success" if info.get("reached_goal", False) else "collision"
                break
            if truncated:
                outcome="timeout"
                break

            if realtime:
                target=getattr(env, "dt", 0.1)
                slack=target -(time.time() - t0)
                if slack > 0:
                    time.sleep(slack)
                t0 = time.time()

        
        ep_len = len(ep_s)
        accept = True
        if outcome == "aborted":
            accept = False
        elif auto_reject_failures and outcome != "success":
            accept = False

        if accept:
            episode_starts.append(len(states))
            states.extend(ep_s)
            actions.extend(ep_a)
            next_states.extend(ep_sn)
            dones.extend(ep_d)
            kept += 1
            print(f"[recorder] ep {kept}/{num_episodes}  "
                  f"outcome={outcome:9s}  steps={ep_len:4d}  KEPT")
        else:
            print(f"[recorder] attempt {total_attempts}  "
                  f"outcome={outcome:9s}  steps={ep_len:4d}  rejected")

    return (
        np.asarray(states,      dtype=np.float32),
        np.asarray(actions,     dtype=np.float32),
        np.asarray(next_states, dtype=np.float32),
        np.asarray(dones,       dtype=np.float32),
        np.asarray(episode_starts, dtype=np.int64),
    )




def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Record the  expert navigation demonstrations to  HDF5.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--backend", choices=["toy", "gazebo"], default="toy")
    p.add_argument("--device", choices=["joystick", "keyboard", "scripted"],
                   default="joystick")
    p.add_argument("--num-episodes", type=int, default=40)
    p.add_argument("--output", type=str, default="data/expert.h5")
    p.add_argument("--v-max", type=float, default=1.0)
    p.add_argument("--omega-max", type=float, default=1.5)
    p.add_argument("--deadzone", type=float, default=0.10)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--auto-reject-failures", action="store_true",
                   help="Drop collision / timeout episodes from the dataset.")
    p.add_argument("--no-realtime", action="store_true")
    return p


def main(argv: Optional[list[str]] = None) -> None:
    args = build_arg_parser().parse_args(argv)
    env = make_env(args.backend, seed=args.seed)

    if args.device == "joystick":
        device: InputDevice = JoystickInput(
            v_max=args.v_max, omega_max=args.omega_max,
            deadzone=args.deadzone,
        )
    elif args.device == "keyboard":
        device = KeyboardInput(v_max=args.v_max, omega_max=args.omega_max)
    else:
        device = ScriptedInput(v_max=args.v_max, omega_max=args.omega_max)

    realtime = not args.no_realtime and args.device != "scripted"

    try:
        states, actions, next_states, dones, ep_starts = record_episodes(
            env, device,
            num_episodes=args.num_episodes,
            auto_reject_failures=args.auto_reject_failures,
            realtime=realtime,
            seed=args.seed,
        )
    finally:
        device.close()
        if hasattr(env, "close"):
            env.close()

    if len(states) == 0:
        print("[recorder] no episodes captured so nothing to save.")
        sys.exit(1)

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    ExpertBuffer.write_hdf5(
        args.output, states, actions, next_states, dones, ep_starts
    )
    print(f"\n[recorder] wrote {args.output}")
    print(f"           N = {len(states)}  "
          f"episodes = {len(ep_starts)}  "
          f"avg_len = {len(states) / max(1, len(ep_starts)):.1f}")
