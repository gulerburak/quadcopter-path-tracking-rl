"""Evaluate a trained WaypointAviary policy in isolation (no planner/room).

Reports mean/std final tracking error and success rate over several episodes.

Example
-------
    python3 -m src.rl.evaluate --model models/ppo_.../best_model.zip --episodes 20 --gui
"""
import argparse
import numpy as np

from stable_baselines3 import PPO, SAC
from gym_pybullet_drones.utils.enums import ObservationType, ActionType

from src.rl.waypoint_aviary import WaypointAviary

ALGOS = {"ppo": PPO, "sac": SAC}


def main(args):
    algo_cls = ALGOS[args.algo]
    model = algo_cls.load(args.model, device=args.device)

    env = WaypointAviary(gui=args.gui, obs=ObservationType.KIN, act=ActionType.RPM, ctrl_freq=args.ctrl_freq,
                         episode_len_sec=args.episode_len_sec, target_bounds_xy=args.target_bounds_xy,
                         target_bounds_z=(args.target_bounds_z_min, args.target_bounds_z_max),
                         num_segments=args.num_segments)

    final_dists = []
    successes = 0
    for ep in range(args.episodes):
        obs, info = env.reset(seed=args.seed + ep)
        done = False
        last_dist = info["dist_to_target"]
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            last_dist = info["dist_to_target"]
            done = terminated or truncated
        final_dists.append(last_dist)
        if last_dist < args.success_threshold:
            successes += 1
        print(f"[episode {ep}] final dist to target = {last_dist:.3f} m")

    env.close()

    final_dists = np.array(final_dists)
    print(f"\nFinal tracking error: mean={final_dists.mean():.3f} m, std={final_dists.std():.3f} m")
    print(f"Success rate (< {args.success_threshold} m): {100 * successes / args.episodes:.1f}%")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, help="path to a trained SB3 model .zip")
    parser.add_argument("--algo", default="ppo", choices=list(ALGOS.keys()))
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--ctrl_freq", type=int, default=40)
    parser.add_argument("--episode_len_sec", type=float, default=14.0,
                        help="must match the final training stage's episode_len_sec")
    parser.add_argument("--target_bounds_xy", type=float, default=6.0)
    parser.add_argument("--target_bounds_z_min", type=float, default=0.5)
    parser.add_argument("--target_bounds_z_max", type=float, default=2.5)
    parser.add_argument("--num_segments", type=int, default=1,
                        help=">1 evaluates the multi-waypoint path task instead of one static target")
    parser.add_argument("--success_threshold", type=float, default=0.3)
    parser.add_argument("--gui", action="store_true")
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    args = parser.parse_args()

    main(args)
