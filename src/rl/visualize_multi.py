"""Multi-instance GUI visualization of a trained policy (demo / screen recording).

Not used for training. Opens `--n_instances` PyBullet windows running the same
checkpoint on independently sampled targets, stepped in lockstep.

Example
-------
    python3 -m src.rl.visualize_multi --model models/ppo_.../best_model.zip --n_instances 4
"""
import argparse
import time

import numpy as np
from stable_baselines3 import PPO, SAC

from gym_pybullet_drones.utils.enums import ObservationType, ActionType
from src.rl.waypoint_aviary import WaypointAviary

ALGOS = {"ppo": PPO, "sac": SAC}


def main(args):
    algo_cls = ALGOS[args.algo]
    model = algo_cls.load(args.model, device="cpu")

    envs = [
        WaypointAviary(gui=True, obs=ObservationType.KIN, act=ActionType.RPM,
                       ctrl_freq=args.ctrl_freq, episode_len_sec=args.episode_len_sec)
        for _ in range(args.n_instances)
    ]

    for ep in range(args.episodes):
        obs = [env.reset(seed=args.seed + ep * args.n_instances + i)[0] for i, env in enumerate(envs)]
        done = [False] * args.n_instances
        print(f"--- episode {ep} ---")
        for i, env in enumerate(envs):
            print(f"  instance {i}: target = {env.TARGET_POS.round(2)}")

        while not all(done):
            obs_batch = np.stack(obs, axis=0)
            actions, _ = model.predict(obs_batch, deterministic=True)
            for i, env in enumerate(envs):
                if done[i]:
                    continue
                o, reward, terminated, truncated, info = env.step(actions[i])
                obs[i] = o
                if terminated or truncated:
                    done[i] = True
                    print(f"  instance {i}: finished, final dist = {info['dist_to_target']:.2f} m")
            time.sleep(1.0 / args.ctrl_freq)

    for env in envs:
        env.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Multi-instance GUI visualization for report figures/clips")
    parser.add_argument("--model", required=True, help="path to a trained SB3 model .zip")
    parser.add_argument("--algo", default="ppo", choices=list(ALGOS.keys()))
    parser.add_argument("--n_instances", type=int, default=4, help="number of side-by-side GUI windows")
    parser.add_argument("--episodes", type=int, default=3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--ctrl_freq", type=int, default=40)
    parser.add_argument("--episode_len_sec", type=float, default=14.0)
    args = parser.parse_args()

    main(args)
