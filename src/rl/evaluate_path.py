"""Evaluate a trained policy on the deployed path-following task.

Unlike `evaluate.py` (one static target), this drives `CtrlAviary` + `RLControl`
along a densified multi-segment path and reports path completion, final distance,
crash rate, and success.

Example
-------
    python3 -m src.rl.evaluate_path --model models/ppo_.../best_model.zip --episodes 20 --gui
"""
import argparse

import numpy as np
from gym_pybullet_drones.utils.enums import DroneModel

from src.envs.CtrlAviary import CtrlAviary
from src.rl.path_utils import densify_path

MAX_TILT = 1.0  # rad; matches WaypointAviary's crash criterion


def generate_path(rng, start, n_segments, target_bounds_xy, target_bounds_z):
    """Random multi-segment path within the same bounds as training / evaluate.py."""
    waypoints = [np.asarray(start, dtype=float)]
    for _ in range(n_segments):
        xy = rng.uniform(-target_bounds_xy, target_bounds_xy, size=2)
        z = rng.uniform(target_bounds_z[0], target_bounds_z[1])
        waypoints.append(np.array([xy[0], xy[1], z]))
    return np.array(waypoints)


def is_crashed(state, target_bounds_xy, target_bounds_z, margin=2.0):
    pos = state[0:3]
    rpy = state[7:10]
    out_of_bounds = (abs(pos[0]) > target_bounds_xy + margin
                      or abs(pos[1]) > target_bounds_xy + margin
                      or pos[2] > target_bounds_z[1] + margin
                      or pos[2] < 0.02)
    excessive_tilt = abs(rpy[0]) > MAX_TILT or abs(rpy[1]) > MAX_TILT
    return out_of_bounds or excessive_tilt


def main(args):
    rng = np.random.default_rng(args.seed)
    action = np.zeros((1, 4))  # ignored: CtrlAviary path controller overrides it

    final_dists = []
    frac_reached = []
    crashes = 0
    successes = 0

    for ep in range(args.episodes):
        start = np.array([0.0, 0.0, 1.0])
        path = generate_path(rng, start, args.n_segments, args.target_bounds_xy,
                             (args.target_bounds_z_min, args.target_bounds_z_max))
        path_dense = densify_path(path, factor=5)

        env = CtrlAviary(drone_model=DroneModel.CF2X,
                         num_drones=1,
                         initial_xyzs=np.array([start]),
                         initial_rpys=np.array([[0.0, 0.0, 0.0]]),
                         pyb_freq=240,
                         ctrl_freq=40,
                         gui=args.gui,
                         obstacles=False,
                         use_path_controller=True,
                         controller_type='rl',
                         rl_model_path=args.model,
                         rl_algo=args.algo,
                         rl_device=args.device)
        env.PATH_REF = path_dense
        env.wp_counters = np.zeros(env.NUM_DRONES, dtype=int)

        max_steps = int(args.duration_sec * env.CTRL_FREQ)
        crashed = False
        for _ in range(max_steps):
            env.step(action)
            state = env._getDroneStateVector(0)
            if is_crashed(state, args.target_bounds_xy,
                          (args.target_bounds_z_min, args.target_bounds_z_max)):
                crashed = True
                break

        final_state = env._getDroneStateVector(0)
        final_dist = float(np.linalg.norm(path_dense[-1] - final_state[0:3]))
        reached_frac = float(env.wp_counters[0]) / (len(path_dense) - 1)
        env.close()

        final_dists.append(final_dist)
        frac_reached.append(reached_frac)
        if crashed:
            crashes += 1
        success = (not crashed) and reached_frac >= 0.99 and final_dist < args.success_threshold
        if success:
            successes += 1

        print(f"[episode {ep}] path_reached={100 * reached_frac:.1f}%  "
             f"final_dist={final_dist:.3f} m  crashed={crashed}")

    final_dists = np.array(final_dists)
    frac_reached = np.array(frac_reached)
    print(f"\nFinal distance to path end: mean={final_dists.mean():.3f} m, std={final_dists.std():.3f} m")
    print(f"Path completion: mean={100 * frac_reached.mean():.1f}%, std={100 * frac_reached.std():.1f}%")
    print(f"Crash rate: {100 * crashes / args.episodes:.1f}%")
    print(f"Success rate (path fully reached, no crash, < {args.success_threshold} m): "
         f"{100 * successes / args.episodes:.1f}%")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, help="path to a trained SB3 model .zip")
    parser.add_argument("--algo", default="ppo", choices=["ppo", "sac"])
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--n_segments", type=int, default=3, help="waypoints per random path (excluding start)")
    parser.add_argument("--duration_sec", type=float, default=20.0, help="per-episode time budget")
    parser.add_argument("--target_bounds_xy", type=float, default=6.0)
    parser.add_argument("--target_bounds_z_min", type=float, default=0.5)
    parser.add_argument("--target_bounds_z_max", type=float, default=2.5)
    parser.add_argument("--success_threshold", type=float, default=0.3)
    parser.add_argument("--gui", action="store_true")
    parser.add_argument("--device", default="cpu", choices=["auto", "cpu", "cuda"])
    args = parser.parse_args()

    main(args)
