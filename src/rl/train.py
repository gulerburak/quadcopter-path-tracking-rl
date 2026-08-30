"""Train a Stable-Baselines3 PPO/SAC policy on `WaypointAviary`.

Example
-------
    python3 -m src.rl.train --algo ppo --timesteps 4000000
    python3 -m src.rl.train --algo sac --timesteps 1000000 --gamma 0.98 --lr 1e-3
    python3 -m src.rl.train --algo ppo --timesteps 4000000 --no-curriculum

Default is a 3-stage target-range curriculum. Stages are parallel comma-separated
`--stage_*` lists; `--no-curriculum` trains on the full volume from step 0.

Checkpoints land under `models/<algo>_<timestamp>/`. Deploy `best_model.zip` with
`--controller rl --rl_model <path>`, or score it with `src/rl/evaluate.py`.
"""
import os
import shutil
import argparse
from datetime import datetime

from stable_baselines3 import PPO, SAC
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import EvalCallback
from gym_pybullet_drones.utils.enums import ObservationType, ActionType

from src.rl.waypoint_aviary import WaypointAviary

ALGOS = {"ppo": PPO, "sac": SAC}


def parse_stages(args):
    """Build curriculum stages from the parallel --stage_* CLI lists."""
    fractions = [float(x) for x in args.stage_fractions.split(",")]
    bounds_xy = [float(x) for x in args.stage_target_bounds_xy.split(",")]
    bounds_z = [tuple(float(v) for v in pair.split(":")) for pair in args.stage_target_bounds_z.split(",")]
    ep_lens = [float(x) for x in args.stage_episode_len_sec.split(",")]
    num_segments = [int(x) for x in args.stage_num_segments.split(",")]
    n = len(fractions)
    if not (len(bounds_xy) == len(bounds_z) == len(ep_lens) == len(num_segments) == n):
        raise ValueError(
            f"--stage_* lists must all have the same length; got "
            f"fractions={len(fractions)}, target_bounds_xy={len(bounds_xy)}, "
            f"target_bounds_z={len(bounds_z)}, episode_len_sec={len(ep_lens)}, "
            f"num_segments={len(num_segments)}"
        )
    if abs(sum(fractions) - 1.0) > 1e-6:
        raise ValueError(f"--stage_fractions must sum to 1.0, got {fractions} (sum={sum(fractions)})")
    return [
        dict(fraction=fractions[i], target_bounds_xy=bounds_xy[i], target_bounds_z=bounds_z[i],
            episode_len_sec=ep_lens[i], num_segments=num_segments[i])
        for i in range(n)
    ]


def make_env_kwargs(args, target_bounds_xy, target_bounds_z, episode_len_sec, num_segments=1):
    return dict(
        obs=ObservationType.KIN,
        act=ActionType.RPM,
        ctrl_freq=args.ctrl_freq,
        pyb_freq=args.pyb_freq,
        episode_len_sec=episode_len_sec,
        target_bounds_xy=target_bounds_xy,
        target_bounds_z=target_bounds_z,
        max_tilt=args.max_tilt,
        tilt_penalty_weight=args.tilt_penalty_weight,
        alive_bonus=args.alive_bonus,
        crash_penalty=args.crash_penalty,
        num_segments=num_segments,
        wp_threshold=args.wp_threshold,
        progress_bonus=args.progress_bonus,
        path_complete_bonus=args.path_complete_bonus,
        local_dist_norm=args.local_dist_norm,
        crash_terminates=args.crash_terminates,
        init_randomization=args.init_randomization,
        action_scale=args.action_scale,
        rotor_randomization=args.rotor_randomization,
    )


def run_stage(model, env_kwargs, timesteps, out_dir, args, tb_log_name, reset_num_timesteps):
    """Attach a fresh vec env for this stage, train, then close it."""
    train_env = make_vec_env(WaypointAviary, env_kwargs=env_kwargs, n_envs=args.n_envs, seed=args.seed)
    eval_kwargs = dict(env_kwargs)
    if args.eval_init_randomization is not None:
        eval_kwargs['init_randomization'] = args.eval_init_randomization
    eval_env = WaypointAviary(**eval_kwargs)
    model.set_env(train_env)

    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=os.path.join(out_dir, tb_log_name),
        log_path=os.path.join(out_dir, tb_log_name),
        eval_freq=max(args.eval_freq // args.n_envs, 1),
        n_eval_episodes=args.n_eval_episodes,
        deterministic=True,
        render=False,
    )
    model.learn(total_timesteps=timesteps, callback=eval_callback, log_interval=100,
               tb_log_name=tb_log_name, reset_num_timesteps=reset_num_timesteps)

    train_env.close()
    eval_env.close()


def main(args):
    out_dir = os.path.join(args.output_folder, f"{args.algo}_{datetime.now().strftime('%m.%d.%Y_%H.%M.%S')}")
    os.makedirs(out_dir, exist_ok=True)

    full_env_kwargs = make_env_kwargs(
        args,
        target_bounds_xy=args.target_bounds_xy,
        target_bounds_z=(args.target_bounds_z_min, args.target_bounds_z_max),
        episode_len_sec=args.episode_len_sec,
        num_segments=args.num_segments,
    )

    if args.curriculum:
        stages = parse_stages(args)
        print("[INFO] Curriculum stages:", stages)
        init_env_kwargs = make_env_kwargs(args, stages[0]["target_bounds_xy"], stages[0]["target_bounds_z"],
                                          stages[0]["episode_len_sec"], stages[0]["num_segments"])
    else:
        stages = None
        print("[INFO] No curriculum; training on full-task env kwargs:", full_env_kwargs)
        init_env_kwargs = full_env_kwargs

    algo_cls = ALGOS[args.algo]
    policy_kwargs = dict(net_arch=[args.net_width] * args.net_depth)
    probe_env = WaypointAviary(**full_env_kwargs)
    print("[INFO] Action space:", probe_env.action_space)
    print("[INFO] Observation space:", probe_env.observation_space)
    probe_env.close()

    init_train_env = make_vec_env(WaypointAviary, env_kwargs=init_env_kwargs, n_envs=args.n_envs, seed=args.seed)
    algo_kwargs = {}
    if args.algo == "ppo" and args.target_kl is not None:
        algo_kwargs["target_kl"] = args.target_kl
    model = algo_cls(
        "MlpPolicy",
        init_train_env,
        learning_rate=args.lr,
        gamma=args.gamma,
        policy_kwargs=policy_kwargs,
        tensorboard_log=os.path.join(out_dir, "tb"),
        seed=args.seed,
        device=args.device,
        verbose=1,
        **algo_kwargs,
    )
    init_train_env.close()  # run_stage always builds its own train_env

    if args.curriculum:
        remaining = args.timesteps
        for i, stage in enumerate(stages):
            is_last = (i == len(stages) - 1)
            stage_steps = remaining if is_last else int(args.timesteps * stage["fraction"])
            remaining -= stage_steps
            stage_env_kwargs = make_env_kwargs(args, stage["target_bounds_xy"], stage["target_bounds_z"],
                                              stage["episode_len_sec"], stage["num_segments"])
            print(f"[INFO] Curriculum stage {i + 1}/{len(stages)}: {stage_steps} steps, "
                  f"target_bounds_xy={stage['target_bounds_xy']}, target_bounds_z={stage['target_bounds_z']}, "
                  f"episode_len_sec={stage['episode_len_sec']}, num_segments={stage['num_segments']}")
            run_stage(model, stage_env_kwargs, stage_steps, out_dir, args,
                     tb_log_name=f"stage{i + 1}", reset_num_timesteps=(i == 0))
        last_stage_best = os.path.join(out_dir, f"stage{len(stages)}", "best_model.zip")
        if os.path.exists(last_stage_best):
            shutil.copy(last_stage_best, os.path.join(out_dir, "best_model.zip"))
    else:
        run_stage(model, full_env_kwargs, args.timesteps, out_dir, args, tb_log_name=args.algo, reset_num_timesteps=True)
        shutil.copy(os.path.join(out_dir, args.algo, "best_model.zip"), os.path.join(out_dir, "best_model.zip"))

    model.save(os.path.join(out_dir, "final_model.zip"))
    print(f"[INFO] Saved final model and best checkpoint under {out_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train PPO/SAC on WaypointAviary")
    parser.add_argument("--algo", default="ppo", choices=list(ALGOS.keys()))
    parser.add_argument("--timesteps", type=int, default=int(4e6))
    parser.add_argument("--n_envs", type=int, default=4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--target_kl", type=float, default=None,
                        help="PPO-only: stop an epoch once approx_kl exceeds this (SB3 default: no limit)")
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--net_width", type=int, default=256)
    parser.add_argument("--net_depth", type=int, default=2)
    parser.add_argument("--ctrl_freq", type=int, default=40, help="must match the deployment CtrlAviary's ctrl_freq")
    parser.add_argument("--pyb_freq", type=int, default=240)
    parser.add_argument("--episode_len_sec", type=float, default=14.0, help="used only with --no-curriculum")
    parser.add_argument("--target_bounds_xy", type=float, default=6.0, help="full-task (final stage) target bound")
    parser.add_argument("--target_bounds_z_min", type=float, default=0.5)
    parser.add_argument("--target_bounds_z_max", type=float, default=2.5)
    parser.add_argument("--max_tilt", type=float, default=1.0, help="radians; truncate+penalize past this roll/pitch")
    parser.add_argument("--tilt_penalty_weight", type=float, default=0.5)
    parser.add_argument("--alive_bonus", type=float, default=1.0,
                        help="per-step reward so surviving beats ending the episode early")
    parser.add_argument("--crash_penalty", type=float, default=5.0)
    parser.add_argument("--crash_terminates", action="store_true",
                        help="treat a crash as terminated so SB3 does not bootstrap value past it")
    parser.add_argument("--init_randomization", type=float, default=0.0,
                        help="per-episode initial-state noise (0=off; ~1.0 = +-0.3m, +-0.3rad, +-1 m/s)")
    parser.add_argument("--eval_init_randomization", type=float, default=None,
                        help="EvalCallback override of --init_randomization (default: same as training)")
    parser.add_argument("--action_scale", type=float, default=0.05,
                        help="RPM = HOVER_RPM*(1 + action_scale*action); must match RLControl at deploy")
    parser.add_argument("--rotor_randomization", type=float, default=0.0,
                        help="per-episode single-rotor thrust loss (0=off; 0.2 = one rotor in [0.8, 1.0])")
    parser.add_argument("--num_segments", type=int, default=1,
                        help="used only with --no-curriculum: waypoint count (1 = one static target)")
    parser.add_argument("--wp_threshold", type=float, default=0.2,
                        help="distance to advance a waypoint; must match CtrlAviary path_pid_threshold")
    parser.add_argument("--progress_bonus", type=float, default=2.0,
                        help="path mode: reward for advancing a waypoint")
    parser.add_argument("--path_complete_bonus", type=float, default=50.0,
                        help="path mode: reward for reaching the final waypoint")
    parser.add_argument("--local_dist_norm", type=float, default=1.0,
                        help="path mode: normalizer for distance-to-active-waypoint")
    parser.add_argument("--eval_freq", type=int, default=10000)
    parser.add_argument("--n_eval_episodes", type=int, default=10)
    parser.add_argument("--output_folder", default="models")
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"],
                        help="'auto' picks cuda if available; cpu is often as fast for this MLP")
    parser.add_argument("--curriculum", dest="curriculum", action="store_true", default=True,
                        help="multi-stage target-range curriculum (default: on)")
    parser.add_argument("--no-curriculum", dest="curriculum", action="store_false")
    parser.add_argument("--stage_fractions", default="0.2,0.35,0.45",
                        help="comma-separated fractions of --timesteps per stage; must sum to 1.0")
    parser.add_argument("--stage_target_bounds_xy", default="1.5,3.5,6.0",
                        help="comma-separated per-stage target_bounds_xy")
    parser.add_argument("--stage_target_bounds_z", default="0.75:1.5,0.6:2.0,0.5:2.5",
                        help="comma-separated per-stage 'zmin:zmax' pairs")
    parser.add_argument("--stage_episode_len_sec", default="6,10,14",
                        help="comma-separated per-stage episode length in seconds")
    parser.add_argument("--stage_num_segments", default="1,1,1",
                        help="comma-separated per-stage waypoint count")
    args = parser.parse_args()

    main(args)
