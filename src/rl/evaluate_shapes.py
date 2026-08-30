"""Shape-tracking evaluation in an empty environment, with optional mid-flight
disturbances.

Drives `CtrlAviary` along a geometric reference (`shape_paths.py`) and measures
cross-track deviation. Controllers can be compared on the same shape
(`--controller rl,pid,mppi`). `--disturbance wind|rotor` injects a gust or
partial rotor loss at `--fault_time`. `--gui --realtime` paces the sim for live
supervision.

Examples
--------
    python -m src.rl.evaluate_shapes --shape circle --controller rl,pid --model M
    python -m src.rl.evaluate_shapes --shape cube --controller rl --model M \
        --gui --realtime --speed 0.5
    python -m src.rl.evaluate_shapes --shape hsquare --controller rl,pid --model M \
        --disturbance wind --fault_time 4 --wind_impulse 1.5
    python -m src.rl.evaluate_shapes --shape circle --controller rl,pid --model M \
        --disturbance rotor --fault_time 4 --fault_rotor 0 --fault_factor 0.7
"""
import argparse
import inspect
import os
import time

import numpy as np
import pybullet as p
import matplotlib
matplotlib.use('Agg')  # headless: we save PNGs, no display needed
import matplotlib.pyplot as plt

from gym_pybullet_drones.utils.enums import DroneModel

from src.envs.CtrlAviary import CtrlAviary
from src.rl.shape_paths import SHAPES, build_path_ref

MAX_TILT = 1.0  # rad; matches WaypointAviary / evaluate_path.py crash criterion

# Per-shape (elev, azim) for the saved 3D figure and GUI camera.
VIEWS = {
    'hsquare': (78, -90), 'triangle': (78, -90), 'circle': (78, -90),
    'vsquare': (6, -90), 'circle_v': (6, -90), 'cube': (22, -55),
}
CTRL_COLORS = {'rl': 'tab:blue', 'pid': 'tab:green', 'mppi': 'tab:orange'}


class FaultCtrlAviary(CtrlAviary):
    """Scale one rotor's RPM by `fault_factor` after `fault_time` seconds."""

    def __init__(self, *args, fault_rotor=0, fault_factor=1.0, fault_time=None, **kwargs):
        self._fault_rotor = int(fault_rotor)
        self._fault_factor = float(fault_factor)
        self._fault_time = fault_time
        super().__init__(*args, **kwargs)

    def _preprocessAction(self, action):
        rpm = super()._preprocessAction(action)
        if self._fault_time is not None and (self.step_counter / self.PYB_FREQ) >= self._fault_time:
            rpm[:, self._fault_rotor] *= self._fault_factor
        return rpm


def make_shape(name, args):
    """Build a shape's vertices, forwarding the size/z params it accepts."""
    fn = SHAPES[name]
    sig = inspect.signature(fn)
    kw = {}
    if 'z' in sig.parameters:
        kw['z'] = args.z
    if 'z0' in sig.parameters:
        kw['z0'] = args.z
    if 'side' in sig.parameters:
        kw['side'] = args.size
    if 'size' in sig.parameters:
        kw['size'] = args.size
    if 'radius' in sig.parameters:
        kw['radius'] = args.size / 2.0
    return fn(**kw)


def is_crashed(state, bounds=8.0, zmax=4.0):
    pos = state[0:3]
    rpy = state[7:10]
    out_of_bounds = (abs(pos[0]) > bounds or abs(pos[1]) > bounds
                     or pos[2] > zmax or pos[2] < 0.02)
    excessive_tilt = abs(rpy[0]) > MAX_TILT or abs(rpy[1]) > MAX_TILT
    return out_of_bounds or excessive_tilt


def cross_track(traj, ref):
    """Per-position distance to the nearest dense reference point."""
    d = np.linalg.norm(traj[:, None, :] - ref[None, :, :], axis=2)
    return d.min(axis=1)


def _rms(a):
    a = np.asarray(a)
    return float(np.sqrt((a ** 2).mean())) if a.size else float('nan')


class GuiOverlay:
    """Reference outline, chased-waypoint marker, trail, and deviation readout.

    Draws vertices (not the dense path) to stay under PyBullet's debug-line
    limit; the trail reuses a fixed pool of line ids for the same reason.
    """

    REF_COLOR = [0.45, 0.45, 0.45]
    TRAIL_COLOR = [0.1, 0.45, 0.95]

    def __init__(self, env, vertices, path_ref, trail_len=300, trail_every=2):
        self.env, self.path_ref = env, path_ref
        self.trail_len, self.trail_every = int(trail_len), max(int(trail_every), 1)
        self.cid = env.CLIENT
        self.trail_ids, self.trail_head, self.last_pos = [], 0, None
        self.text_id = -1

        for a, b in zip(vertices[:-1], vertices[1:]):
            p.addUserDebugLine(a, b, lineColorRGB=self.REF_COLOR, lineWidth=1.5,
                               lifeTime=0, physicsClientId=self.cid)

        # Massless / collisionless marker so it cannot perturb the drone.
        vs = p.createVisualShape(p.GEOM_SPHERE, radius=0.035,
                                 rgbaColor=[1.0, 0.25, 0.1, 0.85], physicsClientId=self.cid)
        self.target_id = p.createMultiBody(baseMass=0, baseCollisionShapeIndex=-1,
                                           baseVisualShapeIndex=vs,
                                           basePosition=path_ref[0].tolist(),
                                           physicsClientId=self.cid)

        p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0, physicsClientId=self.cid)

        ctr = (path_ref.max(0) + path_ref.min(0)) / 2.0
        span = float((path_ref.max(0) - path_ref.min(0)).max())
        elev, azim = VIEWS.get(getattr(env, '_shape_name', None), (35, -60))
        p.resetDebugVisualizerCamera(cameraDistance=max(span * 1.8, 2.0),
                                     cameraYaw=azim + 90, cameraPitch=-elev,
                                     cameraTargetPosition=ctr.tolist(),
                                     physicsClientId=self.cid)
        self.text_pos = [ctr[0], ctr[1], path_ref[:, 2].max() + 0.35]

    def update(self, step, pos, dev, completed):
        wp = int(min(self.env.wp_counters[0], len(self.path_ref) - 1))
        p.resetBasePositionAndOrientation(self.target_id, self.path_ref[wp].tolist(),
                                          [0, 0, 0, 1], physicsClientId=self.cid)

        if not self.trail_len:
            pass
        elif self.last_pos is not None and step % self.trail_every == 0:
            kw = dict(lineColorRGB=self.TRAIL_COLOR, lineWidth=2.0, lifeTime=0,
                      physicsClientId=self.cid)
            if len(self.trail_ids) < self.trail_len:
                self.trail_ids.append(p.addUserDebugLine(self.last_pos, pos, **kw))
            else:
                self.trail_ids[self.trail_head] = p.addUserDebugLine(
                    self.last_pos, pos,
                    replaceItemUniqueId=self.trail_ids[self.trail_head], **kw)
                self.trail_head = (self.trail_head + 1) % self.trail_len
            self.last_pos = np.array(pos)
        elif self.last_pos is None:
            self.last_pos = np.array(pos)

        if step % 10 == 0:
            self.text_id = p.addUserDebugText(
                f"dev {dev:.3f} m   path {100 * completed:.0f}%",
                self.text_pos, textColorRGB=[0.1, 0.1, 0.1], textSize=1.3,
                lifeTime=0, replaceItemUniqueId=self.text_id, physicsClientId=self.cid)


def run_one(controller, args, vertices, path_ref):
    start = vertices[0]
    env_kwargs = dict(drone_model=DroneModel.CF2X,
                      num_drones=1,
                      initial_xyzs=np.array([start]),
                      initial_rpys=np.array([[0.0, 0.0, 0.0]]),
                      pyb_freq=240,
                      ctrl_freq=40,
                      gui=args.gui,
                      obstacles=False,
                      use_path_controller=True,
                      controller_type=controller,
                      rl_model_path=(args.model if controller == 'rl' else None),
                      rl_algo=args.algo,
                      rl_action_scale=args.rl_action_scale,
                      rl_device=args.device)
    if args.disturbance == 'rotor':
        env = FaultCtrlAviary(fault_rotor=args.fault_rotor, fault_factor=args.fault_factor,
                              fault_time=args.fault_time, **env_kwargs)
    else:
        env = CtrlAviary(**env_kwargs)
    env.PATH_REF = path_ref
    env.wp_counters = np.zeros(env.NUM_DRONES, dtype=int)

    env._shape_name = args.shape
    overlay = GuiOverlay(env, vertices, path_ref, args.trail) if args.gui else None
    step_period = 1.0 / (env.CTRL_FREQ * max(args.speed, 1e-6)) if args.realtime else None

    action = np.zeros((1, 4))  # ignored while the path controller is active
    settle = int(0.25 * env.CTRL_FREQ)  # skip motor spin-up before logging
    max_steps = int(args.duration_sec * env.CTRL_FREQ)
    fault_step = int(args.fault_time * env.CTRL_FREQ)
    wind_vec = np.array([args.wind_impulse, 0.0, 0.0])

    traj = []
    fault_traj_idx = None
    crashed = False
    t_start = time.time()
    for t in range(max_steps):
        if args.disturbance == 'wind' and t == fault_step:
            lin, ang = p.getBaseVelocity(env.DRONE_IDS[0], physicsClientId=env.CLIENT)
            p.resetBaseVelocity(env.DRONE_IDS[0],
                                linearVelocity=(np.array(lin) + wind_vec).tolist(),
                                angularVelocity=ang, physicsClientId=env.CLIENT)
        env.step(action)
        state = env._getDroneStateVector(0)
        if t >= settle:
            if t >= fault_step and fault_traj_idx is None:
                fault_traj_idx = len(traj)
            traj.append(state[0:3].copy())
        if overlay is not None:
            pos = state[0:3]
            overlay.update(t, pos, float(np.linalg.norm(path_ref - pos, axis=1).min()),
                           float(env.wp_counters[0]) / (len(path_ref) - 1))
        if step_period is not None:
            lag = t_start + (t + 1) * step_period - time.time()
            if lag > 0:
                time.sleep(lag)

        if is_crashed(state):
            crashed = True
            break
        if env.wp_counters[0] >= len(path_ref) - 1:
            break
    completed = float(env.wp_counters[0]) / (len(path_ref) - 1)
    env.close()

    traj = np.array(traj) if traj else np.zeros((1, 3))
    dev = cross_track(traj, path_ref)

    result = dict(controller=controller, traj=traj, dev=dev,
                  crashed=crashed, completed=completed, fault_traj_idx=fault_traj_idx)
    if args.disturbance != 'none' and fault_traj_idx is not None:
        recovery = int(args.recovery_sec * env.CTRL_FREQ)
        result['pre_rms'] = _rms(dev[:fault_traj_idx])
        result['peak'] = float(dev.max())
        result['post_rms'] = _rms(dev[fault_traj_idx + recovery:])
        result['fault_pos'] = traj[min(fault_traj_idx, len(traj) - 1)].copy()
    return result


def plot(results, path_ref, args, outpath):
    fig = plt.figure(figsize=(7.5, 6.5))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot(path_ref[:, 0], path_ref[:, 1], path_ref[:, 2],
            color='0.35', lw=1.3, ls='--', label='reference')
    for r in results:
        color = CTRL_COLORS.get(r['controller'], 'tab:red')
        rms = _rms(r['dev'])
        label = f"{r['controller']} (RMS {rms:.3f} m)" + ("  [CRASH]" if r['crashed'] else "")
        ax.plot(r['traj'][:, 0], r['traj'][:, 1], r['traj'][:, 2],
                color=color, lw=1.7, label=label)
        if r.get('fault_pos') is not None:
            fp = r['fault_pos']
            ax.scatter([fp[0]], [fp[1]], [fp[2]], color=color, marker='X', s=70,
                       edgecolors='k', linewidths=0.6, zorder=5)

    allpts = np.vstack([path_ref] + [r['traj'] for r in results])
    ctr = (allpts.max(0) + allpts.min(0)) / 2.0
    rng = (allpts.max(0) - allpts.min(0)).max() / 2.0 + 0.25
    ax.set_xlim(ctr[0] - rng, ctr[0] + rng)
    ax.set_ylim(ctr[1] - rng, ctr[1] + rng)
    ax.set_zlim(ctr[2] - rng, ctr[2] + rng)
    ax.set_box_aspect((1, 1, 1))

    elev, azim = VIEWS.get(args.shape, (30, -60))
    ax.view_init(elev=elev, azim=azim)
    ax.set_xlabel('x [m]')
    ax.set_ylabel('y [m]')
    ax.set_zlabel('z [m]')
    title = f"{args.shape} tracking (size={args.size} m, z={args.z} m)"
    if args.disturbance != 'none':
        dtxt = (f"wind {args.wind_impulse} m/s" if args.disturbance == 'wind'
                else f"rotor {args.fault_rotor} x{args.fault_factor}")
        title += f"\ndisturbance: {dtxt} @ {args.fault_time}s (X = impact)"
    ax.set_title(title)
    ax.legend(loc='upper left', fontsize=8)
    fig.tight_layout()
    fig.savefig(outpath, dpi=140)
    plt.close(fig)


def main(args):
    controllers = [c.strip() for c in args.controller.split(',') if c.strip()]
    vertices = make_shape(args.shape, args)
    path_ref = build_path_ref(vertices)
    os.makedirs(args.out, exist_ok=True)

    # Self-paced waypoint advancing: auto-size from path length unless overridden.
    if args.duration_sec is None:
        arc_len = float(np.linalg.norm(np.diff(path_ref, axis=0), axis=1).sum())
        args.duration_sec = arc_len / 0.2 + 20.0
        print(f"[INFO] auto duration_sec={args.duration_sec:.1f} (path arc length {arc_len:.2f} m)")

    results = []
    for c in controllers:
        print(f"[INFO] shape={args.shape} controller={c} disturbance={args.disturbance} "
              f"({len(path_ref)} dense waypoints, {len(vertices)} vertices)")
        r = run_one(c, args, vertices, path_ref)
        results.append(r)
        if args.disturbance != 'none' and 'peak' in r:
            print(f"  completed={100 * r['completed']:.1f}%  crashed={r['crashed']}  "
                  f"pre_rms={r['pre_rms']:.3f}  peak={r['peak']:.3f}  "
                  f"post_rms={r['post_rms']:.3f} m")
        else:
            print(f"  completed={100 * r['completed']:.1f}%  crashed={r['crashed']}  "
                  f"deviation: mean={r['dev'].mean():.3f}  rms={_rms(r['dev']):.3f}  "
                  f"max={r['dev'].max():.3f} m")

    tag = f"{args.shape}_{'_'.join(controllers)}"
    if args.disturbance != 'none':
        tag += f"_{args.disturbance}"
    outpng = os.path.join(args.out, tag + '.png')
    plot(results, path_ref, args, outpng)
    np.savez(os.path.join(args.out, tag + '.npz'),
             path_ref=path_ref,
             **{f"traj_{r['controller']}": r['traj'] for r in results},
             **{f"dev_{r['controller']}": r['dev'] for r in results})
    print(f"[INFO] figure -> {outpng}")
    print(f"[INFO] arrays -> {os.path.join(args.out, tag + '.npz')}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Shape-tracking evaluation in an empty environment")
    parser.add_argument("--shape", default="hsquare", choices=list(SHAPES.keys()))
    parser.add_argument("--controller", default="rl",
                        help="comma-separated: rl,pid,mppi")
    parser.add_argument("--model", default=None, help="SB3 checkpoint (required if 'rl' in --controller)")
    parser.add_argument("--algo", default="ppo", choices=["ppo", "sac"])
    parser.add_argument("--rl_action_scale", type=float, default=0.05,
                        help="must match the checkpoint's training action_scale")
    parser.add_argument("--size", type=float, default=2.0, help="square side / triangle side / circle diameter / cube side [m]")
    parser.add_argument("--z", type=float, default=1.0, help="altitude (center altitude for vertical/cube shapes) [m]")
    parser.add_argument("--duration_sec", type=float, default=None,
                        help="per-run time budget (default: auto-sized from path length)")
    parser.add_argument("--disturbance", default="none", choices=["none", "wind", "rotor"],
                        help="mid-flight disturbance injected at --fault_time")
    parser.add_argument("--fault_time", type=float, default=4.0, help="disturbance onset [s]")
    parser.add_argument("--wind_impulse", type=float, default=1.5, help="wind gust: +x velocity impulse [m/s]")
    parser.add_argument("--fault_rotor", type=int, default=0, help="rotor fault: which rotor (0-3)")
    parser.add_argument("--fault_factor", type=float, default=0.7, help="rotor fault: RPM scale (0.7 = 30%% loss)")
    parser.add_argument("--recovery_sec", type=float, default=2.0,
                        help="post-fault settling window skipped before measuring recovery RMS")
    parser.add_argument("--out", default="results/shape_eval")
    parser.add_argument("--gui", action="store_true",
                        help="open the PyBullet window with reference/target/trail overlay")
    parser.add_argument("--realtime", action="store_true",
                        help="pace the sim to wall clock (implies --gui usage)")
    parser.add_argument("--speed", type=float, default=1.0,
                        help="--realtime playback rate (0.5 = half speed)")
    parser.add_argument("--trail", type=int, default=300,
                        help="--gui: flown-path trail length in segments (0 disables)")
    parser.add_argument("--device", default="cpu", choices=["auto", "cpu", "cuda"])
    args = parser.parse_args()

    if 'rl' in [c.strip() for c in args.controller.split(',')] and not args.model:
        parser.error("--model is required when 'rl' is in --controller")

    main(args)
