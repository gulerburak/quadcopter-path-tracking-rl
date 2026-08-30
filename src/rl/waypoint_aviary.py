import numpy as np
import pybullet as p
from gymnasium import spaces

from gym_pybullet_drones.envs.BaseRLAviary import BaseRLAviary
from gym_pybullet_drones.utils.enums import DroneModel, Physics, ActionType, ObservationType

from src.rl.path_utils import densify_path


class WaypointAviary(BaseRLAviary):
    """Single-drone env: fly to a random 3D target, or chase a multi-waypoint path.

    Observation is BaseRLAviary kinematics plus `target_err(3)`. Action is RPM in
    `[-1, 1]^4`, decoded as `HOVER_RPM * (1 + action_scale * action)`.

    `num_segments=1` is one static target. `num_segments>1` chains random waypoints,
    densified like `CtrlAviary`/`RLControl`, and advances when within `wp_threshold`.
    `target_err` and the distance reward look `lookahead_dist` ahead on the dense
    path; waypoint advancement still uses the next dense point.
    """

    def __init__(self,
                 drone_model: DroneModel = DroneModel.CF2X,
                 initial_xyzs=None,
                 initial_rpys=None,
                 physics: Physics = Physics.PYB,
                 pyb_freq: int = 240,
                 ctrl_freq: int = 40,
                 gui=False,
                 record=False,
                 obs: ObservationType = ObservationType.KIN,
                 act: ActionType = ActionType.RPM,
                 episode_len_sec: float = 8.0,
                 target_bounds_xy: float = 6.0,
                 target_bounds_z: tuple = (0.5, 2.5),
                 max_tilt: float = 1.0,
                 tilt_penalty_weight: float = 0.5,
                 alive_bonus: float = 1.0,
                 crash_penalty: float = 5.0,
                 num_segments: int = 1,
                 wp_threshold: float = 0.2,
                 raw_step_size: float = 0.2,
                 progress_bonus: float = 2.0,
                 path_complete_bonus: float = 50.0,
                 local_dist_norm: float = 1.0,
                 lookahead_dist: float = 0.75,
                 crash_terminates: bool = False,
                 init_randomization: float = 0.0,
                 action_scale: float = 0.05,
                 rotor_randomization: float = 0.0,
                 seed: int = None,
                 ):
        self.EPISODE_LEN_SEC = episode_len_sec
        self.TARGET_BOUNDS_XY = target_bounds_xy
        self.TARGET_BOUNDS_Z = target_bounds_z
        self.MAX_TILT = max_tilt
        self.TILT_PENALTY_WEIGHT = tilt_penalty_weight
        self.ALIVE_BONUS = alive_bonus
        self.CRASH_PENALTY = crash_penalty
        self.NUM_SEGMENTS = max(int(num_segments), 1)
        self.WP_THRESHOLD = wp_threshold
        self.RAW_STEP_SIZE = raw_step_size
        self.PROGRESS_BONUS = progress_bonus
        self.PATH_COMPLETE_BONUS = path_complete_bonus
        self.LOCAL_DIST_NORM = local_dist_norm
        self.LOOKAHEAD_DIST = lookahead_dist
        self.CRASH_TERMINATES = crash_terminates
        self.INIT_RANDOMIZATION = init_randomization
        self.ACTION_SCALE = action_scale
        self.ROTOR_RANDOMIZATION = rotor_randomization
        self._rotor_scale = np.ones(4)
        self._lookahead_offset = 0
        # BaseAviary.reset() does not seed self.np_random; use our own RNG so
        # reset(seed=...) / --seed actually control target sampling.
        self._target_rng = np.random.default_rng(seed)
        self.TARGET_POS = np.array([0.0, 0.0, 1.0])
        self._dense_path = self.TARGET_POS.reshape(1, 3)
        self._wp_idx = 0
        self._just_advanced = False
        super().__init__(drone_model=drone_model,
                         num_drones=1,
                         initial_xyzs=initial_xyzs,
                         initial_rpys=initial_rpys,
                         physics=physics,
                         pyb_freq=pyb_freq,
                         ctrl_freq=ctrl_freq,
                         gui=gui,
                         record=record,
                         obs=obs,
                         act=act
                         )

    ################################################################################

    def _sampleTarget(self):
        xy = self._target_rng.uniform(-self.TARGET_BOUNDS_XY, self.TARGET_BOUNDS_XY, size=2)
        z = self._target_rng.uniform(self.TARGET_BOUNDS_Z[0], self.TARGET_BOUNDS_Z[1])
        return np.array([xy[0], xy[1], z])

    def _samplePath(self):
        """Chain NUM_SEGMENTS random points from the start, then densify like deployment."""
        current = self.INIT_XYZS[0].copy()
        raw_points = [current]
        for _ in range(self.NUM_SEGMENTS):
            target = self._sampleTarget()
            seg_vec = target - current
            seg_len = np.linalg.norm(seg_vec)
            n_steps = max(int(round(seg_len / self.RAW_STEP_SIZE)), 1)
            for i in range(1, n_steps + 1):
                raw_points.append(current + seg_vec * (i / n_steps))
            current = target
        return densify_path(np.array(raw_points), factor=5)

    def reset(self, seed=None, options=None):
        if seed is not None:
            self._target_rng = np.random.default_rng(seed)
        if self.ROTOR_RANDOMIZATION > 0.0:
            self._rotor_scale = np.ones(4)
            idx = int(self._target_rng.integers(0, 4))
            self._rotor_scale[idx] = 1.0 - self._target_rng.uniform(0.0, self.ROTOR_RANDOMIZATION)
        else:
            self._rotor_scale = np.ones(4)
        if self.NUM_SEGMENTS <= 1:
            self.TARGET_POS = self._sampleTarget()
            self._dense_path = self.TARGET_POS.reshape(1, 3)
            self._lookahead_offset = 0
        else:
            self._dense_path = self._samplePath()
            self.TARGET_POS = self._dense_path[-1].copy()
            # Dense points are uniformly spaced, so an index offset ≈ LOOKAHEAD_DIST.
            spacing = np.linalg.norm(self._dense_path[1] - self._dense_path[0])
            self._lookahead_offset = max(int(round(self.LOOKAHEAD_DIST / max(spacing, 1e-6))), 1)
        self._wp_idx = 0
        self._just_advanced = False
        obs, info = super().reset(seed=seed, options=options)
        if self.INIT_RANDOMIZATION > 0.0:
            obs, info = self._randomizeInitialState()
        return obs, info

    def _randomizeInitialState(self):
        """Perturb initial pose/velocity (scaled by INIT_RANDOMIZATION) and refresh obs.

        At scale 1.0: +-0.3 m, +-0.3 rad rpy, +-1.0 m/s horizontal / +-0.5 m/s
        vertical, +-1.0 rad/s. Spawn altitude is lifted so a downward start can recover.
        """
        scale = self.INIT_RANDOMIZATION
        rng = self._target_rng
        base = self.INIT_XYZS[0].copy()
        base[2] = max(base[2], 0.6)  # room to recover a downward start
        pos = base + rng.uniform(-0.3, 0.3, size=3) * scale
        pos[2] = max(pos[2], 0.4)  # hard floor, above the crash floor
        drpy = rng.uniform(-0.3, 0.3, size=3) * scale
        quat = p.getQuaternionFromEuler(drpy.tolist())
        vel = rng.uniform(-1.0, 1.0, size=3) * scale
        vel[2] *= 0.5  # gentler vertical velocity
        ang_v = (rng.uniform(-1.0, 1.0, size=3) * scale).tolist()
        p.resetBasePositionAndOrientation(self.DRONE_IDS[0], pos.tolist(), quat, physicsClientId=self.CLIENT)
        p.resetBaseVelocity(self.DRONE_IDS[0], linearVelocity=vel.tolist(), angularVelocity=ang_v, physicsClientId=self.CLIENT)
        self._updateAndStoreKinematicInformation()
        self._wp_idx = 0
        self._just_advanced = False
        return self._computeObs(), self._computeInfo()

    def _preprocessAction(self, action):
        """Decode action to RPM with ACTION_SCALE and optional per-episode rotor loss.

        ACTION_SCALE must match RLControl at deployment. The action buffer still stores
        the commanded action; the rotor fault is unmodeled.
        """
        rpm = super()._preprocessAction(action)
        if self.ACTION_SCALE != 0.05:
            rpm = self.HOVER_RPM + (self.ACTION_SCALE / 0.05) * (rpm - self.HOVER_RPM)
            rpm = np.clip(rpm, 0.0, self.MAX_RPM)
        if self.ROTOR_RANDOMIZATION > 0.0:
            rpm = rpm * self._rotor_scale
        return rpm

    def _lookaheadTarget(self):
        """Point lookahead_offset dense indices past the active waypoint, capped at the end."""
        idx = min(self._wp_idx + self._lookahead_offset, len(self._dense_path) - 1)
        return self._dense_path[idx]

    ################################################################################

    def _observationSpace(self):
        """Base kinematic observation plus relative target error (3 dims)."""
        base_space = super()._observationSpace()
        extra_lo = -np.inf * np.ones((self.NUM_DRONES, 3), dtype=np.float32)
        extra_hi = np.inf * np.ones((self.NUM_DRONES, 3), dtype=np.float32)
        lo = np.hstack([base_space.low, extra_lo])
        hi = np.hstack([base_space.high, extra_hi])
        return spaces.Box(low=lo, high=hi, dtype=np.float32)

    def _advanceWaypoint(self):
        """Advance `_wp_idx` when within WP_THRESHOLD of the active dense waypoint."""
        pos = self._getDroneStateVector(0)[0:3]
        self._just_advanced = False
        if np.linalg.norm(pos - self._dense_path[self._wp_idx]) < self.WP_THRESHOLD:
            if self._wp_idx < len(self._dense_path) - 1:
                self._wp_idx += 1
                self._just_advanced = True

    def _computeObs(self):
        # Advance before reward so this step's waypoint / _just_advanced is visible.
        self._advanceWaypoint()
        obs = super()._computeObs()
        pos = self._getDroneStateVector(0)[0:3]
        target_err = (self._lookaheadTarget() - pos).reshape(1, 3).astype(np.float32)
        return np.hstack([obs, target_err]).astype(np.float32)

    ################################################################################

    def _isDiverged(self):
        """True if out of bounds or tilted past MAX_TILT."""
        state = self._getDroneStateVector(0)
        pos = state[0:3]
        rpy = state[7:10]
        margin = 2.0
        out_of_bounds = (abs(pos[0]) > self.TARGET_BOUNDS_XY + margin
                          or abs(pos[1]) > self.TARGET_BOUNDS_XY + margin
                          or pos[2] > self.TARGET_BOUNDS_Z[1] + margin
                          or pos[2] < 0.02)
        excessive_tilt = abs(rpy[0]) > self.MAX_TILT or abs(rpy[1]) > self.MAX_TILT
        return out_of_bounds or excessive_tilt

    def _computeReward(self):
        """Alive bonus minus scale-normalized distance, tilt, and velocity.

        Distance is normalized by TARGET_BOUNDS_XY (static target) or LOCAL_DIST_NORM
        (path). Path mode also adds PROGRESS_BONUS on waypoint advance and
        PATH_COMPLETE_BONUS at the last waypoint.
        """
        state = self._getDroneStateVector(0)
        pos = state[0:3]
        rpy = state[7:10]
        vel = state[10:13]
        local_target = self._lookaheadTarget()
        dist = np.linalg.norm(local_target - pos)
        if self.NUM_SEGMENTS <= 1:
            dist_norm = dist / max(self.TARGET_BOUNDS_XY, 1e-3)
        else:
            dist_norm = dist / max(self.LOCAL_DIST_NORM, 1e-3)
        tilt_penalty = self.TILT_PENALTY_WEIGHT * (rpy[0] ** 2 + rpy[1] ** 2)
        reward = self.ALIVE_BONUS - dist_norm - 0.05 * np.linalg.norm(vel) - tilt_penalty
        if dist < 0.15:
            reward += 10.0
        if self.NUM_SEGMENTS > 1 and self._just_advanced:
            reward += self.PROGRESS_BONUS
            if self._wp_idx == len(self._dense_path) - 1:
                reward += self.PATH_COMPLETE_BONUS
        if self._isDiverged():
            reward -= self.CRASH_PENALTY
        return reward

    ################################################################################

    def _computeTerminated(self):
        # Crash is a true terminal state; default off keeps older checkpoints reproducible.
        return bool(self.CRASH_TERMINATES and self._isDiverged())

    def _computeTruncated(self):
        if self._isDiverged() and not self.CRASH_TERMINATES:
            return True
        if self.step_counter / self.PYB_FREQ > self.EPISODE_LEN_SEC:
            return True
        return False

    ################################################################################

    def _computeInfo(self):
        state = self._getDroneStateVector(0)
        dist = float(np.linalg.norm(self.TARGET_POS - state[0:3]))
        info = {"target_pos": self.TARGET_POS.copy(), "dist_to_target": dist}
        if self.NUM_SEGMENTS > 1:
            info["wp_idx"] = int(self._wp_idx)
            info["num_waypoints"] = len(self._dense_path)
            info["dist_to_local_target"] = float(np.linalg.norm(self._dense_path[self._wp_idx] - state[0:3]))
        return info
