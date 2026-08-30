import numpy as np
from collections import deque
from typing import Optional
from scipy.spatial.transform import Rotation
from gym_pybullet_drones.control.BaseControl import BaseControl


class RLControl(BaseControl):
    """Stable-Baselines3 (PPO/SAC) policy wrapped as a drone controller.

    Same `computeControlFromState` interface as `DSLPIDControl`. The policy must
    have been trained by `WaypointAviary`:

    - observation: ``[pos(3), rpy(3), vel(3), ang_vel(3), action_buffer(4*N),
      target_err(3)]`` (`N` = `ctrl_freq // 2`)
    - action: ``[-1, 1]^4`` decoded as ``HOVER_RPM * (1 + action_scale * action)``

    `action_buffer_size`, `ctrl_freq`, and `action_scale` must match training.
    """

    def __init__(self,
                 drone_model,
                 model_path: str,
                 hover_rpm: float,
                 max_rpm: float,
                 g: float = 9.8,
                 ctrl_freq: int = 40,
                 algo: str = "ppo",
                 action_scale: float = 0.05,
                 device: str = "cpu",
                 collision_checker: Optional[object] = None):
        # BaseControl.__init__ calls self.reset(), so buffer state must exist first.
        self.HOVER_RPM = hover_rpm
        self.MAX_RPM = max_rpm
        self.ACTION_SCALE = action_scale  # must match WaypointAviary at training
        self.ACTION_BUFFER_SIZE = max(int(ctrl_freq // 2), 1)
        self.action_buffer = deque(maxlen=self.ACTION_BUFFER_SIZE)
        self.collision_checker = collision_checker

        super().__init__(drone_model=drone_model, g=g)

        if algo.lower() == "sac":
            from stable_baselines3 import SAC as AlgoCls
        else:
            from stable_baselines3 import PPO as AlgoCls
        # Single-step inference: CPU avoids per-step host<->device transfer.
        self.model = AlgoCls.load(model_path, device=device)
        self.reset()

    def reset(self):
        super().reset()
        self.action_buffer.clear()
        for _ in range(self.ACTION_BUFFER_SIZE):
            self.action_buffer.append(np.zeros(4, dtype=np.float32))

    def computeControlFromState(self,
                                control_timestep,
                                state,
                                target_pos,
                                target_rpy=np.zeros(3),
                                target_vel=np.zeros(3),
                                target_rpy_rates=np.zeros(3),
                                target_traj=None):
        return self.computeControl(control_timestep=control_timestep,
                                   cur_pos=state[0:3],
                                   cur_quat=state[3:7],
                                   cur_vel=state[10:13],
                                   cur_ang_vel=state[13:16],
                                   target_pos=target_pos,
                                   target_rpy=target_rpy)

    def computeControl(self,
                       control_timestep,
                       cur_pos,
                       cur_quat,
                       cur_vel,
                       cur_ang_vel,
                       target_pos,
                       target_rpy=np.zeros(3),
                       target_vel=np.zeros(3),
                       target_rpy_rates=np.zeros(3)):
        try:
            cur_rpy = Rotation.from_quat(cur_quat).as_euler('xyz')
        except Exception:
            cur_rpy = np.zeros(3)

        kin = np.hstack([cur_pos, cur_rpy, cur_vel, cur_ang_vel]).astype(np.float32)
        buffer_flat = np.hstack(list(self.action_buffer)).astype(np.float32)
        target_err = (np.asarray(target_pos, dtype=np.float32) - cur_pos).astype(np.float32)
        obs = np.hstack([kin, buffer_flat, target_err]).reshape(1, -1)

        action, _ = self.model.predict(obs, deterministic=True)
        action = np.asarray(action[0], dtype=np.float32)
        self.action_buffer.append(action.copy())

        rpm = self.HOVER_RPM * (1.0 + self.ACTION_SCALE * action)
        rpm = np.clip(rpm, 0.0, self.MAX_RPM)

        pos_e = np.asarray(target_pos) - cur_pos
        yaw_e = float(target_rpy[2] - cur_rpy[2])
        return rpm.astype(float), pos_e, yaw_e
