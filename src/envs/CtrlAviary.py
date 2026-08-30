"""`gym-pybullet-drones` `CtrlAviary` plus an optional path-following harness.

With `use_path_controller=True`, a `'rl'` or `'pid'` controller chases `PATH_REF`
and the `action` passed to `.step()` is ignored. `path_ref` may be assigned later
as `env.PATH_REF`. Derived from upstream `CtrlAviary.py` (MIT).
"""
import numpy as np

from gym_pybullet_drones.envs.CtrlAviary import CtrlAviary as _UpstreamCtrlAviary
from gym_pybullet_drones.utils.enums import DroneModel, Physics


class CtrlAviary(_UpstreamCtrlAviary):
    """`CtrlAviary` with an optional dense-waypoint path-following controller."""

    ################################################################################

    def __init__(self,
                 drone_model: DroneModel = DroneModel.CF2X,
                 num_drones: int = 1,
                 neighbourhood_radius: float = np.inf,
                 initial_xyzs=None,
                 initial_rpys=None,
                 physics: Physics = Physics.PYB,
                 pyb_freq: int = 240,
                 ctrl_freq: int = 240,
                 gui=False,
                 record=False,
                 obstacles=False,
                 user_debug_gui=True,
                 output_folder='results',
                 use_path_controller: bool = False,
                 path_ref=None,
                 path_pid_threshold: float = 0.2,
                 controller_type: str = 'pid',
                 rl_model_path: str = None,
                 rl_algo: str = 'ppo',
                 rl_device: str = 'cpu',
                 rl_action_scale: float = 0.05,
                 ):
        """See `gym_pybullet_drones.envs.CtrlAviary` for the inherited parameters.

        Parameters
        ----------
        use_path_controller : bool, optional
            Drive along `path_ref` instead of applying the `action` given to `.step()`.
        path_ref : ndarray | None, optional
            (N, 3) dense reference path. May also be assigned later as `env.PATH_REF`.
        path_pid_threshold : float, optional
            Distance in meters at which the active waypoint advances.
        controller_type : str, optional
            `'pid'` (`DSLPIDControl`) or `'rl'` (`RLControl`).
        rl_model_path : str | None, optional
            SB3 checkpoint. Required for `controller_type='rl'`.
        rl_algo : str, optional
            `'ppo'` or `'sac'`.
        rl_device : str, optional
            Torch device for policy inference (CPU is the usual default).
        rl_action_scale : float, optional
            Control-authority scale the policy was trained with.

        """
        # Set before super().__init__() so `_preprocessAction` is safe during construction.
        self.USE_PATH_CONTROLLER = use_path_controller
        self.PATH_REF = np.array(path_ref) if path_ref is not None else None
        self.PATH_PID_THRESHOLD = path_pid_threshold
        self.CONTROLLER_TYPE = controller_type.lower() if isinstance(controller_type, str) else 'pid'
        self.controllers = None
        self._control_failure_warned = False

        super().__init__(drone_model=drone_model,
                         num_drones=num_drones,
                         neighbourhood_radius=neighbourhood_radius,
                         initial_xyzs=initial_xyzs,
                         initial_rpys=initial_rpys,
                         physics=physics,
                         pyb_freq=pyb_freq,
                         ctrl_freq=ctrl_freq,
                         gui=gui,
                         record=record,
                         obstacles=obstacles,
                         user_debug_gui=user_debug_gui,
                         output_folder=output_folder
                         )

        self.wp_counters = np.zeros(self.NUM_DRONES, dtype=int)

        # Instantiate even without path_ref; callers often assign PATH_REF afterwards.
        if self.USE_PATH_CONTROLLER:
            if self.CONTROLLER_TYPE == 'pid':
                try:
                    from gym_pybullet_drones.control.DSLPIDControl import DSLPIDControl
                    self.controllers = [DSLPIDControl(drone_model=drone_model) for _ in range(self.NUM_DRONES)]
                    print('[INFO] Path-following PID controller enabled')
                except Exception as e:
                    print(f'[WARNING] DSLPIDControl not available ({e}); path PID disabled')
                    self.USE_PATH_CONTROLLER = False
            elif self.CONTROLLER_TYPE == 'rl':
                if not rl_model_path:
                    print('[WARNING] controller_type="rl" requires rl_model_path; path RL disabled')
                    self.USE_PATH_CONTROLLER = False
                else:
                    try:
                        from src.controllers.RLControl import RLControl
                        self.controllers = [RLControl(drone_model=drone_model,
                                                      model_path=rl_model_path,
                                                      hover_rpm=self.HOVER_RPM,
                                                      max_rpm=self.MAX_RPM,
                                                      g=self.G,
                                                      ctrl_freq=ctrl_freq,
                                                      algo=rl_algo,
                                                      action_scale=rl_action_scale,
                                                      device=rl_device)
                                            for _ in range(self.NUM_DRONES)]
                        print('[INFO] Path-following RL controller enabled')
                    except Exception as e:
                        print(f'[WARNING] RLControl not available/loadable ({e}); path RL disabled')
                        self.USE_PATH_CONTROLLER = False
            else:
                print(f"[WARNING] Unknown controller type '{self.CONTROLLER_TYPE}'; path controller disabled")
                self.USE_PATH_CONTROLLER = False

    ################################################################################

    def _preprocessAction(self,
                          action
                          ):
        """Turn `.step()`'s action into motor RPMs.

        With the path-following harness active, `action` is ignored and RPMs come
        from the selected controller chasing `PATH_REF`. Otherwise defers to upstream.

        Parameters
        ----------
        action : ndarray
            Unbounded input action for each drone.

        Returns
        -------
        ndarray
            (NUM_DRONES, 4) clipped RPMs.

        """
        if not (self.USE_PATH_CONTROLLER and self.controllers is not None and self.PATH_REF is not None):
            return super()._preprocessAction(action)

        rpm_out = np.zeros((self.NUM_DRONES, 4))
        for j in range(self.NUM_DRONES):
            state = self._getDroneStateVector(j)

            wp_idx = int(self.wp_counters[j])
            if wp_idx >= len(self.PATH_REF):
                target_pos = self.PATH_REF[-1]
            else:
                target_pos = self.PATH_REF[wp_idx]
            try:
                rpm_cmd, _, _ = self.controllers[j].computeControlFromState(
                    control_timestep=self.CTRL_TIMESTEP,
                    state=state,
                    target_pos=target_pos
                    )
            except Exception as e:
                if not self._control_failure_warned:
                    print(f'[WARNING] controller raised ({e}); commanding zero RPM')
                    self._control_failure_warned = True
                rpm_cmd = np.zeros(4)
            rpm_out[j, :] = np.clip(rpm_cmd, 0, self.MAX_RPM)

            pos = state[0:3]
            if np.linalg.norm(pos - target_pos) < self.PATH_PID_THRESHOLD:
                if wp_idx < len(self.PATH_REF) - 1:
                    self.wp_counters[j] += 1

        return rpm_out
