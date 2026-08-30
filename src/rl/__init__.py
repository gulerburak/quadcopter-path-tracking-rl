"""RL training environment and scripts for the AE4350 quadcopter controller.

`WaypointAviary` trains a Stable-Baselines3 policy; `RLControl` loads it behind
upstream's `BaseControl` interface.
"""
from .waypoint_aviary import WaypointAviary

__all__ = ["WaypointAviary"]
