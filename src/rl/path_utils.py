"""Shared path densification used by training and the deployed path-following eval."""
import numpy as np


def densify_path(path, factor=5):
    """Arc-length-uniform interpolation to `factor * len(path)` points."""
    num_wp = factor * len(path)
    distances = np.concatenate(([0], np.cumsum(np.linalg.norm(np.diff(path, axis=0), axis=1))))
    new_distances = np.linspace(0, distances[-1], num_wp)
    interpolated = np.zeros((num_wp, 3))
    for dim in range(3):
        interpolated[:, dim] = np.interp(new_distances, distances, path[:, dim])
    return interpolated
