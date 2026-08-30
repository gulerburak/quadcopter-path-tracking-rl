"""Deterministic geometric reference paths for shape-tracking evaluation.

Each shape is an ordered closed loop of vertices. `build_path_ref()` subdivides
edges to `raw_step_size` then densifies 5x, matching the deployed `PATH_REF`.
"""
import numpy as np

from src.rl.path_utils import densify_path


def _close(vertices):
    """Append the first vertex if needed so the loop is closed."""
    v = np.asarray(vertices, dtype=float)
    if not np.allclose(v[0], v[-1]):
        v = np.vstack([v, v[0]])
    return v


def horizontal_square(center=(0.0, 0.0), z=1.0, side=2.0):
    """Axis-aligned square in the horizontal plane at altitude `z`."""
    cx, cy = center
    h = side / 2.0
    return _close([[cx - h, cy - h, z], [cx + h, cy - h, z],
                   [cx + h, cy + h, z], [cx - h, cy + h, z]])


def vertical_square(center=(0.0, 0.0), z0=1.5, side=1.5, axis='x'):
    """Square in a vertical plane (x-z if axis='x', else y-z), centered at `z0`."""
    cx, cy = center
    h = side / 2.0
    if axis == 'x':
        return _close([[cx - h, cy, z0 - h], [cx + h, cy, z0 - h],
                       [cx + h, cy, z0 + h], [cx - h, cy, z0 + h]])
    return _close([[cx, cy - h, z0 - h], [cx, cy + h, z0 - h],
                   [cx, cy + h, z0 + h], [cx, cy - h, z0 + h]])


def triangle(center=(0.0, 0.0), z=1.0, size=2.0):
    """Equilateral triangle of side `size` in the horizontal plane."""
    cx, cy = center
    r = size / np.sqrt(3.0)  # circumradius for side `size`
    angs = np.deg2rad([90.0, 210.0, 330.0])
    return _close([[cx + r * np.cos(a), cy + r * np.sin(a), z] for a in angs])


def circle(center=(0.0, 0.0), z=1.0, radius=1.0, n=64, plane='xy'):
    """Circle approximated by `n` chords. `plane`: 'xy', 'xz', or 'yz'."""
    cx, cy = center
    t = np.linspace(0.0, 2.0 * np.pi, n, endpoint=False)
    if plane == 'xy':
        pts = np.stack([cx + radius * np.cos(t), cy + radius * np.sin(t), np.full(n, z)], axis=1)
    elif plane == 'xz':
        pts = np.stack([cx + radius * np.cos(t), np.full(n, cy), z + radius * np.sin(t)], axis=1)
    else:  # 'yz'
        pts = np.stack([np.full(n, cx), cy + radius * np.cos(t), z + radius * np.sin(t)], axis=1)
    return _close(pts)


def cube_edges(center=(0.0, 0.0), z0=1.5, side=1.5):
    """Closed cycle of the 8 cube corners along real cube edges."""
    h = side / 2.0
    cx, cy = center
    seq = [(-1, -1, -1), (1, -1, -1), (1, 1, -1), (-1, 1, -1),
           (-1, 1, 1), (1, 1, 1), (1, -1, 1), (-1, -1, 1)]
    return _close([[cx + h * sx, cy + h * sy, z0 + h * sz] for sx, sy, sz in seq])


SHAPES = {
    'hsquare': horizontal_square,
    'vsquare': vertical_square,
    'triangle': triangle,
    'circle': circle,
    'circle_v': lambda **k: circle(plane='xz', **k),
    'cube': cube_edges,
}


def subdivide(vertices, raw_step_size=0.2):
    """Insert nodes so consecutive points are at most `raw_step_size` apart."""
    v = np.asarray(vertices, dtype=float)
    out = [v[0]]
    for a, b in zip(v[:-1], v[1:]):
        seg = b - a
        length = np.linalg.norm(seg)
        n = max(int(np.ceil(length / raw_step_size)), 1)
        for i in range(1, n + 1):
            out.append(a + seg * (i / n))
    return np.array(out)


def build_path_ref(vertices, raw_step_size=0.2, factor=5):
    """Vertices -> dense `PATH_REF` (subdivide, then densify `factor`x)."""
    return densify_path(subdivide(vertices, raw_step_size), factor=factor)
