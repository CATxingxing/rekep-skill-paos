"""Seeded rejection sampling of non-overlapping table positions."""
from __future__ import annotations

import numpy as np


def sample_positions(rng: np.random.Generator, n: int, x_range, y_range, min_separation: float,
                     exclusions=(), radius=None, tries: int = 5000) -> list[tuple[float, float]]:
    """``n`` xy points in the box, pairwise >= min_separation apart and outside every
    ``(cx, cy, radius)`` exclusion circle.  ``radius=(rmin, rmax)`` limits the distance from the
    robot base (origin) to the arm's comfortably reachable annulus.  Deterministic for a given generator state."""
    points: list[tuple[float, float]] = []
    for _ in range(tries):
        if len(points) == n:
            return points
        p = (float(rng.uniform(*x_range)), float(rng.uniform(*y_range)))
        if radius is not None and not radius[0] <= np.hypot(*p) <= radius[1]:
            continue
        if any(np.hypot(p[0] - q[0], p[1] - q[1]) < min_separation for q in points):
            continue
        if any(np.hypot(p[0] - cx, p[1] - cy) < r for cx, cy, r in exclusions):
            continue
        points.append(p)
    if len(points) == n:
        return points
    raise RuntimeError(f"could not place {n} objects (min separation {min_separation}) in the workspace")
