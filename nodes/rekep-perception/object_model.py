"""Rigid object and static region models built from base-frame depth points.

ReKep keeps a keypoint's identity by registering it on the object it was
proposed on and moving it with that object. Each object here is registered
once per scene revision from its first observation: a point cloud in an
object frame (origin at the estimated center, axes parallel to the base at
registration) and named keypoint offsets in that frame. Later observations
estimate the object pose (translation and yaw about base z) by trimmed ICP of
the visible points against the registered cloud, so partially occluded views
(gripper fingers, a block placed on top) still yield the same physical
keypoints. Objects are assumed to stay upright; tilt is not estimated.
"""
from __future__ import annotations

import math
from typing import Any

import numpy as np

MAX_MODEL_POINTS = 2500
MAX_QUERY_POINTS = 1500


def _rot(yaw: float) -> np.ndarray:
    c, s = math.cos(yaw), math.sin(yaw)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def _subsample(points: np.ndarray, limit: int) -> np.ndarray:
    if len(points) <= limit:
        return points
    index = np.linspace(0, len(points) - 1, limit).round().astype(int)
    return points[index]


def principal_axes(xy: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Unit axes of the minimum-area bounding rectangle, longer side first.

    Unlike the covariance axes these are defined for squares and stay on the
    edges of rectangles, so extreme points land on edge midpoints.
    """
    centered = xy - xy.mean(axis=0)
    best = None
    for degrees in np.arange(0.0, 90.0, 1.0):
        angle = math.radians(degrees)
        a = np.array([math.cos(angle), math.sin(angle)])
        b = np.array([-a[1], a[0]])
        pa, pb = centered @ a, centered @ b
        ea, eb = float(pa.max() - pa.min()), float(pb.max() - pb.min())
        area = ea * eb
        if best is None or area < best[0] - 1e-12:
            best = (area, a, b, ea, eb)
    _, a, b, ea, eb = best
    major, minor = (a, b) if ea >= eb else (b, a)
    # Deterministic sign: major axis points to +x (or +y when vertical).
    if major[0] < -1e-9 or (abs(major[0]) <= 1e-9 and major[1] < 0):
        major = -major
    return major, np.array([-major[1], major[0]])


def extreme_points(points: np.ndarray, center_xy: np.ndarray, axes: tuple[np.ndarray, np.ndarray], fraction: float = 0.03) -> list[tuple[str, np.ndarray]]:
    """Mean of the outermost points along +/- each principal axis.

    On a T these are the stem end, the outer edge of the bar and both bar
    ends; on a rectangle the four edge midpoints.
    """
    result = []
    xy = points[:, :2] - center_xy
    count = max(3, int(len(points) * fraction))
    for name, axis in (("major", axes[0]), ("minor", axes[1])):
        projection = xy @ axis
        order = np.argsort(projection)
        result.append((f"+{name}", points[order[-count:]].mean(axis=0)))
        result.append((f"-{name}", points[order[:count]].mean(axis=0)))
    return result


def fill_ratio(xy: np.ndarray, axes: tuple[np.ndarray, np.ndarray], cell: float = 0.002) -> tuple[float, float, float]:
    """(filled area / oriented bounding-rectangle area, major extent, minor extent).

    Depth pixels sample a face sparsely, so the occupancy grid is closed and
    hole-filled before its area is compared with the bounding rectangle.
    """
    from scipy import ndimage

    local = np.stack([xy @ axes[0], xy @ axes[1]], axis=1)
    lower, upper = local.min(axis=0), local.max(axis=0)
    extent = upper - lower
    grid_cell = max(cell, 0.003)
    index = np.floor((local - lower) / grid_cell).astype(int)
    shape = index.max(axis=0) + 1
    grid = np.zeros(shape + 2, dtype=bool)
    grid[index[:, 0] + 1, index[:, 1] + 1] = True
    grid = ndimage.binary_fill_holes(ndimage.binary_closing(grid, iterations=1))
    return float(grid.sum() / max(1, int(shape[0] * shape[1]))), float(extent[0]), float(extent[1])


def shape_hint(ratio: float, major: float, minor: float) -> str:
    """Coarse top-face shape: rectangle-like, round, or irregular (e.g. T/L)."""
    aspect = minor / max(major, 1e-6)
    if ratio >= 0.86:
        return "box"
    if 0.66 <= ratio < 0.86 and aspect >= 0.8:
        return "cylinder"
    return "irregular"


class RigidModel:
    """One registered object: model cloud, keypoint offsets, current pose."""

    def __init__(self, cloud: np.ndarray, center: np.ndarray, keypoints: list[dict[str, Any]], attributes: dict[str, Any]):
        from scipy.spatial import cKDTree

        self.cloud = _subsample(np.asarray(cloud, dtype=float) - center, MAX_MODEL_POINTS)
        self.tree = cKDTree(self.cloud)
        self.keypoints = keypoints  # each: {"name", "kind", "offset" (object frame), "confidence"}
        self.attributes = attributes
        self.position = np.asarray(center, dtype=float)
        self.yaw = 0.0
        self.residual = 0.0
        self.inliers = 1.0

    def keypoint_positions_by_index(self) -> list[tuple[int, list[float]]]:
        return [(index, position) for index, (_, position) in enumerate(self.keypoint_positions())]

    def keypoint_positions(self) -> list[tuple[dict[str, Any], list[float]]]:
        rotation = _rot(self.yaw)
        return [(item, (self.position + rotation @ np.asarray(item["offset"], dtype=float)).tolist()) for item in self.keypoints]

    def _icp(self, points: np.ndarray, position: np.ndarray, yaw: float, iterations: int = 30, keep: float = 0.8) -> tuple[np.ndarray, float, float, float]:
        position = np.asarray(position, dtype=float).copy()
        residual, inlier_fraction = math.inf, 0.0
        for _ in range(iterations):
            rotation = _rot(yaw)
            local = (points - position) @ rotation  # rotation.T applied row-wise
            distance, index = self.tree.query(local)
            threshold = max(float(np.quantile(distance, keep)), 0.002)
            inlier = distance <= threshold
            if inlier.sum() < 10:
                break
            source = self.cloud[index[inlier]]  # model frame
            target = points[inlier]  # base frame
            a, b = source[:, :2].mean(axis=0), target[:, :2].mean(axis=0)
            h = (source[:, :2] - a).T @ (target[:, :2] - b)
            new_yaw = math.atan2(h[0, 1] - h[1, 0], h[0, 0] + h[1, 1])
            r2 = _rot(new_yaw)[:2, :2]
            new_position = np.empty(3)
            new_position[:2] = b - r2 @ a
            new_position[2] = float(np.mean(target[:, 2] - source[:, 2]))
            change = float(np.linalg.norm(new_position - position)) + abs(math.remainder(new_yaw - yaw, math.tau)) * 0.05
            position, yaw = new_position, new_yaw
            residual = float(distance[inlier].mean())
            inlier_fraction = float((distance <= 0.004).mean())
            if change < 1e-5:
                break
        return position, yaw, residual, inlier_fraction

    def _fit(self, points: np.ndarray, position: np.ndarray, yaw: float) -> tuple[float, float]:
        """(fraction of points within 4 mm of the model at this pose, their mean distance)."""
        distance, _ = self.tree.query((points - position) @ _rot(yaw))
        inlier = distance <= 0.004
        return float(inlier.mean()), float(distance[inlier].mean()) if inlier.any() else math.inf

    def track(self, points: np.ndarray) -> None:
        """Update the pose from the currently visible points of this object.

        An object still explained where it was (most visible points on its
        registered surface) keeps its pose: a partial view of a flat face
        fits equally well anywhere along that face, so ICP alone may slide
        an unmoved, half-occluded object. Otherwise ICP starts from the
        previous pose, the visible-centroid shift and a sweep over yaw (a
        push or a grasp may have turned it); the pose explaining the most
        points wins, and among equal fits the smallest rotation.
        """
        points = _subsample(np.asarray(points, dtype=float), MAX_QUERY_POINTS)
        if len(points) < 10:
            return
        staying, staying_residual = self._fit(points, self.position, self.yaw)
        if staying >= 0.9:
            local = self._icp(points, self.position, self.yaw)
            fit = self._fit(points, local[0], local[1])
            moved = float(np.linalg.norm(local[0] - self.position))
            turned = abs(math.remainder(local[1] - self.yaw, math.tau))
            if fit[0] >= staying and moved <= 0.005 and turned <= 0.05:
                self.position, self.yaw = local[0], math.remainder(local[1], math.tau)
                self.inliers, self.residual = fit
            else:
                self.inliers, self.residual = staying, staying_residual
            return
        visible_shift = points.mean(axis=0) - (self.position + _rot(self.yaw) @ self.cloud.mean(axis=0))
        starts = [(self.position, self.yaw, 30), (self.position + visible_shift, self.yaw, 30)]
        starts += [(self.position + visible_shift, self.yaw + step * math.tau / 24, 15) for step in range(1, 24)]
        best, best_key = None, None
        for position, yaw, iterations in starts:
            candidate = self._icp(points, position, yaw, iterations=iterations)
            fraction, residual = self._fit(points, candidate[0], candidate[1])
            turn = abs(math.remainder(candidate[1] - self.yaw, math.tau))
            key = (round(fraction / 0.02), -(residual + 0.0003 * turn))
            if best_key is None or key > best_key:
                best, best_key = candidate, key
        assert best is not None
        best = self._icp(points, best[0], best[1])
        self.position, self.yaw = best[0], math.remainder(best[1], math.tau)
        self.inliers, self.residual = self._fit(points, self.position, self.yaw)


def filled_cells(xy: np.ndarray, cell: float = 0.003) -> tuple[np.ndarray, np.ndarray]:
    """(centers of the closed, hole-filled occupancy cells, centers of its boundary cells)."""
    from scipy import ndimage

    lower = xy.min(axis=0) - cell
    index = np.floor((xy - lower) / cell).astype(int) + 1
    grid = np.zeros(index.max(axis=0) + 2, dtype=bool)
    grid[index[:, 0], index[:, 1]] = True
    grid = ndimage.binary_fill_holes(ndimage.binary_closing(grid, iterations=1))
    boundary = grid & ~ndimage.binary_erosion(grid)
    to_xy = lambda mask: (np.argwhere(mask) - 1 + 0.5) * cell + lower  # noqa: E731
    return to_xy(grid), to_xy(boundary)


def fit_shape(source: np.ndarray, target: np.ndarray, tolerance: float = 0.004, hidden: Any = None) -> tuple[float, float, float, float]:
    """Planar rigid transform (tx, ty, yaw) placing ``source`` cells onto ``target`` cells.

    Returns (tx, ty, yaw, score); score is the smaller of the fraction of
    source cells on the target and of target cells covered by the source, so
    only a matching outline scores high. ``hidden(points) -> bool array``
    marks points the camera cannot see (e.g. under or behind an object
    standing on the outline): source cells there are unknown rather than off
    the target, so they neither pull the alignment nor lower the score.
    Without it, the visible target's share of the source area is used instead.
    """
    from scipy.spatial import cKDTree

    tree, back = cKDTree(target), None
    best = (0.0, 0.0, 0.0, -1.0)
    for step in range(24):
        yaw = step * math.tau / 24
        rotation = _rot(yaw)[:2, :2]
        t = target.mean(axis=0) - rotation @ source.mean(axis=0)
        for _ in range(40 if hidden is not None else 30):
            moved = source @ rotation.T + t
            distance, index = tree.query(moved)
            use = np.ones(len(moved), dtype=bool) if hidden is None else ~(hidden(moved) & (distance > tolerance))
            if use.sum() < 10:
                break
            keep = use & (distance <= max(float(np.quantile(distance[use], 0.9)), tolerance))
            a, b = source[keep], target[index[keep]]
            ca, cb = a.mean(axis=0), b.mean(axis=0)
            h = (a - ca).T @ (b - cb)
            new_yaw = math.atan2(h[0, 1] - h[1, 0], h[0, 0] + h[1, 1])
            rotation = _rot(new_yaw)[:2, :2]
            new_t = cb - rotation @ ca
            done = float(np.linalg.norm(new_t - t)) < 1e-5 and abs(math.remainder(new_yaw - yaw, math.tau)) < 1e-5
            t, yaw = new_t, new_yaw
            if done:
                break
        moved = source @ rotation.T + t
        on_target = tree.query(moved)[0] <= tolerance
        back = cKDTree(moved)
        covered = float((back.query(target)[0] <= tolerance).mean())
        if hidden is not None:
            score = min(float((on_target | hidden(moved)).mean()), covered)
        else:
            # A partly hidden outline (the object itself may stand on it) can
            # only receive the visible share of the object's cells.
            visible_share = min(1.0, len(target) / max(1, len(source)))
            score = min(float(on_target.mean()) / visible_share, covered)
        if score > best[3] + 1e-9:
            best = (float(t[0]), float(t[1]), math.remainder(yaw, math.tau), score)
    return best
