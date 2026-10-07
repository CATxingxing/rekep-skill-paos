"""Small rotation helpers (quaternions are MuJoCo order: w, x, y, z)."""
from __future__ import annotations

import math

import numpy as np


def wrap_angle(angle: float) -> float:
    """Wrap to (-pi, pi]."""
    a = (float(angle) + math.pi) % (2.0 * math.pi) - math.pi
    return math.pi if a == -math.pi else a


def quat_from_yaw(yaw: float) -> np.ndarray:
    return np.array([math.cos(yaw / 2.0), 0.0, 0.0, math.sin(yaw / 2.0)])


def quat_to_mat(q) -> np.ndarray:
    w, x, y, z = (float(v) for v in q)
    n = w * w + x * x + y * y + z * z
    s = 2.0 / n
    return np.array([
        [1 - s * (y * y + z * z), s * (x * y - w * z), s * (x * z + w * y)],
        [s * (x * y + w * z), 1 - s * (x * x + z * z), s * (y * z - w * x)],
        [s * (x * z - w * y), s * (y * z + w * x), 1 - s * (x * x + y * y)],
    ])


def quat_mul(a, b) -> np.ndarray:
    aw, ax, ay, az = a
    bw, bx, by, bz = b
    return np.array([
        aw * bw - ax * bx - ay * by - az * bz,
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
    ])


def quat_conj(q) -> np.ndarray:
    return np.array([q[0], -q[1], -q[2], -q[3]])


def yaw_from_quat(q) -> float:
    """Rotation about world z of a (mostly upright) body."""
    w, x, y, z = (float(v) for v in q)
    return math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


def tilt_from_quat(q) -> float:
    """Angle between the body z axis and world z (0 = upright)."""
    zz = quat_to_mat(q)[2, 2]
    return math.acos(max(-1.0, min(1.0, zz)))


def rotation_angle(qa, qb) -> float:
    """Geodesic angle between two orientations."""
    dot = abs(float(np.dot(qa, qb)) / (np.linalg.norm(qa) * np.linalg.norm(qb)))
    return 2.0 * math.acos(min(1.0, dot))


def rot_z(angle: float) -> np.ndarray:
    c, s = math.cos(angle), math.sin(angle)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
