from __future__ import annotations

import math
from typing import Iterable


def finite_vector(value: object, size: int, label: str) -> list[float]:
    if not isinstance(value, (list, tuple)) or len(value) != size:
        raise ValueError(f"{label} must contain {size} values")
    result = []
    for index, item in enumerate(value):
        if isinstance(item, bool) or not isinstance(item, (int, float)):
            raise ValueError(f"{label}[{index}] must be numeric")
        parsed = float(item)
        if not math.isfinite(parsed):
            raise ValueError(f"{label}[{index}] must be finite")
        result.append(parsed)
    return result


def transform_point(matrix: Iterable[Iterable[float]], point: object) -> list[float]:
    rows = [finite_vector(row, 4, "transform row") for row in matrix]
    if len(rows) != 4:
        raise ValueError("transform must be 4x4")
    homogeneous = (*finite_vector(point, 3, "point"), 1.0)
    result = [sum(rows[row][column] * homogeneous[column] for column in range(4)) for row in range(3)]
    weight = sum(rows[3][column] * homogeneous[column] for column in range(4))
    if abs(weight) < 1e-12:
        raise ValueError("transform produced a zero homogeneous weight")
    return [item / weight for item in result]


def distance(left: object, right: object) -> float:
    a = finite_vector(left, 3, "left")
    b = finite_vector(right, 3, "right")
    return math.sqrt(sum((a[index] - b[index]) ** 2 for index in range(3)))
