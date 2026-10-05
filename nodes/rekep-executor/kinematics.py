from __future__ import annotations

import math
from typing import Any

from rekep_core.contracts import ContractError
from rekep_core.geometry import finite_vector


class KinematicGuard:
    def __init__(self, config: dict[str, Any]):
        self.bounds = config["workspace_m"]
        self.maximum_segment = float(config.get("max_segment_length_m", 0.45))

    def validate_pose(self, pose: dict[str, Any]) -> None:
        position = finite_vector(pose.get("position_m"), 3, "pose.position_m")
        quaternion = finite_vector(pose.get("quaternion_xyzw"), 4, "pose.quaternion_xyzw")
        for index, axis in enumerate(("x", "y", "z")):
            lower, upper = self.bounds[axis]
            if not float(lower) <= position[index] <= float(upper):
                raise ContractError(f"pose is outside {axis} workspace bounds")
        norm = math.sqrt(sum(item * item for item in quaternion))
        if not 0.99 <= norm <= 1.01:
            raise ContractError("pose quaternion must be normalized")

    def validate_segment(self, start: list[float], end: list[float]) -> None:
        length = math.dist(start, end)
        if length > self.maximum_segment:
            raise ContractError(f"motion segment length {length:.3f} exceeds limit {self.maximum_segment:.3f}")
