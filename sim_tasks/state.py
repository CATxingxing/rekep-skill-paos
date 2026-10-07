"""Ground-truth snapshots handed to evaluators (never to the policy)."""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class BodyState:
    pos: np.ndarray       # world position of the body frame origin
    quat: np.ndarray      # w, x, y, z
    linvel: np.ndarray    # world frame
    angvel: np.ndarray    # world frame


@dataclass
class Contact:
    geom1: str
    geom2: str
    body1: str
    body2: str
    force: float          # normal force magnitude (N)
    dist: float


@dataclass
class SceneState:
    time: float
    bodies: dict[str, BodyState]
    contacts: list[Contact] = field(default_factory=list)
    pinch_pos: np.ndarray | None = None
    gripper_ctrl: float = 0.0

    def contacts_between(self, body_a: str, body_b: str, min_force: float = 0.0) -> list[Contact]:
        pair = {body_a, body_b}
        return [c for c in self.contacts if {c.body1, c.body2} == pair and c.force >= min_force]

    def gripper_contacts(self, body: str) -> list[Contact]:
        """Contacts between the gripper (any r2f85_* body) and ``body``."""
        out = []
        for c in self.contacts:
            if c.body1 == body and c.body2.startswith("r2f85"):
                out.append(c)
            elif c.body2 == body and c.body1.startswith("r2f85"):
                out.append(c)
        return out

    def robot_contacts(self, body: str) -> list[Contact]:
        """Contacts between any robot link / gripper body and ``body``."""
        def robot(name: str) -> bool:
            return name.startswith(("r2f85", "Link", "base_link"))
        return [c for c in self.contacts
                if (c.body1 == body and robot(c.body2)) or (c.body2 == body and robot(c.body1))]
