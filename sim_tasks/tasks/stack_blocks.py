"""Stack Blocks: stack three cubes in a given colour order on a green zone."""
from __future__ import annotations

import itertools
from typing import Any

import numpy as np

from ..env import TaskEnv
from ..evaluator import BaseEvaluator, FailureReason, is_inside_region, probe_stats, result, upright_error
from ..mathutil import quat_from_yaw
from ..sampling import sample_positions
from ..scene import free_body, geom
from ..state import SceneState

ZONE_RGBA = (0.10, 0.85, 0.20, 1.0)   # same green as the baseline place_zone; perception tracks this hue


class StackEvaluator(BaseEvaluator):
    """Pair (upper on lower) is valid when XY aligned, Z offset == block size, in contact, both upright.

    success: all pairs of the requested order valid, still valid and at rest after
    a probe with the arm held still, bottom block on the zone, gripper released.
    """

    needs_probe = True

    def reset(self, state: SceneState) -> None:
        self.names = [o["name"] for o in self.plan["blocks"]]
        self.order = list(self.plan["order"])           # bottom -> top
        self.size = float(self.plan["block_size"])
        self.table_top = float(self.common["table"]["top_z"])
        self.zone = self.plan["zone"]
        self.pair_first_valid: dict[str, float] = {}
        self.max_chain = 1                              # longest valid tower seen so far (blocks)
        self.sequence: list[str] = []                   # order in which valid pairs first appeared

    # -- geometry ------------------------------------------------------------
    def pair_metrics(self, state: SceneState, upper: str, lower: str) -> dict[str, Any]:
        cfg = self.cfg
        u, l = state.bodies[upper], state.bodies[lower]
        xy_error = float(np.linalg.norm(u.pos[:2] - l.pos[:2]))
        z_error = float(u.pos[2] - l.pos[2] - self.size)
        force = sum(c.force for c in state.contacts_between(upper, lower))   # all contact points of the pair
        in_contact = bool(state.contacts_between(upper, lower, cfg["contact_min_force_n"]))
        tilt = max(upright_error(u.quat), upright_error(l.quat))
        valid = (xy_error <= cfg["xy_tolerance_m"] and abs(z_error) <= cfg["z_tolerance_m"]
                 and in_contact and tilt <= cfg["max_tilt_rad"])
        return {"upper": upper, "lower": lower, "xy_error": round(xy_error, 4), "z_error": round(z_error, 4),
                "contact_force": round(force, 3), "max_tilt_rad": round(tilt, 3), "valid": valid}

    def valid_pairs(self, state: SceneState) -> set[tuple[str, str]]:
        return {(a, b) for a, b in itertools.permutations(self.names, 2) if self.pair_metrics(state, a, b)["valid"]}

    def observe(self, state: SceneState) -> None:
        for upper, lower in self.valid_pairs(state):
            key = f"{upper}->{lower}"
            if key not in self.pair_first_valid:
                self.pair_first_valid[key] = state.time
                self.sequence.append(key)
        below = dict(self.valid_pairs(state))              # upper -> lower
        for name in below:
            length, cur = 1, name
            while cur in below and length <= len(self.names):
                cur, length = below[cur], length + 1
            self.max_chain = max(self.max_chain, length)

    # -- evaluation ----------------------------------------------------------
    def _chain_pairs(self, order: list[str]) -> list[tuple[str, str]]:
        return [(order[i + 1], order[i]) for i in range(len(order) - 1)]

    def evaluate(self, state: SceneState, probe=None, timed_out: bool = False) -> dict[str, Any]:
        cfg = self.cfg
        last = probe[-1] if probe else state
        order = self.order
        if not self.plan["require_order"]:       # any complete stack counts
            order = max(itertools.permutations(self.names), key=lambda p: sum(
                self.pair_metrics(state, a, b)["valid"] for a, b in self._chain_pairs(list(p))))
            order = list(order)
        pairs = self._chain_pairs(order)
        now = [self.pair_metrics(state, a, b) for a, b in pairs]
        later = [self.pair_metrics(last, a, b) for a, b in pairs]
        stable_pairs = [n["valid"] and l["valid"] for n, l in zip(now, later)]
        n_valid = sum(stable_pairs)
        stats = probe_stats(probe, self.names) if probe else {"max_linear_speed": None, "max_angular_speed": None,
                                                              "max_pose_drift": None}
        at_rest = probe is not None and (stats["max_linear_speed"] <= cfg["max_linear_speed"]
                                         and stats["max_angular_speed"] <= cfg["max_angular_speed"]
                                         and stats["max_pose_drift"] <= cfg["max_pose_drift_m"])
        base = state.bodies[order[0]]
        z_ok = abs(float(base.pos[2]) - (self.table_top + self.size / 2)) <= cfg["z_tolerance_m"]
        in_zone = is_inside_region(base.pos, self.zone["center"], [self.zone["half"]] * 2,
                                   margin=cfg["zone_margin_m"]) and z_ok
        released = not any(state.gripper_contacts(n) for n in self.names)
        wrong_pairs = sorted(f"{a}->{b}" for a, b in self.valid_pairs(state) if (a, b) not in set(pairs))
        tilted = [n for n in self.names if upright_error(state.bodies[n].quat) > cfg["max_tilt_rad"]]
        fallen = [n for n in self.names
                  if state.bodies[n].pos[2] < self.table_top + self.size / 2 - cfg["fallen_margin_m"]]
        full = n_valid == len(pairs)
        zone_ok = in_zone or not self.plan["require_base_in_zone"]
        released_ok = released or not cfg["require_released"]
        success = bool(full and at_rest and zone_ok and released_ok)
        metrics = {
            "order_bottom_to_top": order,
            "pairs": now,
            "pairs_after_probe": later,
            "stable_pairs": n_valid,
            "num_blocks": len(self.names),
            "base_in_zone": bool(in_zone),
            "released": bool(released),
            "at_rest": bool(at_rest),
            **stats,
            "valid_sequence": list(self.sequence),
            "wrong_pairs": wrong_pairs,
            "max_tower_height_blocks": self.max_chain,
            "tilted_blocks": tilted,
            "fallen_blocks": fallen,
        }
        if success:
            score = 1.0
        elif full:
            score = cfg["two_pair_unfinished_score"]
        else:
            score = cfg["partial_score"] if n_valid >= 1 else 0.0
        reason = None
        if not success:
            if fallen:
                reason = FailureReason.OBJECT_DROPPED
            elif wrong_pairs and n_valid < len(pairs):
                reason = FailureReason.SEQUENCE_ERROR
            elif tilted or (all(n["valid"] for n in now) and not at_rest and probe is not None) or \
                    any(n["valid"] and not l["valid"] for n, l in zip(now, later)):
                reason = FailureReason.STACK_UNSTABLE
            elif full and at_rest and not released_ok:
                reason = FailureReason.NOT_RELEASED
            elif timed_out:
                reason = FailureReason.TIMEOUT
            else:
                reason = FailureReason.PLACEMENT_ERROR
        return result(success, score, metrics, reason, partial=n_valid >= 1)


class StackBlocksEnv(TaskEnv):
    TASK = "stack_blocks"

    def plan_scene(self, rng, overrides):
        cfg, common = self.cfg, self.common
        overrides = overrides or {}
        size = float(cfg["block_size_m"])
        names = list(cfg["colors"])[: int(cfg["num_blocks"])]
        order = [names[i] for i in rng.permutation(len(names))]       # bottom -> top colour order
        zone = cfg["zone"]
        zc = (float(rng.uniform(*zone["center_x"])), float(rng.uniform(*zone["center_y"])))
        ws = common["workspace"]
        points = sample_positions(rng, len(names), ws["x"], ws["y"], cfg["min_center_separation_m"],
                                  exclusions=[(zc[0], zc[1], zone["half_size_m"] + 0.065)], radius=ws["radius"])
        yaws = rng.uniform(-np.pi / 4, np.pi / 4, len(names))
        blocks = []
        for i, color in enumerate(names):
            blocks.append({"name": f"cube_{color}", "color": color, "rgba": list(cfg["colors"][color]) + [1.0],
                           "pos": [points[i][0], points[i][1], common["table"]["top_z"] + size / 2 + 1e-4],
                           "yaw": float(yaws[i])})
        return {"blocks": blocks, "order": [f"cube_{c}" for c in order], "order_colors": order,
                "block_size": size, "block_mass": float(cfg["block_mass_kg"]),
                "zone": {"center": list(zc), "half": float(zone["half_size_m"])},
                "require_order": bool(overrides.get("require_order", cfg["require_order"])),
                "require_base_in_zone": bool(overrides.get("require_base_in_zone", cfg["require_base_in_zone"]))}

    def fill_scene(self, world, plan):
        phys, col = self.common["object_physics"], self.common["collision"]["object"]
        size, top = plan["block_size"], self.common["table"]["top_z"]
        zc, half = plan["zone"]["center"], plan["zone"]["half"]
        geom(world, "stack_zone", "box", (half, half, 0.0015), pos=(zc[0], zc[1], top + 0.0015),
             rgba=ZONE_RGBA, contype=0, conaffinity=0)
        for block in plan["blocks"]:
            body = free_body(world, block["name"], block["pos"], quat_from_yaw(block["yaw"]))
            geom(body, f"{block['name']}_geom", "box", [size / 2] * 3, rgba=block["rgba"], mass=plan["block_mass"],
                 friction=phys["friction"], contype=col["contype"], conaffinity=col["conaffinity"], condim=phys["condim"])

    def tracked_bodies(self, plan):
        return [b["name"] for b in plan["blocks"]]

    def make_evaluator(self, plan):
        return StackEvaluator(self.cfg["evaluator"], plan, self.common)

    def instruction_for(self, plan):
        c = plan["order_colors"]
        zone = " on the green square" if plan["require_base_in_zone"] else ""
        if not plan["require_order"]:
            return f"Stack the three cubes into one tower{zone}."
        return (f"Stack the three cubes into one tower{zone}: the {c[0]} cube at the bottom, "
                f"the {c[1]} cube in the middle and the {c[2]} cube on top.")
