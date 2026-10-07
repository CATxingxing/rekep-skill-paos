"""General Pickup: grasp the language-specified object among distractors and lift it."""
from __future__ import annotations

from typing import Any

import numpy as np

from ..env import TaskEnv
from ..evaluator import BaseEvaluator, FailureReason, result
from ..mathutil import quat_from_yaw
from ..sampling import sample_positions
from ..scene import free_body, geom
from ..state import SceneState


class PickupEvaluator(BaseEvaluator):
    """success: target z - initial z >= lift_height_m, held continuously for hold_time_s."""

    PADS = ("left_pad", "right_pad")

    def reset(self, state: SceneState) -> None:
        self.target = self.plan["target"]
        self.names = [o["name"] for o in self.plan["objects"]]
        self.z0 = {n: float(state.bodies[n].pos[2]) for n in self.names}
        self.xy0 = {n: state.bodies[n].pos[:2].copy() for n in self.names}
        self.max_lift = 0.0
        self.max_hold = 0.0
        self.hold_start: float | None = None
        self.success_time: float | None = None
        self.grasp_achieved = False
        self.carried = False
        self.dropped = 0
        self.touched = False
        self.wrong_max_lift = {n: 0.0 for n in self.names if n != self.target}
        self.wrong_max_disp = {n: 0.0 for n in self.names if n != self.target}

    def observe(self, state: SceneState) -> None:
        cfg, t = self.cfg, state.time
        body = state.bodies[self.target]
        lift = float(body.pos[2]) - self.z0[self.target]
        self.max_lift = max(self.max_lift, lift)
        if lift >= cfg["lift_height_m"]:
            if self.hold_start is None:
                self.hold_start = t
            held = t - self.hold_start
            self.max_hold = max(self.max_hold, held)
            if held >= cfg["hold_time_s"] and self.success_time is None:
                self.success_time = t
        else:
            self.hold_start = None
        contacts = state.gripper_contacts(self.target)
        if contacts:
            self.touched = True
        sides = {p for c in contacts if c.force >= 0.5 for p in self.PADS if p in c.geom1 or p in c.geom2}
        if len(sides) == len(self.PADS):
            self.grasp_achieved = True
        if lift >= cfg["drop_lift_height_m"]:
            self.carried = True
        elif self.carried and lift < cfg["drop_rest_height_m"] and self.success_time is None:
            self.dropped += 1
            self.carried = False
        for name in self.wrong_max_lift:
            other = state.bodies[name]
            self.wrong_max_lift[name] = max(self.wrong_max_lift[name], float(other.pos[2]) - self.z0[name])
            self.wrong_max_disp[name] = max(self.wrong_max_disp[name],
                                            float(np.linalg.norm(other.pos[:2] - self.xy0[name])))

    def evaluate(self, state: SceneState, probe=None, timed_out: bool = False) -> dict[str, Any]:
        cfg = self.cfg
        lift_now = float(state.bodies[self.target].pos[2]) - self.z0[self.target]
        moved = sorted(n for n in self.wrong_max_lift
                       if self.wrong_max_disp[n] > cfg["wrong_object_moved_m"]
                       or self.wrong_max_lift[n] > cfg["wrong_object_moved_m"])
        wrong_lifted = sorted(n for n, v in self.wrong_max_lift.items() if v >= cfg["wrong_object_lift_m"])
        success = self.success_time is not None
        metrics = {
            "target": self.target,
            "max_lift_height": round(self.max_lift, 4),
            "current_lift_height": round(lift_now, 4),
            "max_hold_s": round(self.max_hold, 3),
            "grasp_achieved": self.grasp_achieved,
            "object_dropped": self.dropped > 0,
            "drop_count": self.dropped,
            "wrong_object_moved": bool(moved),
            "wrong_objects_moved": moved,
            "wrong_objects_lifted": wrong_lifted,
            "time_to_success_s": self.success_time,
        }
        w = cfg["score_weights"]
        if success:
            score = 1.0
        elif wrong_lifted and self.max_lift < cfg["drop_lift_height_m"]:
            score = 0.0
        else:
            score = (w["grasp"] * self.grasp_achieved
                     + w["lift"] * min(1.0, max(0.0, self.max_lift) / cfg["lift_height_m"])
                     + w["hold"] * min(1.0, self.max_hold / cfg["hold_time_s"]))
        reason = None
        if not success:
            if wrong_lifted and self.max_lift < cfg["drop_lift_height_m"]:
                reason = FailureReason.WRONG_OBJECT
            elif self.dropped:
                reason = FailureReason.OBJECT_DROPPED
            elif timed_out:
                reason = FailureReason.TIMEOUT
            elif self.max_lift < cfg["lift_height_m"]:
                reason = FailureReason.GRASP_FAILED
            else:
                reason = FailureReason.UNKNOWN
        partial = (not success) and score > 0.0 and (self.grasp_achieved or self.max_lift >= cfg["drop_lift_height_m"])
        return result(success, score, metrics, reason, partial)


class GeneralPickupEnv(TaskEnv):
    TASK = "general_pickup"

    def plan_scene(self, rng, overrides):
        cfg, common = self.cfg, self.common
        overrides = overrides or {}
        n = int(overrides.get("num_objects", cfg["num_objects"]))
        colors = list(cfg["colors"])
        shapes = list(cfg["shapes"])
        color_pick = [colors[i] for i in rng.permutation(len(colors))[:n]]          # object ordering/identity
        shape_pick = [shapes[int(rng.integers(len(shapes)))] for _ in range(n)]
        ws = common["workspace"]
        points = sample_positions(rng, n, ws["x"], ws["y"], cfg["min_center_separation_m"], radius=ws["radius"])
        yaws = rng.uniform(-np.pi / 2, np.pi / 2, n)
        target = int(overrides.get("target_index", rng.integers(n)))
        objects = []
        for i in range(n):
            spec = cfg["shapes"][shape_pick[i]]
            half_z = spec["half"][2] if spec["type"] == "box" else spec["half_height"]
            objects.append({
                "name": f"obj_{color_pick[i]}_{shape_pick[i]}", "color": color_pick[i], "shape": shape_pick[i],
                "type": spec["type"], "mass": spec["mass"],
                "half": list(spec["half"]) if spec["type"] == "box" else [spec["radius"], spec["half_height"]],
                "rgba": list(cfg["colors"][color_pick[i]]) + [1.0],
                "pos": [points[i][0], points[i][1], common["table"]["top_z"] + half_z + 1e-4],
                "yaw": float(yaws[i]),
            })
        return {"objects": objects, "target": objects[target]["name"], "target_index": target}

    def fill_scene(self, world, plan):
        phys, col = self.common["object_physics"], self.common["collision"]["object"]
        mat = self.cfg["reference_mat"]
        top = self.common["table"]["top_z"]
        geom(world, "reference_mat", "box", (mat["half_size_m"], mat["half_size_m"], 0.0015),
             pos=(*mat["center_xy"], top + 0.0015), rgba=(0.10, 0.85, 0.20, 1.0), contype=0, conaffinity=0)
        for obj in plan["objects"]:
            body = free_body(world, obj["name"], obj["pos"], quat_from_yaw(obj["yaw"]))
            gtype = "box" if obj["type"] == "box" else "cylinder"
            geom(body, f"{obj['name']}_geom", gtype, obj["half"], rgba=obj["rgba"], mass=obj["mass"],
                 friction=phys["friction"], contype=col["contype"], conaffinity=col["conaffinity"], condim=phys["condim"])

    def tracked_bodies(self, plan):
        return [o["name"] for o in plan["objects"]]

    def make_evaluator(self, plan):
        return PickupEvaluator(self.cfg["evaluator"], plan, self.common)

    def instruction_for(self, plan):
        target = next(o for o in plan["objects"] if o["name"] == plan["target"])
        return self.cfg["instruction"].format(color=target["color"], shape=target["shape"])
