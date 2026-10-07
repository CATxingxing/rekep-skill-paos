"""MuJoCo task environment shared by all tasks.

Interfaces
----------
Policy-facing (no ground truth): ``reset``, ``get_instruction``, ``observe``,
``step``.  Evaluator-facing (ground truth): ``evaluate``, ``is_success``,
``get_metrics``, ``snapshot``, ``scene_info``.
"""
from __future__ import annotations

import json
from collections import deque
from pathlib import Path
from typing import Any

import mujoco
import numpy as np

from .config import GENERATED_DIR, load_config
from .evaluator import BaseEvaluator
from .scene import build_mjcf
from .state import BodyState, Contact, SceneState

ARM_JOINTS = [f"joint{i}" for i in range(1, 7)]
GRIPPER_ACTUATOR = "fingers_actuator"
GRIPPER_STATE_SCALE = 318.75   # profile mujoco-simulator.yaml: gripper = right_driver_joint * scale


class SceneUnstable(RuntimeError):
    """The freshly built scene did not settle (objects moving, NaNs, ...)."""


class SceneRejected(RuntimeError):
    """The sampled layout is legal physics-wise but unusable (objects occluded or merged in the camera view)."""


class TaskEnv:
    TASK = ""

    def __init__(self, config: dict | None = None):
        self.config = config or load_config()
        self.common = self.config["common"]
        self.cfg = self.config[self.TASK]
        self.model: mujoco.MjModel | None = None
        self.data: mujoco.MjData | None = None
        self.plan: dict[str, Any] = {}
        self.scene_path: Path | None = None
        self.evaluator: BaseEvaluator | None = None
        self._renderer = None

    # ------------------------------------------------------------ task hooks
    def plan_scene(self, rng: np.random.Generator, overrides: dict | None) -> dict[str, Any]:
        raise NotImplementedError

    def fill_scene(self, world, plan: dict) -> None:
        raise NotImplementedError

    def make_evaluator(self, plan: dict) -> BaseEvaluator:
        raise NotImplementedError

    def tracked_bodies(self, plan: dict) -> list[str]:
        raise NotImplementedError

    def instruction_for(self, plan: dict) -> str:
        raise NotImplementedError

    # ------------------------------------------------------------ lifecycle
    def reset(self, seed: int = 0, overrides: dict | None = None, out_dir: Path | None = None,
              check_settle: bool = True) -> dict:
        """Build the seeded scene, load it, settle, and arm the evaluator.

        Same ``seed`` (and ``overrides``) gives byte-identical XML and therefore
        the same initial state.  Layouts in which the camera cannot see every object
        as a separate blob are resampled from the same random stream (deterministic).
        Returns the policy-facing observation.  ``check_settle=False`` only skips the
        generation-time rest check (used by replay_eval, whose states come from the log).
        """
        rng = np.random.default_rng(seed)
        directory = Path(out_dir) if out_dir else GENERATED_DIR
        directory.mkdir(parents=True, exist_ok=True)
        last: Exception | None = None
        for attempt in range(int(self.common["max_scene_attempts"])):
            plan = self.plan_scene(rng, overrides)
            plan["task"], plan["seed"], plan["attempt"] = self.TASK, int(seed), attempt
            xml = build_mjcf(lambda world: self.fill_scene(world, plan), self.common)
            self.scene_path = directory / f"{self.TASK}_seed{seed}.xml"
            self.scene_path.write_text(xml, encoding="utf-8")
            self.plan = plan
            self._close_renderer()
            self.model = mujoco.MjModel.from_xml_path(str(self.scene_path))
            self.data = mujoco.MjData(self.model)
            self._index_model()
            self._apply_home()
            self._settle(check_settle)
            try:
                self._check_visibility()
            except SceneRejected as exc:
                last = exc
                continue
            break
        else:
            raise SceneRejected(f"no visible layout for seed {seed} after {attempt + 1} attempts: {last}")
        self.evaluator = self.make_evaluator(plan)
        self.evaluator.reset(self.snapshot())
        self.t_start = float(self.data.time)
        return self.observe()

    def _check_visibility(self) -> None:
        """Every tracked object shows >= min_pixels pixels and forms its own image blob."""
        vis = self.common["visibility"]
        r = self.common["render"]
        if self._renderer is None:
            self._renderer = mujoco.Renderer(self.model, r["height"], r["width"])
        self._renderer.enable_segmentation_rendering()
        self._renderer.update_scene(self.data, camera=self.common["camera"])
        seg = self._renderer.render().copy()
        self._renderer.disable_segmentation_rendering()
        geom_ids, types = seg[..., 0], seg[..., 1]
        owner = np.full(self.model.ngeom + 1, -1)
        for index, name in enumerate(self.tracked):
            owner[np.flatnonzero(self.model.geom_bodyid == self.body_id[name])] = index
        label = np.where(types == int(mujoco.mjtObj.mjOBJ_GEOM), owner[np.clip(geom_ids, 0, self.model.ngeom)], -1)
        for index, name in enumerate(self.tracked):
            count = int((label == index).sum())
            if count < vis["min_pixels"]:
                raise SceneRejected(f"{name} shows only {count} pixels")
        # blobs (8-connected, grown by `gap_px`) must not join two different objects
        grown = np.zeros_like(label)
        grown[:] = -1
        gap = int(vis["gap_px"])
        for index in range(len(self.tracked)):
            mask = label == index
            dil = mask.copy()
            for dy in range(-gap, gap + 1):
                for dx in range(-gap, gap + 1):
                    dil |= np.roll(np.roll(mask, dy, 0), dx, 1)
            clash = (dil & (grown >= 0) & (grown != index))
            if clash.any():
                raise SceneRejected(f"{self.tracked[index]} touches another object in the image")
            grown[dil & (grown < 0)] = index

    def _index_model(self) -> None:
        m = self.model
        self.timestep = float(m.opt.timestep)
        self.substeps = max(1, round(1.0 / (self.common["control_hz"] * self.timestep)))
        self.arm_joint_ids = [m.joint(n).id for n in ARM_JOINTS]
        self.arm_qpos_adr = [int(m.jnt_qposadr[j]) for j in self.arm_joint_ids]
        self.arm_dof_adr = [int(m.jnt_dofadr[j]) for j in self.arm_joint_ids]
        self.arm_act = [m.actuator(n).id for n in ARM_JOINTS]
        self.grip_act = m.actuator(GRIPPER_ACTUATOR).id
        self.pinch_site = m.site("pinch").id
        self.right_driver_qadr = int(m.jnt_qposadr[m.joint("right_driver_joint").id])
        self.geom_body = [mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, int(b)) or "world" for b in m.geom_bodyid]
        self.geom_name = [mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, i) or f"geom{i}" for i in range(m.ngeom)]
        self.tracked = self.tracked_bodies(self.plan)
        self.tracked_set = set(self.tracked)
        self.body_id = {n: m.body(n).id for n in self.tracked}

    def _apply_home(self) -> None:
        for adr, value in zip(self.arm_qpos_adr, self.common["home_joint_positions"]):
            self.data.qpos[adr] = value
        for act, value in zip(self.arm_act, self.common["home_joint_positions"]):
            self.data.ctrl[act] = value
        self.data.ctrl[self.grip_act] = self.common["gripper_open_ctrl"]
        mujoco.mj_forward(self.model, self.data)

    def _settle(self, check: bool = True) -> None:
        """Run physics with the arm held; reject scenes that do not come to rest."""
        start = self.snapshot()
        for _ in range(round(self.common["settle_s"] / self.timestep)):
            mujoco.mj_step(self.model, self.data)
            if not np.isfinite(self.data.qpos).all():
                raise SceneUnstable("NaN in qpos while settling")
        end = self.snapshot()
        for name in self.tracked if check else ():
            drift = float(np.linalg.norm(end.bodies[name].pos - start.bodies[name].pos))
            speed = float(np.linalg.norm(end.bodies[name].linvel))
            if drift > self.common["settle_max_drift_m"] or speed > self.common["settle_max_speed"]:
                raise SceneUnstable(f"{name} did not settle: drift {drift:.4f} m, speed {speed:.4f} m/s")

    # ------------------------------------------------------------ simulation
    @property
    def sim_time(self) -> float:
        return float(self.data.time - self.t_start)

    @property
    def timed_out(self) -> bool:
        limit = self.cfg.get("time_limit_s")
        return limit is not None and self.sim_time > float(limit)

    def step(self, arm_ctrl=None, gripper_ctrl: float | None = None, ticks: int = 1) -> None:
        """Advance ``ticks`` control periods with the given (held) targets."""
        if arm_ctrl is not None:
            self.data.ctrl[self.arm_act] = np.asarray(arm_ctrl, float)
        if gripper_ctrl is not None:
            self.data.ctrl[self.grip_act] = float(gripper_ctrl)
        for _ in range(ticks):
            for _ in range(self.substeps):
                mujoco.mj_step(self.model, self.data)
            if not np.isfinite(self.data.qpos).all():
                raise SceneUnstable("NaN in qpos")
            self.evaluator.observe(self.snapshot())

    def arm_qpos(self) -> np.ndarray:
        return np.array([self.data.qpos[a] for a in self.arm_qpos_adr])

    # ------------------------------------------------------------ ground truth
    def _snapshot(self, data: mujoco.MjData) -> SceneState:
        m = self.model
        bodies = {}
        res = np.zeros(6)
        for name, bid in self.body_id.items():
            mujoco.mj_objectVelocity(m, data, mujoco.mjtObj.mjOBJ_BODY, bid, res, 0)
            bodies[name] = BodyState(data.xpos[bid].copy(), data.xquat[bid].copy(), res[3:].copy(), res[:3].copy())
        contacts, force = [], np.zeros(6)
        for i in range(data.ncon):
            c = data.contact[i]
            b1, b2 = self.geom_body[c.geom1], self.geom_body[c.geom2]
            if b1 not in self.tracked_set and b2 not in self.tracked_set:
                continue
            mujoco.mj_contactForce(m, data, i, force)
            contacts.append(Contact(self.geom_name[c.geom1], self.geom_name[c.geom2], b1, b2, float(force[0]), float(c.dist)))
        return SceneState(float(data.time), bodies, contacts, data.site_xpos[self.pinch_site].copy(),
                          float(data.ctrl[self.grip_act]))

    def snapshot(self) -> SceneState:
        return self._snapshot(self.data)

    def probe(self, duration: float, dt: float) -> list[SceneState]:
        """Roll a *copy* of the simulation forward with controls held; live state is untouched."""
        if hasattr(mujoco, "mj_copyData"):
            copy = mujoco.MjData(self.model)
            mujoco.mj_copyData(copy, self.model, self.data)
        else:  # older bindings (e.g. 3.3.7, the version bundled in mujoco_sim) only offer __copy__
            copy = self.data.__copy__()
        states = [self._snapshot(copy)]
        sub = max(1, round(dt / self.timestep))
        for _ in range(round(duration / dt)):
            for _ in range(sub):
                mujoco.mj_step(self.model, copy)
            states.append(self._snapshot(copy))
        return states

    def set_body_pose(self, name: str, pos, quat=None, settle_s: float = 0.0) -> None:
        """Teleport a free body (test / scenario tool).  Zeroes its velocity."""
        jid = self.model.joint(f"{name}_free").id
        qadr, dadr = int(self.model.jnt_qposadr[jid]), int(self.model.jnt_dofadr[jid])
        self.data.qpos[qadr:qadr + 3] = pos
        if quat is not None:
            self.data.qpos[qadr + 3:qadr + 7] = quat
        self.data.qvel[dadr:dadr + 6] = 0.0
        mujoco.mj_forward(self.model, self.data)
        for _ in range(round(settle_s / self.timestep)):
            mujoco.mj_step(self.model, self.data)

    # ------------------------------------------------------------ policy-facing API
    def get_instruction(self) -> str:
        return self.instruction_for(self.plan)

    def observe(self) -> dict[str, Any]:
        """What a policy may see: RGB, depth (m), joint state, sim time.  No ground truth."""
        rgb, depth = self.render_rgbd()
        names = ARM_JOINTS + ["gripper"]
        positions = [float(self.data.qpos[a]) for a in self.arm_qpos_adr]
        positions.append(float(self.data.qpos[self.right_driver_qadr]) * GRIPPER_STATE_SCALE)
        return {"rgb": rgb, "depth": depth, "joint_names": names, "joint_positions": positions,
                "sim_time": self.sim_time, "instruction": self.get_instruction()}

    def render_rgbd(self, camera: str | None = None) -> tuple[np.ndarray, np.ndarray]:
        r = self.common["render"]
        if self._renderer is None:
            self._renderer = mujoco.Renderer(self.model, r["height"], r["width"])
        camera = camera or self.common["camera"]
        self._renderer.update_scene(self.data, camera=camera)
        rgb = self._renderer.render().copy()
        self._renderer.enable_depth_rendering()
        self._renderer.update_scene(self.data, camera=camera)
        depth = self._renderer.render().copy()
        self._renderer.disable_depth_rendering()
        return rgb, depth

    def _close_renderer(self) -> None:
        if self._renderer is not None:
            try:
                self._renderer.close()
            except Exception:  # EGL teardown can complain; harmless
                pass
            self._renderer = None

    def close(self) -> None:
        self._close_renderer()

    # ------------------------------------------------------------ evaluator-facing API
    def evaluate(self, final: bool = True) -> dict[str, Any]:
        """success / partial / score / metrics / failure_reason from ground truth.

        ``final=True`` additionally runs the evaluator's stability probe on a
        copy of the simulation (arm held still).  The live simulation is not
        advanced by evaluation.
        """
        probe = None
        if final and self.evaluator.needs_probe:
            probe = self.probe(self.evaluator.cfg["probe_s"], self.evaluator.cfg["probe_dt_s"])
        return self.evaluator.evaluate(self.snapshot(), probe, timed_out=self.timed_out)

    def is_success(self) -> bool:
        return bool(self.evaluate(final=True)["success"])

    def get_metrics(self) -> dict[str, Any]:
        return self.evaluate(final=True)["metrics"]

    def scene_info(self) -> dict[str, Any]:
        """Ground-truth description of the scene (for logs only; keep away from the policy)."""
        return json.loads(json.dumps(self.plan, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o)))
