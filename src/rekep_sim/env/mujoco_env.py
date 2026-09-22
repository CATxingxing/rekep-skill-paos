"""MuJoCo implementation of the ReKep environment interface.

Mirrors the methods the reference simulation line expects (``get_cam_obs``,
``register_keypoints``/``get_keypoint_positions``, ``get_ee_pose``,
``execute_action``, ``get_sdf_voxels``, ``get_collision_points``, ...) but runs
on the Dobot Nova2 + Robotiq 2F-85 MuJoCo scene instead of OmniGibson/Fetch.
The algorithm modules (keypoint proposal, solvers) are the reference ones.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

os.environ.setdefault("MUJOCO_GL", "egl")

import numpy as np  # noqa: E402

import mujoco  # noqa: E402

from .camera import backproject, inject_camera, intrinsics  # noqa: E402

DEFAULT_CAMERA = {
    "name": "vlm",
    "eye": (0.55, -0.80, 0.80),
    "target": (0.05, -0.30, 0.15),
    "fovy": 45.0,
}

# Object model: by default the env auto-enumerates scene geoms into
#   * movable objects  (parent body has a free joint)
#   * regions          (name contains "zone" or starts with "region")
# An explicit mapping ``{geom_name: label}`` can still be passed to override.
_EXCLUDE_NAME_TOKENS = ("floor", "ground", "platform", "table", "wall", "ceiling")

# Robot spec (explicit). If omitted the env auto-detects joint1..7|joint1..6, a
# gripper actuator, and an existing tool site (pinch/tool); if none exists it
# injects one on ``tool_body`` (used for the Franka Panda).
_ROBOT_PANDA = {
    "arm_joints": [f"joint{i}" for i in range(1, 8)],
    "gripper_actuator": "actuator8",
    "gripper_open": 0.0,
    "gripper_close": 255.0,
    "tool_site": "tool",
    "tool_body": "hand",
    "tool_offset": [0.0, 0.0, 0.10],
    "base_frame": "link0",
}


from ..errors import UnreachablePose  # noqa: E402


@dataclass
class Capture:
    rgb: np.ndarray
    depth: np.ndarray
    points: np.ndarray
    seg: np.ndarray


@dataclass
class _IKResult:
    success: bool
    num_descents: int
    position_error: float
    cspace_position: np.ndarray
    orientation_error: float = 0.0


class MujocoReKepEnv:
    def __init__(
        self,
        scene_path: str | Path,
        *,
        height: int = 240,
        width: int = 320,
        bounds_min=(-0.6, -0.9, 0.0),
        bounds_max=(0.7, 0.6, 1.3),
        camera: dict | None = None,
        objects: dict[str, int] | None = None,
        robot: dict | None = None,
        video_size: int = 2000,
        seed: int = 0,
    ):
        self.scene_path = str(Path(scene_path).resolve())
        self.height, self.width = int(height), int(width)
        self.bounds_min = np.asarray(bounds_min, dtype=float)
        self.bounds_max = np.asarray(bounds_max, dtype=float)
        self.camera_cfg = dict(camera or DEFAULT_CAMERA)
        self.objects_override = dict(objects) if objects else None
        self.robot_override = dict(robot) if robot else None
        self.video_size = int(video_size)
        self.rng = np.random.default_rng(seed)
        self.verbose = False

        self._build_model()
        self._build_objects()
        self.renderer = mujoco.Renderer(self.model, height=self.height, width=self.width)
        self.intr = intrinsics(self.camera_cfg["fovy"], self.height, self.width)
        self.video_cache: list[np.ndarray] = []
        self.step_counter = 0
        self.last_og_gripper_action = float(self.robot["gripper_open"])
        self._grasp_label: int | None = None
        self._grasp_offset = np.eye(4)
        self.grasp_distance_threshold = 0.06
        self.reset_joint_pos = self.data.qpos[self.arm_qposadr].copy()
        self._keypoint_registry: dict[int, tuple[int, np.ndarray]] = {}
        self._keypoint2object: dict[int, int] = {}
        self.reset()

    # ---------------------------------------------------------------- model
    @staticmethod
    def _spec_has(collection, name: str) -> bool:
        try:
            return collection(name) is not None
        except Exception:
            return False

    def _build_model(self) -> None:
        spec = mujoco.MjSpec.from_file(self.scene_path)
        inject_camera(
            spec,
            self.camera_cfg["name"],
            self.camera_cfg["eye"],
            self.camera_cfg["target"],
            self.camera_cfg["fovy"],
        )
        override = self.robot_override or {}
        arm_joints = override.get("arm_joints") or [
            f"joint{i}" for i in range(1, 8) if self._spec_has(spec.joint, f"joint{i}")
        ]
        if not arm_joints:
            arm_joints = [f"joint{i}" for i in range(1, 7)]
        gripper_act = override.get("gripper_actuator") or next(
            (n for n in ("actuator8", "fingers_actuator", "gripper") if self._spec_has(spec.actuator, n)),
            "",
        )
        desired_site = override.get("tool_site")
        if desired_site and self._spec_has(spec.site, desired_site):
            tool_site = desired_site
        elif desired_site is None:
            tool_site = next(
                (n for n in ("pinch", "tool", "attachment_site", "end_effector") if self._spec_has(spec.site, n)),
                "",
            )
        else:
            tool_site = ""  # requested but not present -> inject below
        if not tool_site:
            tool_body = override.get("tool_body") or next(
                (n for n in ("hand", "r2f85_base", "Link6") if self._spec_has(spec.body, n)), ""
            )
            if not tool_body:
                raise RuntimeError("scene has no tool site and no tool body")
            site = spec.body(tool_body).add_site()
            site.name = "tool"
            site.pos = list(override.get("tool_offset", [0.0, 0.0, 0.10]))
            site.size = [0.005, 0.005, 0.005]
            tool_site = "tool"
        self.robot = {
            "arm_joints": list(arm_joints),
            "gripper_actuator": gripper_act,
            "gripper_open": float(override.get("gripper_open", 0.0)),
            "gripper_close": float(override.get("gripper_close", 255.0)),
            "tool_site": tool_site,
            "base_frame": override.get("base_frame", "base"),
        }
        self.model = spec.compile()
        self.data = mujoco.MjData(self.model)
        self.cam_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_CAMERA, self.camera_cfg["name"]
        )
        self.pinch_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, tool_site)
        if self.pinch_id < 0:
            raise RuntimeError(f"scene is missing tool site {tool_site!r}")
        self.cube_joint = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_JOINT, "pick_cube_free"
        )
        self.cube_qposadr = int(self.model.jnt_qposadr[self.cube_joint]) if self.cube_joint >= 0 else -1
        jids = [
            mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, n)
            for n in self.robot["arm_joints"]
        ]
        self.arm_joints = [j for j in jids if j >= 0]
        if not self.arm_joints:
            raise RuntimeError("no arm joints resolved")
        self.arm_qposadr = np.array([self.model.jnt_qposadr[j] for j in self.arm_joints])
        self.arm_dofadr = np.array([self.model.jnt_dofadr[j] for j in self.arm_joints])
        self.arm_lo = self.model.jnt_range[self.arm_joints, 0].copy()
        self.arm_hi = self.model.jnt_range[self.arm_joints, 1].copy()
        # Resolve the robot base frame as the root of the first arm joint's chain.
        if not override.get("base_frame"):
            b = int(self.model.jnt_bodyid[self.arm_joints[0]])
            while int(self.model.body_parentid[b]) > 0:
                b = int(self.model.body_parentid[b])
            self.robot["base_frame"] = self.model.body(b).name or self.robot.get("base_frame", "base")
        joint_to_act: dict[int, int] = {}
        for a in range(self.model.nu):
            if self.model.actuator_trntype[a] == mujoco.mjtTrn.mjTRN_JOINT:
                joint_to_act[int(self.model.actuator_trnid[a][0])] = a
        self.arm_actuator_ids = np.array(
            [joint_to_act[j] for j in self.arm_joints if j in joint_to_act], dtype=int
        )
        grip = self.robot["gripper_actuator"]
        self.gripper_actuator_id = (
            mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, grip) if grip else -1
        )
        self.target_rot = np.diag([1.0, -1.0, -1.0])

    def _body_has_freejoint(self, body_id: int) -> bool:
        if body_id <= 0:
            return False
        adr, num = int(self.model.body_jntadr[body_id]), int(self.model.body_jntnum[body_id])
        return any(
            self.model.jnt_type[j] == mujoco.mjtJoint.mjJNT_FREE for j in range(adr, adr + num)
        )

    def _build_objects(self) -> None:
        """Enumerate scene objects/regions (or use an explicit override)."""
        self.objects_info: dict[int, dict] = {}
        robot_bodies = self._robot_body_ids()
        if self.objects_override is not None:
            items = sorted(self.objects_override.items(), key=lambda kv: kv[1])
            for name, label in items:
                gid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, name)
                if gid < 0:
                    raise RuntimeError(f"scene is missing geom {name!r}")
                body_id = int(self.model.geom_bodyid[gid])
                self.objects_info[int(label)] = {
                    "name": name,
                    "display": name.removesuffix("_geom"),
                    "geom_id": int(gid),
                    "body_id": body_id,
                    "movable": self._body_has_freejoint(body_id),
                    "region": name.startswith("region") or "zone" in name,
                }
        else:
            label = 0
            for gid in range(self.model.ngeom):
                name = self.model.geom(gid).name or ""
                body_id = int(self.model.geom_bodyid[gid])
                if not name or body_id in robot_bodies:
                    continue
                if self.model.geom_type[gid] == mujoco.mjtGeom.mjGEOM_PLANE:
                    continue
                if any(tok in name.lower() for tok in _EXCLUDE_NAME_TOKENS):
                    continue
                movable = self._body_has_freejoint(body_id)
                is_region = name.startswith("region") or "zone" in name
                if not (movable or is_region):
                    continue
                label += 1
                self.objects_info[label] = {
                    "name": name,
                    "display": name.removesuffix("_geom"),
                    "geom_id": int(gid),
                    "body_id": body_id,
                    "movable": movable,
                    "region": is_region and not movable,
                }
        # derived maps used by tracking / grasping
        self.geom_label = {v["geom_id"]: k for k, v in self.objects_info.items()}
        self.label_geom = {k: v["geom_id"] for k, v in self.objects_info.items()}
        self.label_body = {k: v["body_id"] for k, v in self.objects_info.items()}
        self.label_display = {k: v["display"] for k, v in self.objects_info.items()}
        self.movable_labels = [k for k, v in self.objects_info.items() if v["movable"]]
        self.region_labels = [k for k, v in self.objects_info.items() if v["region"]]

    # ------------------------------------------------------------- rendering
    def _render(self) -> Capture:
        self.renderer.disable_depth_rendering()
        self.renderer.disable_segmentation_rendering()
        self.renderer.update_scene(self.data, camera=self.camera_cfg["name"])
        rgb = self.renderer.render().copy()
        self.renderer.enable_depth_rendering()
        self.renderer.update_scene(self.data, camera=self.camera_cfg["name"])
        depth = self.renderer.render().copy()
        self.renderer.enable_segmentation_rendering()
        self.renderer.update_scene(self.data, camera=self.camera_cfg["name"])
        geom_seg = self.renderer.render()[..., 0].copy()
        # map geom id -> object label via the configured object geoms
        seg = np.zeros(geom_seg.shape, dtype=np.int32)
        for gid, label in self.geom_label.items():
            seg[geom_seg == gid] = label
        points = backproject(
            depth, self.data.cam_xpos[self.cam_id], self.data.cam_xmat[self.cam_id].reshape(3, 3), self.intr
        )
        return Capture(rgb=rgb, depth=depth, points=points, seg=seg)

    def get_cam_obs(self) -> dict[int, dict[str, np.ndarray]]:
        cap = self._render()
        self.last_cam_obs = {
            0: {"rgb": cap.rgb, "depth": cap.depth, "points": cap.points, "seg": cap.seg}
        }
        return self.last_cam_obs

    # -------------------------------------------------------- keypoint track
    def register_keypoints(self, keypoints) -> None:
        keypoints = np.asarray(keypoints, dtype=float)
        if keypoints.ndim == 1:
            keypoints = keypoints[None, :]
        self._keypoint_registry = {}
        self._keypoint2object = {}
        labels = list(self.label_geom)
        geom_pose = {
            label: (
                self.data.geom_xpos[self.label_geom[label]].copy(),
                self.data.geom_xmat[self.label_geom[label]].reshape(3, 3).copy(),
            )
            for label in labels
        }
        for idx, kp in enumerate(keypoints):
            best_label, best_local, best_dist = None, None, np.inf
            for label in labels:
                pos, rot = geom_pose[label]
                local = rot.T @ (kp - pos)
                dist = float(np.linalg.norm(local))
                if dist < best_dist:
                    best_label, best_local, best_dist = label, local, dist
            assert best_label is not None
            self._keypoint_registry[idx] = (best_label, best_local)
            self._keypoint2object[idx] = best_label

    def get_keypoint_positions(self) -> np.ndarray:
        if not self._keypoint_registry:
            raise RuntimeError("keypoints have not been registered")
        out = []
        for idx in sorted(self._keypoint_registry):
            label, local = self._keypoint_registry[idx]
            gid = self.label_geom[label]
            pos = self.data.geom_xpos[gid]
            rot = self.data.geom_xmat[gid].reshape(3, 3)
            out.append(pos + rot @ local)
        return np.asarray(out)

    def get_object_by_keypoint(self, keypoint_idx: int) -> int:
        return self._keypoint2object[int(keypoint_idx)]

    def geom_id_for_display(self, name: str) -> int:
        for info in self.objects_info.values():
            if info["display"] == name or info["name"] == name:
                return info["geom_id"]
        return -1

    def object_positions(self) -> dict[str, list[float]]:
        return {
            info["display"]: [float(x) for x in self.data.geom_xpos[info["geom_id"]]]
            for info in self.objects_info.values()
        }

    def object_state(self) -> list[dict]:
        out = []
        for label, info in self.objects_info.items():
            gid = info["geom_id"]
            out.append(
                {
                    "label": label,
                    "name": info["display"],
                    "geom": info["name"],
                    "movable": info["movable"],
                    "region": info["region"],
                    "position_m": [float(x) for x in self.data.geom_xpos[gid]],
                    "quaternion_xyzw": [
                        float(x)
                        for x in _mat_to_quat_xyzw(self.data.geom_xmat[gid].reshape(3, 3))
                    ],
                    "geom_size": [float(x) for x in self.model.geom_size[gid]],
                }
            )
        return out

    def is_grasping(self, candidate_obj=None) -> bool:
        """Assisted grasp (if attached) or gripper-pad contact with the object."""
        if self._grasp_label is not None and (
            candidate_obj is None or int(candidate_obj) == self._grasp_label
        ):
            return True
        target_geoms = None
        if candidate_obj is not None and int(candidate_obj) in self.label_geom:
            target_geoms = {self.label_geom[int(candidate_obj)]}
        for ci in range(self.data.ncon):
            c = self.data.contact[ci]
            g1, g2 = int(c.geom1), int(c.geom2)
            n1 = self.model.geom(g1).name or ""
            n2 = self.model.geom(g2).name or ""
            if not ("pad" in n1 or "pad" in n2):
                continue
            if target_geoms is None:
                return True
            if g1 in target_geoms or g2 in target_geoms:
                return True
        return False

    # -------------------------------------------------------------- ee / arm
    def get_ee_pose(self) -> np.ndarray:
        return np.concatenate(
            [self.data.site_xpos[self.pinch_id].copy(), _mat_to_quat_xyzw(self.data.site_xmat[self.pinch_id].reshape(3, 3))]
        )

    def get_ee_pos(self) -> np.ndarray:
        return self.data.site_xpos[self.pinch_id].copy()

    def get_ee_quat(self) -> np.ndarray:
        return self.get_ee_pose()[3:]

    def tool_down_quat(self) -> np.ndarray:
        """Quaternion (xyzw) of the fixed top-down tool orientation."""
        return _mat_to_quat_xyzw(self.target_rot)

    def get_arm_joint_postions(self) -> np.ndarray:
        return self.data.qpos[self.arm_qposadr].copy()

    def reset_joint_positions(self) -> np.ndarray:
        return self.reset_joint_pos.copy()

    # ------------------------------------------------------------------- IK
    def ik(self, target_pos, target_rot=None, seed=None, iterations: int = 2000,
           pos_tol: float = 2e-5, rot_tol: float = 2e-4, step: float = 0.25) -> np.ndarray:
        target_pos = np.asarray(target_pos, dtype=float)
        target_rot = self.target_rot if target_rot is None else np.asarray(target_rot, dtype=float)
        q = (self.reset_joint_pos.copy() if seed is None else np.asarray(seed, dtype=float).copy())
        q = np.clip(q, self.arm_lo + 1e-5, self.arm_hi - 1e-5)

        def _err(qq):
            self.data.qpos[self.arm_qposadr] = qq
            mujoco.mj_forward(self.model, self.data)
            pe = target_pos - self.data.site_xpos[self.pinch_id]
            cur = self.data.site_xmat[self.pinch_id].reshape(3, 3)
            re = 0.5 * sum(np.cross(cur[:, i], target_rot[:, i]) for i in range(3))
            return pe, re

        pe, re = _err(q)
        err = np.concatenate([pe, re])
        for _ in range(iterations):
            if np.linalg.norm(pe) < pos_tol and np.linalg.norm(re) < rot_tol:
                return q
            jacp = np.zeros((3, self.model.nv))
            jacr = np.zeros((3, self.model.nv))
            mujoco.mj_jacSite(self.model, self.data, jacp, jacr, self.pinch_id)
            dof = self.arm_dofadr
            jac = np.vstack([jacp[:, dof], jacr[:, dof]])
            dq = jac.T @ np.linalg.solve(jac @ jac.T + 1e-4 * np.eye(6), err)
            # adaptive (Levenberg-Marquardt style) step: accept improvements,
            # otherwise shrink until progress or give up.
            accepted = False
            for _attempt in range(10):
                q_new = np.clip(q + step * dq, self.arm_lo + 1e-5, self.arm_hi - 1e-5)
                pe2, re2 = _err(q_new)
                err2 = np.concatenate([pe2, re2])
                if np.linalg.norm(err2) < np.linalg.norm(err):
                    q, pe, re, err = q_new, pe2, re2, err2
                    step = min(step * 1.4, 0.6)
                    accepted = True
                    break
                step *= 0.5
                if step < 1e-4:
                    break
            if not accepted and step < 1e-4:
                break
        raise RuntimeError(f"IK failed for target {target_pos.round(3).tolist()}")

    def solve_ik_result(
        self,
        target_pose_homo,
        position_tolerance: float = 0.01,
        orientation_tolerance: float = 0.05,
        position_weight: float = 1.0,
        orientation_weight: float = 0.05,
        max_iterations: int = 150,
        initial_joint_pos=None,
    ):
        """Adapter matching the reference IK result interface used by the solvers."""
        target_pose_homo = np.asarray(target_pose_homo, dtype=float).reshape(4, 4)
        target_pos = target_pose_homo[:3, 3]
        target_rot = target_pose_homo[:3, :3]
        iters = max(1, int(max_iterations))
        q = self.get_arm_joint_postions() if initial_joint_pos is None else np.asarray(initial_joint_pos, dtype=float)
        try:
            q = self.ik(target_pos, target_rot, seed=q, iterations=iters)
            success = True
            pos_err = float(np.linalg.norm(target_pos - self.data.site_xpos[self.pinch_id]))
            descents = 1
        except RuntimeError:
            success = False
            pos_err = float("inf")
            q = self.get_arm_joint_postions()
            descents = iters
        return _IKResult(success, descents, pos_err, q)

    def _robust_ik(self, target_pos, target_rot, iterations: int = 4000,
                   pos_tol: float = 2e-5, rot_tol: float = 2e-4) -> np.ndarray:
        seeds = [self.get_arm_joint_postions(), self.reset_joint_positions()]
        for _ in range(14):
            seeds.append(self.rng.uniform(self.arm_lo, self.arm_hi))
        last_err = None
        for seed in seeds:
            try:
                return self.ik(target_pos, target_rot, seed=seed, iterations=iterations,
                               pos_tol=pos_tol, rot_tol=rot_tol)
            except RuntimeError as exc:
                last_err = exc
        raise RuntimeError(str(last_err))

    def _move_to(self, q_target, gripper: float, steps: int = 60) -> None:
        for _ in range(steps):
            if len(self.arm_actuator_ids):
                self.data.ctrl[self.arm_actuator_ids] = q_target
            if self.gripper_actuator_id >= 0:
                self.data.ctrl[self.gripper_actuator_id] = gripper
            mujoco.mj_step(self.model, self.data)
            if self._grasp_label is not None:
                self._apply_grasp()
            self._record_frame()
        self.step_counter += 1

    def _pinch_pose(self) -> np.ndarray:
        pose = np.eye(4)
        pose[:3, :3] = self.data.site_xmat[self.pinch_id].reshape(3, 3)
        pose[:3, 3] = self.data.site_xpos[self.pinch_id]
        return pose

    def _free_body_pose(self, body_id: int) -> np.ndarray:
        pose = np.eye(4)
        pose[:3, :3] = _quat_wxyz_to_mat(self.data.xquat[body_id])
        pose[:3, 3] = self.data.xpos[body_id]
        return pose

    def _try_attach(self) -> None:
        if self._grasp_label is not None:
            return
        pinch = self._pinch_pose()
        for label, gid in self.label_geom.items():
            body_id = self.label_body[label]
            if body_id == 0:  # world/static object (e.g. place zone)
                continue
            obj = self._free_body_pose(body_id)
            if np.linalg.norm(pinch[:3, 3] - obj[:3, 3]) < self.grasp_distance_threshold:
                self._grasp_label = label
                self._grasp_offset = np.linalg.inv(pinch) @ obj
                return

    def _apply_grasp(self) -> None:
        if self._grasp_label is None:
            return
        body_id = self.label_body[self._grasp_label]
        target = self._pinch_pose() @ self._grasp_offset
        jnt = int(self.model.body_jntadr[body_id])
        if jnt < 0:
            return
        qadr = int(self.model.jnt_qposadr[jnt])
        self.data.qpos[qadr:qadr + 3] = target[:3, 3]
        self.data.qpos[qadr + 3:qadr + 7] = _mat_to_quat_wxyz(target[:3, :3])
        mujoco.mj_forward(self.model, self.data)

    def execute_action(self, action, precise: bool = True):
        """action = [x,y,z,qx,qy,qz,qw, gripper] with gripper 1=close,-1=open,0=null."""
        action = np.asarray(action, dtype=float)
        if action.shape != (8,):
            raise ValueError(f"action must be length 8, got {action.shape}")
        target_pos = action[:3]
        target_quat_xyzw = action[3:7]
        gripper = action[7]
        target_rot = _quat_xyzw_to_mat(target_quat_xyzw)
        # Faithful: exact orientation first; if unreachable, retry within a small
        # controller tolerance (~6 deg; the reference OSC uses 3-5 deg rotation
        # thresholds). No silent change of the commanded orientation.
        try:
            q = self._robust_ik(target_pos, target_rot)
        except RuntimeError:
            try:
                q = self._robust_ik(target_pos, target_rot, pos_tol=0.01, rot_tol=0.1)
            except RuntimeError as exc:
                raise UnreachablePose(str(exc)) from exc
        g = self.last_og_gripper_action
        if gripper == self.get_gripper_close_action():
            g = float(self.robot["gripper_close"])
        elif gripper == self.get_gripper_open_action():
            g = float(self.robot["gripper_open"])
        self._move_to(q, g, steps=80 if precise else 40)
        self.last_og_gripper_action = g
        pos_err = float(np.linalg.norm(target_pos - self.data.site_xpos[self.pinch_id]))
        return pos_err, 0.0

    def open_gripper(self) -> None:
        if self.last_og_gripper_action == float(self.robot["gripper_open"]) and self._grasp_label is None:
            return
        self._move_to(self.get_arm_joint_postions(), float(self.robot["gripper_open"]), steps=30)
        self.last_og_gripper_action = float(self.robot["gripper_open"])
        self._grasp_label = None

    def close_gripper(self) -> None:
        if self.last_og_gripper_action != float(self.robot["gripper_close"]):
            self._move_to(self.get_arm_joint_postions(), float(self.robot["gripper_close"]), steps=30)
            self.last_og_gripper_action = float(self.robot["gripper_close"])
        self._try_attach()

    def get_gripper_open_action(self) -> float:
        return -1.0

    def get_gripper_close_action(self) -> float:
        return 1.0

    def get_gripper_null_action(self) -> float:
        return 0.0

    def sleep(self, seconds: float) -> None:
        steps = max(1, int(seconds * 50))
        self._move_to(self.get_arm_joint_postions(), self.last_og_gripper_action, steps=steps)

    # --------------------------------------------------------- sdf / collision
    def get_sdf_voxels(self, resolution: float, exclude_robot: bool = True, exclude_obj_in_hand: bool = True):
        import open3d as o3d
        import trimesh

        meshes = []
        robot_bodies = self._robot_body_ids()
        for geom_id in range(self.model.ngeom):
            if self.model.geom_type[geom_id] != mujoco.mjtGeom.mjGEOM_MESH:
                continue
            if exclude_robot and self.model.geom_bodyid[geom_id] in robot_bodies:
                continue
            mesh_id = self.model.geom_dataid[geom_id]
            if mesh_id < 0:
                continue
            vadr, vnum = self.model.mesh_vertadr[mesh_id], self.model.mesh_vertnum[mesh_id]
            fadr, fnum = self.model.mesh_faceadr[mesh_id], self.model.mesh_facenum[mesh_id]
            verts = self.model.mesh_vert[vadr:vadr + vnum].reshape(-1, 3)
            faces = self.model.mesh_face[fadr:fadr + fnum].reshape(-1, 3)
            if self.model.mesh_scale[mesh_id] is not None:
                verts = verts * self.model.mesh_scale[mesh_id]
            tm = trimesh.Trimesh(vertices=verts, faces=faces, process=False)
            pose = np.eye(4)
            pose[:3, :3] = self.data.geom_xmat[geom_id].reshape(3, 3)
            pose[:3, 3] = self.data.geom_xpos[geom_id]
            tm.apply_transform(pose)
            meshes.append(tm)
        if not meshes:
            return np.zeros((1, 1, 1))
        scene_mesh = trimesh.util.concatenate(meshes)
        scene = o3d.t.geometry.RaycastingScene()
        scene.add_triangles(
            o3d.core.Tensor(scene_mesh.vertices, dtype=o3d.core.Dtype.Float32),
            o3d.core.Tensor(scene_mesh.faces.astype(np.uint32), dtype=o3d.core.Dtype.UInt32),
        )
        shape = np.ceil((self.bounds_max - self.bounds_min) / resolution).astype(int)
        steps = (self.bounds_max - self.bounds_min) / shape
        grid = np.mgrid[
            self.bounds_min[0]:self.bounds_max[0]:steps[0],
            self.bounds_min[1]:self.bounds_max[1]:steps[1],
            self.bounds_min[2]:self.bounds_max[2]:steps[2],
        ].reshape(3, -1).T
        sdf = scene.compute_signed_distance(grid.astype(np.float32)).cpu().numpy()
        sdf = -sdf  # open3d sign convention
        return sdf.reshape(shape)

    def get_collision_points(self, noise: bool = True) -> np.ndarray:
        pts = []
        body_ids = self._robot_body_ids()
        grasped_bodies = {
            self.label_body[label]
            for label in self.label_geom
            if self.is_grasping(label)
        }
        for geom_id in range(self.model.ngeom):
            body_id = int(self.model.geom_bodyid[geom_id])
            body_name = (self.model.body(body_id).name or "").lower()
            is_gripper = body_id in body_ids and any(
                tok in body_name for tok in ("finger", "hand", "pad", "gripper")
            )
            if not is_gripper and body_id not in grasped_bodies:
                continue
            sampled = self._sample_geom_points(geom_id, 200)
            if sampled is not None and len(sampled):
                pts.append(sampled)
        if not pts:
            return np.zeros((0, 3))
        return np.concatenate(pts, axis=0)

    def _sample_geom_points(self, geom_id: int, n: int) -> np.ndarray | None:
        gtype = self.model.geom_type[geom_id]
        size = self.model.geom_size[geom_id].copy()
        if gtype == mujoco.mjtGeom.mjGEOM_MESH:
            mesh_id = self.model.geom_dataid[geom_id]
            if mesh_id < 0:
                return None
            vadr, vnum = self.model.mesh_vertadr[mesh_id], self.model.mesh_vertnum[mesh_id]
            verts = self.model.mesh_vert[vadr:vadr + vnum].reshape(-1, 3)
            if self.model.mesh_scale[mesh_id] is not None:
                verts = verts * self.model.mesh_scale[mesh_id]
            if len(verts) > n:
                verts = verts[self.rng.choice(len(verts), size=n, replace=False)]
            local = verts
        elif gtype == mujoco.mjtGeom.mjGEOM_BOX:
            local = (self.rng.random((n, 3)) * 2.0 - 1.0) * size
        elif gtype == mujoco.mjtGeom.mjGEOM_SPHERE:
            v = self.rng.normal(size=(n, 3))
            v /= np.linalg.norm(v, axis=1, keepdims=True)
            local = v * size[0]
        else:
            return None
        rot = self.data.geom_xmat[geom_id].reshape(3, 3)
        return local @ rot.T + self.data.geom_xpos[geom_id]

    def _robot_body_ids(self) -> set[int]:
        """All bodies in the kinematic subtree of the robot base frame."""
        base = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, self.robot["base_frame"])
        if base < 0:
            base = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "base_link")
        if base < 0:
            return set()
        ids: set[int] = set()
        for body_id in range(self.model.nbody):
            ancestor = body_id
            while ancestor > 0:
                if ancestor == base:
                    ids.add(body_id)
                    break
                ancestor = int(self.model.body_parentid[ancestor])
        ids.add(base)
        return ids

    # ----------------------------------------------------------------- reset
    def reset(self) -> None:
        mujoco.mj_resetData(self.model, self.data)
        if self.cube_qposadr >= 0:  # legacy demo scene keeps its cube at the XML pose
            self.data.qpos[self.cube_qposadr:self.cube_qposadr + 3] = [0.25, -0.42, 0.05]
            self.data.qpos[self.cube_qposadr + 3:self.cube_qposadr + 7] = [1, 0, 0, 0]
        mujoco.mj_forward(self.model, self.data)
        self.video_cache = []
        self.last_og_gripper_action = float(self.robot["gripper_open"])
        self._grasp_label = None

    def _record_frame(self) -> None:
        self.renderer.disable_depth_rendering()
        self.renderer.disable_segmentation_rendering()
        self.renderer.update_scene(self.data, camera=self.camera_cfg["name"])
        rgb = self.renderer.render().copy()
        if len(self.video_cache) >= self.video_size:
            self.video_cache.pop(0)
        self.video_cache.append(rgb)

    def save_video(self, save_path: str | None = None) -> str:
        import imageio.v2 as imageio

        save_path = save_path or str(Path(os.environ.get("TMPDIR", "/tmp")) / "rekep_mujoco.mp4")
        writer = imageio.get_writer(save_path, fps=20, macro_block_size=1)
        for frame in self.video_cache:
            writer.append_data(frame)
        writer.close()
        self.last_video_path = save_path
        return save_path

    def close(self) -> None:
        try:
            self.renderer.close()
        except Exception:
            pass


def _mat_to_quat_xyzw(mat: np.ndarray) -> np.ndarray:
    quat_wxyz = np.zeros(4)
    mujoco.mju_mat2Quat(quat_wxyz, np.asarray(mat, dtype=np.float64).reshape(9))
    return np.array([quat_wxyz[1], quat_wxyz[2], quat_wxyz[3], quat_wxyz[0]])


def _quat_xyzw_to_mat(quat_xyzw: np.ndarray) -> np.ndarray:
    w, x, y, z = quat_xyzw[3], quat_xyzw[0], quat_xyzw[1], quat_xyzw[2]
    mat = np.zeros(9)
    mujoco.mju_quat2Mat(mat, np.array([w, x, y, z]))
    return mat.reshape(3, 3)


def _mat_to_quat_wxyz(mat: np.ndarray) -> np.ndarray:
    quat = np.zeros(4)
    mujoco.mju_mat2Quat(quat, np.asarray(mat, dtype=np.float64).reshape(9))
    return quat


def _quat_wxyz_to_mat(quat_wxyz: np.ndarray) -> np.ndarray:
    mat = np.zeros(9)
    mujoco.mju_quat2Mat(mat, np.asarray(quat_wxyz, dtype=np.float64))
    return mat.reshape(3, 3)


def _world_to_body(point: np.ndarray, body_pos: np.ndarray, body_quat_wxyz: np.ndarray) -> np.ndarray:
    quat = np.array([body_quat_wxyz[1], body_quat_wxyz[2], body_quat_wxyz[3], body_quat_wxyz[0]])
    rot = _quat_xyzw_to_mat(quat)
    return rot.T @ (point - body_pos)


def _body_to_world(local: np.ndarray, body_pos: np.ndarray, body_quat_wxyz: np.ndarray) -> np.ndarray:
    quat = np.array([body_quat_wxyz[1], body_quat_wxyz[2], body_quat_wxyz[3], body_quat_wxyz[0]])
    rot = _quat_xyzw_to_mat(quat)
    return rot @ local + body_pos
