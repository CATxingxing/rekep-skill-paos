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

# objects that get a unique segmentation label and can carry keypoints
# name -> label, where name is a MuJoCo *geom* name
DEFAULT_OBJECTS = {"pick_cube_geom": 1, "place_zone": 2}


@dataclass
class Capture:
    rgb: np.ndarray
    depth: np.ndarray
    points: np.ndarray
    seg: np.ndarray


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
        video_size: int = 2000,
        seed: int = 0,
    ):
        self.scene_path = str(Path(scene_path).resolve())
        self.height, self.width = int(height), int(width)
        self.bounds_min = np.asarray(bounds_min, dtype=float)
        self.bounds_max = np.asarray(bounds_max, dtype=float)
        self.camera_cfg = dict(camera or DEFAULT_CAMERA)
        self.objects = dict(objects or DEFAULT_OBJECTS)
        self.video_size = int(video_size)
        self.rng = np.random.default_rng(seed)
        self.verbose = False

        self._build_model()
        self._build_objects()
        self.renderer = mujoco.Renderer(self.model, height=self.height, width=self.width)
        self.intr = intrinsics(self.camera_cfg["fovy"], self.height, self.width)
        self.video_cache: list[np.ndarray] = []
        self.step_counter = 0
        self.last_og_gripper_action = 1.0
        self.reset_joint_pos = self.data.qpos[self.arm_qposadr].copy()
        self._keypoint_registry: dict[int, tuple[int, np.ndarray]] = {}
        self._keypoint2object: dict[int, int] = {}
        self.reset()

    # ---------------------------------------------------------------- model
    def _build_model(self) -> None:
        spec = mujoco.MjSpec.from_file(self.scene_path)
        inject_camera(
            spec,
            self.camera_cfg["name"],
            self.camera_cfg["eye"],
            self.camera_cfg["target"],
            self.camera_cfg["fovy"],
        )
        self.model = spec.compile()
        self.data = mujoco.MjData(self.model)
        self.cam_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_CAMERA, self.camera_cfg["name"]
        )
        self.pinch_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, "pinch")
        self.cube_joint = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_JOINT, "pick_cube_free"
        )
        if self.pinch_id < 0 or self.cube_joint < 0:
            raise RuntimeError("scene is missing pinch site or pick_cube_free joint")
        self.cube_qposadr = int(self.model.jnt_qposadr[self.cube_joint])
        self.arm_joints = [
            mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, f"joint{i}")
            for i in range(1, 7)
        ]
        self.arm_qposadr = np.array([self.model.jnt_qposadr[j] for j in self.arm_joints])
        self.arm_dofadr = np.array([self.model.jnt_dofadr[j] for j in self.arm_joints])
        self.arm_lo = self.model.jnt_range[self.arm_joints, 0].copy()
        self.arm_hi = self.model.jnt_range[self.arm_joints, 1].copy()
        self.target_rot = np.diag([1.0, -1.0, -1.0])

    def _build_objects(self) -> None:
        self.geom_label: dict[int, int] = {}
        self.label_geom: dict[int, int] = {}
        self.label_body: dict[int, int] = {}
        for name, label in self.objects.items():
            gid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, name)
            if gid < 0:
                raise RuntimeError(f"scene is missing geom {name!r}")
            self.geom_label[int(gid)] = int(label)
            self.label_geom[int(label)] = int(gid)
            self.label_body[int(label)] = int(self.model.geom_bodyid[gid])

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
        labels = list(self.label_body)
        body_pose = {
            label: (
                self.data.xpos[self.label_body[label]].copy(),
                self.data.xquat[self.label_body[label]].copy(),
            )
            for label in labels
        }
        for idx, kp in enumerate(keypoints):
            best_label, best_local, best_dist = None, None, np.inf
            for label in labels:
                pos, quat = body_pose[label]
                local = _world_to_body(kp, pos, quat)
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
            body_id = self.label_body[label]
            pos, quat = self.data.xpos[body_id], self.data.xquat[body_id]
            out.append(_body_to_world(local, pos, quat))
        return np.asarray(out)

    def get_object_by_keypoint(self, keypoint_idx: int) -> int:
        return self._keypoint2object[int(keypoint_idx)]

    def is_grasping(self, candidate_obj=None) -> bool:
        """Contact between a gripper pad and the (candidate) object geom."""
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

    def get_arm_joint_postions(self) -> np.ndarray:
        return self.data.qpos[self.arm_qposadr].copy()

    def reset_joint_positions(self) -> np.ndarray:
        return self.reset_joint_pos.copy()

    # ------------------------------------------------------------------- IK
    def ik(self, target_pos, target_rot=None, seed=None, iterations: int = 1500,
           pos_tol: float = 2e-5, rot_tol: float = 2e-4, step: float = 0.25) -> np.ndarray:
        target_pos = np.asarray(target_pos, dtype=float)
        target_rot = self.target_rot if target_rot is None else np.asarray(target_rot, dtype=float)
        q = (self.reset_joint_pos.copy() if seed is None else np.asarray(seed, dtype=float).copy())
        q = np.clip(q, self.arm_lo + 1e-5, self.arm_hi - 1e-5)
        for _ in range(iterations):
            self.data.qpos[self.arm_qposadr] = q
            mujoco.mj_forward(self.model, self.data)
            pos_err = target_pos - self.data.site_xpos[self.pinch_id]
            cur_rot = self.data.site_xmat[self.pinch_id].reshape(3, 3)
            rot_err = 0.5 * sum(
                np.cross(cur_rot[:, i], target_rot[:, i]) for i in range(3)
            )
            if np.linalg.norm(pos_err) < pos_tol and np.linalg.norm(rot_err) < rot_tol:
                return q
            jacp = np.zeros((3, self.model.nv))
            jacr = np.zeros((3, self.model.nv))
            mujoco.mj_jacSite(self.model, self.data, jacp, jacr, self.pinch_id)
            dof = self.arm_dofadr
            jac = np.vstack([jacp[:, dof], jacr[:, dof]])
            err = np.concatenate([pos_err, rot_err])
            dq = jac.T @ np.linalg.solve(jac @ jac.T + 1e-4 * np.eye(6), err)
            q = np.clip(q + step * dq, self.arm_lo + 1e-5, self.arm_hi - 1e-5)
        raise RuntimeError(f"IK failed for target {target_pos.round(3).tolist()}")

    def _move_to(self, q_target, gripper: float, steps: int = 60) -> None:
        for _ in range(steps):
            self.data.ctrl[:6] = q_target
            self.data.ctrl[6] = gripper
            mujoco.mj_step(self.model, self.data)
            self._record_frame()
        self.step_counter += 1

    def execute_action(self, action, precise: bool = True):
        """action = [x,y,z,qx,qy,qz,qw, gripper] with gripper 1=close,-1=open,0=null."""
        action = np.asarray(action, dtype=float)
        if action.shape != (8,):
            raise ValueError(f"action must be length 8, got {action.shape}")
        target_pos = action[:3]
        target_quat_xyzw = action[3:7]
        gripper = action[7]
        target_rot = _quat_xyzw_to_mat(target_quat_xyzw)
        q = self.ik(target_pos, target_rot, seed=self.get_arm_joint_postions())
        g = self.last_og_gripper_action
        if gripper == self.get_gripper_close_action():
            g = 255.0
        elif gripper == self.get_gripper_open_action():
            g = 0.0
        self._move_to(q, g, steps=80 if precise else 40)
        self.last_og_gripper_action = g
        pos_err = float(np.linalg.norm(target_pos - self.data.site_xpos[self.pinch_id]))
        return pos_err, 0.0

    def open_gripper(self) -> None:
        if self.last_og_gripper_action == 0.0:
            return
        self._move_to(self.get_arm_joint_postions(), 0.0, steps=30)
        self.last_og_gripper_action = 0.0

    def close_gripper(self) -> None:
        if self.last_og_gripper_action == 255.0:
            return
        self._move_to(self.get_arm_joint_postions(), 255.0, steps=30)
        self.last_og_gripper_action = 255.0

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
            name = (self.model.geom(geom_id).name or "")
            is_gripper = body_id in body_ids and ("pad" in name or "gripper" in name)
            if not is_gripper and body_id not in grasped_bodies:
                continue
            if self.model.geom_type[geom_id] != mujoco.mjtGeom.mjGEOM_MESH:
                continue
            mesh_id = self.model.geom_dataid[geom_id]
            if mesh_id < 0:
                continue
            vadr, vnum = self.model.mesh_vertadr[mesh_id], self.model.mesh_vertnum[mesh_id]
            verts = self.model.mesh_vert[vadr:vadr + vnum].reshape(-1, 3)
            if self.model.mesh_scale[mesh_id] is not None:
                verts = verts * self.model.mesh_scale[mesh_id]
            idx = self.rng.choice(len(verts), size=min(200, len(verts)), replace=False)
            local = verts[idx]
            world = local @ self.data.geom_xmat[geom_id].reshape(3, 3).T + self.data.geom_xpos[geom_id]
            pts.append(world)
        if not pts:
            return np.zeros((0, 3))
        return np.concatenate(pts, axis=0)

    def _robot_body_ids(self) -> set[int]:
        ids = set()
        base = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "base_link")
        for body_id in range(self.model.nbody):
            name = self.model.body(body_id).name or ""
            if body_id == base or any(
                tok in name for tok in ("Link", "r2f85", "robotiq", "base_link")
            ):
                ids.add(body_id)
        return ids

    # ----------------------------------------------------------------- reset
    def reset(self) -> None:
        mujoco.mj_resetData(self.model, self.data)
        # place the cube at a fixed pick position
        self.data.qpos[self.cube_qposadr:self.cube_qposadr + 3] = [0.25, -0.42, 0.05]
        self.data.qpos[self.cube_qposadr + 3:self.cube_qposadr + 7] = [1, 0, 0, 0]
        mujoco.mj_forward(self.model, self.data)
        self.video_cache = []
        self.last_og_gripper_action = 1.0

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


def _world_to_body(point: np.ndarray, body_pos: np.ndarray, body_quat_wxyz: np.ndarray) -> np.ndarray:
    quat = np.array([body_quat_wxyz[1], body_quat_wxyz[2], body_quat_wxyz[3], body_quat_wxyz[0]])
    rot = _quat_xyzw_to_mat(quat)
    return rot.T @ (point - body_pos)


def _body_to_world(local: np.ndarray, body_pos: np.ndarray, body_quat_wxyz: np.ndarray) -> np.ndarray:
    quat = np.array([body_quat_wxyz[1], body_quat_wxyz[2], body_quat_wxyz[3], body_quat_wxyz[0]])
    rot = _quat_xyzw_to_mat(quat)
    return rot @ local + body_pos
