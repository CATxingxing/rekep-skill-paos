"""Verify injected camera + depth back-projection against the cube's true pose."""

from __future__ import annotations

import os
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")

import mujoco  # noqa: E402
import numpy as np  # noqa: E402

from rekep_sim.env.camera import backproject, inject_camera, intrinsics  # noqa: E402

SCENE = (
    Path(__file__).resolve().parents[1]
    / "skill-src/rekep-sim/assets/dobot-nova2-robotiq/mjcf/dobot_nova2_robotiq_2f85_pick_place.xml"
)
EYE = (0.55, -0.80, 0.80)
TARGET = (0.05, -0.30, 0.15)
FOVY = 45.0


def main() -> int:
    spec = mujoco.MjSpec.from_file(str(SCENE))
    inject_camera(spec, "vlm", EYE, TARGET, FOVY)
    model = spec.compile()
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)

    cam_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, "vlm")
    print("cam_id", cam_id, "ncam", model.ncam)
    print("cam_xpos", data.cam_xpos[cam_id].round(4).tolist())
    cam_mat = data.cam_xmat[cam_id].reshape(3, 3)

    h, w = 240, 320
    renderer = mujoco.Renderer(model, height=h, width=w)
    renderer.update_scene(data, camera="vlm")
    rgb = renderer.render().copy()
    renderer.enable_depth_rendering()
    renderer.update_scene(data, camera="vlm")
    depth = renderer.render().copy()
    renderer.enable_segmentation_rendering()
    renderer.update_scene(data, camera="vlm")
    seg = renderer.render().copy()

    cube_geom = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "pick_cube_geom")
    cube_joint = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "pick_cube_free")
    cube_pos = data.qpos[model.jnt_qposadr[cube_joint]:model.jnt_qposadr[cube_joint] + 3]
    mask = seg[..., 0] == cube_geom
    print("cube geom", cube_geom, "pixels", int(mask.sum()), "true cube", cube_pos.round(3).tolist())

    intr = intrinsics(FOVY, h, w)
    points = backproject(depth, data.cam_xpos[cam_id], cam_mat, intr)
    if mask.sum() > 0:
        uu, vv = np.where(mask)
        center = (int(np.median(uu)), int(np.median(vv)))
        est = points[center[0], center[1]]
        print("center pixel", center, "depth", round(float(depth[center]), 4))
        print("backprojected", est.round(4).tolist())
        print("error (m)", round(float(np.linalg.norm(est - cube_pos)), 4))
    renderer.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
