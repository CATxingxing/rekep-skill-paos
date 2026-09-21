"""Smoke test: headless MuJoCo rendering (rgb/depth/segmentation) of the robot scene."""

from __future__ import annotations

import os
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")

import mujoco  # noqa: E402
import numpy as np  # noqa: E402

SCENE = (
    Path(__file__).resolve().parents[1]
    / "skill-src/rekep-sim/assets/dobot-nova2-robotiq/mjcf/dobot_nova2_robotiq_2f85_pick_place.xml"
)


def main() -> int:
    print("scene:", SCENE, SCENE.is_file())
    model = mujoco.MjModel.from_xml_path(str(SCENE))
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    print("nq", model.nq, "nu", model.nu, "ngeom", model.ngeom, "ncam", model.ncam)

    renderer = mujoco.Renderer(model, height=240, width=320)
    cam = mujoco.MjvCamera()
    mujoco.mjv_defaultFreeCamera(model, cam)
    cam.distance, cam.azimuth, cam.elevation = 0.95, 135, -25
    cam.lookat[:] = [0.0, -0.15, 0.2]

    renderer.update_scene(data, cam)
    rgb = renderer.render().copy()
    print("rgb", rgb.shape, rgb.dtype, "mean", round(float(rgb.mean()), 2))

    renderer.enable_depth_rendering()
    renderer.update_scene(data, cam)
    depth = renderer.render().copy()
    print("depth", depth.shape, "min", round(float(depth.min()), 4), "max", round(float(depth.max()), 4))

    renderer.enable_segmentation_rendering()
    renderer.update_scene(data, cam)
    seg = renderer.render().copy()
    print("seg", seg.shape, seg.dtype, "geoms", sorted(int(x) for x in np.unique(seg[..., 0]))[:12])

    # pinch site + object positions
    pinch = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "pinch")
    cube = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "pick_cube_free")
    print("pinch site id", pinch, "pick_cube_free joint", cube)
    print("pinch pos", data.site_xpos[pinch].round(3).tolist())
    print("cube pos", data.qpos[model.jnt_qposadr[cube]:model.jnt_qposadr[cube]+3].round(3).tolist())
    renderer.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
