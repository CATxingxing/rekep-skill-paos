"""Camera helpers for the MuJoCo ReKep environment.

We inject a fixed named camera into the scene at load time (via ``MjSpec``);
the asset XML on disk is never modified. Given camera pose + vertical fov we
can render RGB, depth and segmentation and back-project depth to world points,
which is what the reference ``keypoint_proposal`` consumes.
"""

from __future__ import annotations

import numpy as np


def look_at_quat(eye, target, up=(0.0, 0.0, 1.0)) -> np.ndarray:
    """Quaternion (w,x,y,z) for a camera at ``eye`` looking at ``target``.

    MuJoCo/OpenGL camera convention: the camera looks along its local -Z axis,
    local +Y is up.
    """
    eye = np.asarray(eye, dtype=float)
    target = np.asarray(target, dtype=float)
    up = np.asarray(up, dtype=float)

    forward = target - eye
    forward /= np.linalg.norm(forward)
    z_cam = -forward
    x_cam = np.cross(up, z_cam)
    if np.linalg.norm(x_cam) < 1e-9:
        x_cam = np.cross(np.array([0.0, 1.0, 0.0]), z_cam)
    x_cam /= np.linalg.norm(x_cam)
    y_cam = np.cross(z_cam, x_cam)
    rot = np.column_stack([x_cam, y_cam, z_cam])  # columns are camera axes in world

    # rotation matrix -> quaternion (w,x,y,z), Shepperd's method
    m = rot
    trace = m[0, 0] + m[1, 1] + m[2, 2]
    if trace > 0:
        s = 0.5 / np.sqrt(trace + 1.0)
        w = 0.25 / s
        x = (m[2, 1] - m[1, 2]) * s
        y = (m[0, 2] - m[2, 0]) * s
        z = (m[1, 0] - m[0, 1]) * s
    elif m[0, 0] > m[1, 1] and m[0, 0] > m[2, 2]:
        s = 2.0 * np.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2])
        w = (m[2, 1] - m[1, 2]) / s
        x = 0.25 * s
        y = (m[0, 1] + m[1, 0]) / s
        z = (m[0, 2] + m[2, 0]) / s
    elif m[1, 1] > m[2, 2]:
        s = 2.0 * np.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2])
        w = (m[0, 2] - m[2, 0]) / s
        x = (m[0, 1] + m[1, 0]) / s
        y = 0.25 * s
        z = (m[1, 2] + m[2, 1]) / s
    else:
        s = 2.0 * np.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1])
        w = (m[1, 0] - m[0, 1]) / s
        x = (m[0, 2] + m[2, 0]) / s
        y = (m[1, 2] + m[2, 1]) / s
        z = 0.25 * s
    quat = np.array([w, x, y, z], dtype=float)
    return quat / np.linalg.norm(quat)


def inject_camera(spec, name, eye, target, fovy, up=(0.0, 0.0, 1.0)):
    cam = spec.worldbody.add_camera()
    cam.name = name
    cam.pos = list(np.asarray(eye, dtype=float))
    cam.quat = list(look_at_quat(eye, target, up))
    cam.fovy = float(fovy)
    return cam


def intrinsics(fovy_deg: float, height: int, width: int) -> dict:
    fovy = np.deg2rad(fovy_deg)
    fy = 0.5 * height / np.tan(fovy / 2.0)
    fx = fy  # square pixels
    return {"fx": fx, "fy": fy, "cx": width / 2.0, "cy": height / 2.0,
            "height": height, "width": width, "fovy_deg": float(fovy_deg)}


def backproject(
    depth: np.ndarray,
    cam_pos: np.ndarray,
    cam_mat: np.ndarray,
    intr: dict,
) -> np.ndarray:
    """Back-project a MuJoCo depth image to world points.

    ``depth`` is the renderer's depth buffer (metres). ``cam_mat`` is the camera
    rotation matrix (3x3, columns = camera axes in world), ``cam_pos`` its world
    position. Returns (H, W, 3) world coordinates.
    """
    h, w = depth.shape
    us, vs = np.meshgrid(np.arange(w), np.arange(h))
    z = depth.astype(np.float64)
    x_cam = (us - intr["cx"]) / intr["fx"] * z
    y_cam = -(vs - intr["cy"]) / intr["fy"] * z
    z_cam = -z  # camera looks along -Z
    pts_cam = np.stack([x_cam, y_cam, z_cam], axis=-1)
    cam_mat = np.asarray(cam_mat, dtype=float).reshape(3, 3)
    world = pts_cam @ cam_mat.T + np.asarray(cam_pos, dtype=float)
    return world.astype(np.float32)
