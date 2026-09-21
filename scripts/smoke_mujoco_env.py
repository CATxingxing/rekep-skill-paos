"""Smoke: MuJoCo env -> reference keypoint proposal -> keypoint tracking -> IK move."""

from __future__ import annotations

import os
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")
ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("REKEP_DINOV2_REPO", "/data7/home/linjiongxiao/.paos-rekep/models/dinov2")
os.environ.setdefault(
    "REKEP_DINOV2_WEIGHTS", "/data7/home/linjiongxiao/.paos-rekep/models/weights/dinov2_vits14.pth"
)

import numpy as np  # noqa: E402

from rekep_sim import refimpl  # noqa: E402
from rekep_sim.env.mujoco_env import MujocoReKepEnv  # noqa: E402

SCENE = ROOT / "skill-src/rekep-sim/assets/dobot-nova2-robotiq/mjcf/dobot_nova2_robotiq_2f85_pick_place.xml"


def main() -> int:
    env = MujocoReKepEnv(SCENE, height=240, width=320)
    obs = env.get_cam_obs()[0]
    print("rgb", obs["rgb"].shape, "depth", obs["depth"].shape, "points", obs["points"].shape,
          "seg labels", sorted(int(x) for x in np.unique(obs["seg"])))

    cfg = {
        "num_candidates_per_mask": 5,
        "min_dist_bt_keypoints": 0.06,
        "max_mask_ratio": 0.5,
        "device": "cpu",
        "bounds_min": env.bounds_min.tolist(),
        "bounds_max": env.bounds_max.tolist(),
        "seed": 0,
    }
    prop = refimpl.build_keypoint_proposer(cfg)
    np.random.seed(0)
    kp, proj, meta = prop.get_keypoints(obs["rgb"], obs["points"], obs["seg"], return_metadata=True)
    print("proposed keypoints:", len(kp))
    print(kp[:5].round(3).tolist() if len(kp) else "none")
    if len(kp) == 0:
        raise SystemExit("no keypoints proposed")

    env.register_keypoints(kp)
    tracked = env.get_keypoint_positions()
    print("tracked keypoints:", len(tracked), tracked[:3].round(3).tolist())

    ee0 = env.get_ee_pose()
    print("ee0", ee0.round(3).tolist())
    cube_pos = env.data.qpos[env.cube_qposadr:env.cube_qposadr + 3]
    target = np.array([cube_pos[0], cube_pos[1], 0.22])
    print("target", target.round(3).tolist())
    action = np.concatenate([target, env.get_ee_quat(), [env.get_gripper_null_action()]])
    env.execute_action(action, precise=True)
    ee1 = env.get_ee_pose()
    print("ee1", ee1.round(3).tolist(), "pos err", round(float(np.linalg.norm(ee1[:3]-target[:3])), 4))

    out = ROOT / "evidence/P3"
    out.mkdir(parents=True, exist_ok=True)
    video = env.save_video(str(out / "smoke_mujoco_env.mp4"))
    print("video", video, "frames", len(env.video_cache))
    env.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
