"""End-to-end (offline, no VLM): perception -> template program -> staged execution."""

from __future__ import annotations

import os
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")
ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("REKEP_DINOV2_REPO", "/data7/home/linjiongxiao/.paos-rekep/models/dinov2")
os.environ.setdefault(
    "REKEP_DINOV2_WEIGHTS", "/data7/home/linjiongxiao/.paos-rekep/models/weights/dinov2_vits14.pth"
)

import json  # noqa: E402
import numpy as np  # noqa: E402

from rekep_sim.env.mujoco_env import MujocoReKepEnv  # noqa: E402
from rekep_sim.perception import perceive  # noqa: E402
from rekep_sim.runtime import ReKepRuntime, RuntimeConfig  # noqa: E402

SCENE = ROOT / "skill-src/rekep-sim/assets/dobot-nova2-robotiq/mjcf/dobot_nova2_robotiq_2f85_pick_place.xml"


def main() -> int:
    out = ROOT / "evidence/P3"
    out.mkdir(parents=True, exist_ok=True)
    env = MujocoReKepEnv(SCENE, height=240, width=320)
    try:
        snapshot, projected, kp = perceive(env)
        print("keypoints:", [(k["index"], k["object"], [round(x, 3) for x in k["position_m"]]) for k in snapshot["keypoints"]])
        rt = ReKepRuntime(env, RuntimeConfig())
        program = rt.plan("pick the orange cube and place it in the green target zone", snapshot)
        print("program stages:", program["num_stages"], "grasp", program["grasp_keypoints"], "release", program["release_keypoints"])
        result = rt.execute(program, allow_motion=True, progress=lambda d: print("progress", d))
        print("RESULT", json.dumps({k: v for k, v in result.items() if k != "stages"}, indent=2))
        print("stages", result.get("stages"))
        return 0 if result.get("status") == "succeeded" else 1
    finally:
        env.close()


if __name__ == "__main__":
    raise SystemExit(main())
