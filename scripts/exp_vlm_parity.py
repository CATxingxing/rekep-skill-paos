"""Controlled experiment: VLM constraint generation + ReKep execution (minimal API use).

Mirrors the reference pipeline (perception -> constraint_generation -> sandbox ->
subgoal/path solvers -> MuJoCo execution) in-process, using exactly ONE VLM model
call. Keys are read from the environment and never printed.
"""

from __future__ import annotations

import os
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")
ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("REKEP_DINOV2_REPO", "/data7/home/linjiongxiao/.paos-rekep/models/dinov2")
os.environ.setdefault(
    "REKEP_DINOV2_WEIGHTS", "/data7/home/linjiongxiao/.paos-rekep/models/weights/dinov2_vits14.pth"
)

import cv2  # noqa: E402
import numpy as np  # noqa: E402

from rekep_sim import refimpl  # noqa: E402
from rekep_sim.env.mujoco_env import MujocoReKepEnv  # noqa: E402
from rekep_sim.perception import perceive  # noqa: E402
from rekep_sim.runtime import ReKepRuntime, RuntimeConfig  # noqa: E402

SCENE = ROOT / "skill-src/rekep-sim/assets/dobot-nova2-robotiq/mjcf/dobot_nova2_robotiq_2f85_pick_place.xml"
INSTRUCTION = "pick the orange cube and place it in the green target zone"


def main() -> int:
    out = ROOT / "evidence/P5v"
    out.mkdir(parents=True, exist_ok=True)

    # redirect VLM writes away from the reference submodule
    refimpl.ensure_reference_on_path()
    import constraint_generation

    _orig_init = constraint_generation.ConstraintGenerator.__init__

    def _init(self, config):
        _orig_init(self, config)
        self.base_dir = str(out)

    constraint_generation.ConstraintGenerator.__init__ = _init

    env = MujocoReKepEnv(SCENE, height=240, width=320)
    try:
        snapshot, projected, _kp = perceive(env)
        overlay = out / "overlay.png"
        cv2.imwrite(str(overlay), projected[..., ::-1])
        snapshot["overlay_path"] = str(overlay)
        print("PERCEPTION keypoints:", [(k["index"], k["object"]) for k in snapshot["keypoints"]])

        runtime = ReKepRuntime(env, RuntimeConfig())
        program = runtime.plan(INSTRUCTION, snapshot, source="vlm")
        print("VLM model:", os.environ.get("REKEP_VLM_MODEL"))
        print("PROGRAM source:", program["source"], "num_stages:", program["num_stages"],
              "grasp:", program["grasp_keypoints"], "release:", program["release_keypoints"])
        print("VLM trace:", program.get("vlm_trace"))
        for st in program["stages"]:
            print(f"  stage{st['stage']} subgoal={st['subgoal_constraints']!r} path={st['path_constraints']!r}")

        result = runtime.execute(program, allow_motion=True, progress=lambda d: print("progress", d))
        print("RESULT:", {k: result.get(k) for k in ("status", "video", "place_error_m", "elapsed_s")})
        ok = result.get("status") == "succeeded" and result.get("place_error_m", 1.0) < 0.06
        print("VLM_ACCEPTANCE:", "PASS" if ok else "FAIL")
        return 0 if ok else 1
    finally:
        env.close()


if __name__ == "__main__":
    raise SystemExit(main())
