"""Perception node: reference DINOv2 keypoint proposal on the MuJoCo scene."""

from __future__ import annotations

import os
import threading
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")

import cv2  # noqa: E402

from rekep_sim.env.mujoco_env import MujocoReKepEnv  # noqa: E402
from rekep_sim.perception import perceive, warmup  # noqa: E402
from rekep_sim.provider import Provider  # noqa: E402
from rekep_sim.state import scene_path, state_dir, write_json  # noqa: E402


def main() -> None:
    env = MujocoReKepEnv(
        scene_path(),
        height=int(os.environ.get("REKEP_SIM_HEIGHT", "240")),
        width=int(os.environ.get("REKEP_SIM_WIDTH", "320")),
    )
    # Warm the DINOv2 model in the background: register the provider immediately
    # (fast node readiness) while the model loads.
    threading.Thread(target=warmup, args=(env,), daemon=True).start()

    def handler(arguments, _cancel, _progress):
        seed = int(arguments.get("seed", 0))
        snapshot, projected, _kp = perceive(env, seed=seed)
        sd = state_dir()
        tag = snapshot["observation_id"].split(":")[-1][:16]
        overlay = sd / f"perception_{tag}.png"
        cv2.imwrite(str(overlay), projected[..., ::-1])
        snap = {
            k: snapshot[k]
            for k in (
                "schema_version",
                "observation_id",
                "rgb_sha256",
                "bounds_min",
                "bounds_max",
                "keypoints",
            )
        }
        snap["overlay_path"] = str(overlay)
        write_json(sd / "latest_perception.json", snap)
        return {
            "profile": "sim",
            "backend": "mujoco-headless",
            "robot": "dobot_nova2",
            "gripper": "robotiq_2f85",
            "readiness": "ready",
            "perception": snap,
        }

    Provider(
        endpoint_id="rekep.perception",
        operation="get_context",
        semantics="query",
        handler=handler,
    ).run()


if __name__ == "__main__":
    main()
