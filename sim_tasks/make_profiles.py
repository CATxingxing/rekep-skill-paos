"""Create profiles/sim-dobot-nova2-robotiq-<task>/ from the baseline profile.

Each task profile is a copy of the baseline profile except
``mujoco-simulator.yaml`` (``model_path`` -> the task's generated scene).
``perception.yaml``, ``executor.yaml`` and ``gripper-action-controller.yaml`` of
the task profiles are maintained by hand (shared palette perception;
table-height workspace; stall threshold for curved objects) and are not
overwritten once they exist.  Scenes are written by
``python -m sim_tasks.make_profiles --seed N`` to
``assets/dobot-nova2-robotiq/mjcf/tasks/<task>.xml`` (inside the packaged
Skill, mesh paths relative), i.e. the scene of the most recent call; a copy
also goes to ``sim_tasks/generated/<task>.xml``.
"""
from __future__ import annotations

import argparse
import shutil

from . import TASKS, make_env
from .config import GENERATED_DIR, REPO_ROOT

BASELINE = REPO_ROOT / "profiles" / "sim-dobot-nova2-robotiq"
TASK_SCENES = REPO_ROOT / "assets" / "dobot-nova2-robotiq" / "mjcf" / "tasks"
HAND_MAINTAINED = {"perception.yaml", "executor.yaml", "gripper-action-controller.yaml"}


def make_profile(task: str) -> None:
    target = REPO_ROOT / "profiles" / f"sim-dobot-nova2-robotiq-{task.replace('_', '-')}"
    target.mkdir(exist_ok=True)
    for src in BASELINE.iterdir():
        if src.name in HAND_MAINTAINED and (target / src.name).exists():
            continue
        text = src.read_text(encoding="utf-8")
        if src.name == "mujoco-simulator.yaml":
            old = next(line for line in text.splitlines() if line.startswith("model_path:"))
            text = text.replace(old, f"model_path: ../../assets/dobot-nova2-robotiq/mjcf/tasks/{task}.xml")
            text += ("# Test-only ground truth for sim_tasks.replay_eval (patched mujoco_sim,\n"
                     "# patches/mujoco/0001). No Dora output carries it; no ReKep node reads it.\n"
                     f"state_log: {{path: \"~/.paos-rekep/run/rekep/ground-truth/{task}-{{start_ns}}.jsonl\", hz: 25}}\n")
        (target / src.name).write_text(text, encoding="utf-8")


def relocatable(xml: str) -> str:
    """Mesh directory relative to the scene file (assets/.../mjcf/tasks/)."""
    import re

    return re.sub(r'meshdir="[^"]*"', 'meshdir=".."', xml, count=1)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--tasks", nargs="*", default=sorted(TASKS))
    args = parser.parse_args()
    GENERATED_DIR.mkdir(exist_ok=True)
    TASK_SCENES.mkdir(exist_ok=True)
    for task in args.tasks:
        make_profile(task)
        env = make_env(task)
        env.reset(seed=args.seed)
        shutil.copy2(env.scene_path, GENERATED_DIR / f"{task}.xml")
        (TASK_SCENES / f"{task}.xml").write_text(relocatable(env.scene_path.read_text(encoding="utf-8")), encoding="utf-8")
        (TASK_SCENES / f"{task}.json").write_text(__import__("json").dumps({"task": task, "seed": args.seed, "instruction": env.get_instruction()}, indent=1) + "\n", encoding="utf-8")
        print(task, "seed", args.seed, "->", TASK_SCENES / f"{task}.xml")
        env.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
