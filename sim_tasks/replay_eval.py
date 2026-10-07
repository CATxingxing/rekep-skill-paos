"""Score a live PAOS simulation from mujoco_sim's ground-truth state log.

The task profiles enable ``state_log`` in ``mujoco-simulator.yaml`` (patched
mujoco_sim): full qpos/qvel/ctrl at a fixed rate, written to a JSONL file that
no ReKep node reads. This tool rebuilds the same seeded scene, checks that it
is the scene the simulator loaded, replays every logged state of one episode
through the task evaluator and runs the final stability probe on a copy.

    python -m sim_tasks.replay_eval --task stack_blocks --seed 0 --log <file.jsonl> \
        [--episode 0] [--start-wall-ns N] [--end-wall-ns N] [--out report.json]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import mujoco
import numpy as np

from . import make_env


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--log", type=Path, required=True)
    parser.add_argument("--scene", type=Path, help="scene XML the simulator loaded (checked against the seed)")
    parser.add_argument("--episode", type=int)
    parser.add_argument("--start-wall-ns", type=int, default=0)
    parser.add_argument("--end-wall-ns", type=int, default=2**63 - 1)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    env = make_env(args.task)
    # The scene XML is checked against --scene below and every state comes from the log, so the
    # generator's rest check (version dependent: mujoco_sim bundles MuJoCo 3.3.7) is not needed here.
    env.reset(seed=args.seed, check_settle=False)
    if args.scene is not None:
        from .make_profiles import relocatable

        expected = relocatable(env.scene_path.read_text(encoding="utf-8"))
        if args.scene.read_text(encoding="utf-8") != expected:
            raise SystemExit(f"{args.scene} is not the seed-{args.seed} {args.task} scene")
    states = []
    header = None
    for line in args.log.read_text(encoding="utf-8").splitlines():
        record = json.loads(line)
        if record.get("type") == "header":
            header = record
            continue
        if args.episode is not None and record["episode_index"] != args.episode:
            continue
        if not args.start_wall_ns <= record["wall_ns"] <= args.end_wall_ns:
            continue
        states.append(record)
    if header is None or not states:
        raise SystemExit("state log has no header or no state in the selected window")
    model, data = env.model, env.data
    if (header["nq"], header["nv"], header["nu"]) != (model.nq, model.nv, model.nu):
        raise SystemExit("state log does not match the scene model dimensions")

    def load(record: dict) -> None:
        data.qpos[:] = record["qpos"]
        data.qvel[:] = record["qvel"]
        data.ctrl[:] = record["ctrl"]
        if model.na:
            data.act[:] = record["act"]
        data.time = float(record["time"])
        mujoco.mj_forward(model, data)

    load(states[0])
    env.evaluator.reset(env.snapshot())
    env.t_start = float(data.time)
    for record in states[1:]:
        load(record)
        env.evaluator.observe(env.snapshot())
    evaluation = env.evaluate(final=True)
    report = {
        "task": args.task, "seed": args.seed, "instruction": env.get_instruction(),
        "log": str(args.log), "mujoco_version": mujoco.__version__, "states": len(states),
        "sim_span_s": states[-1]["sim_time"] - states[0]["sim_time"],
        "wall_span_s": (states[-1]["wall_ns"] - states[0]["wall_ns"]) / 1e9,
        "evaluation": json.loads(json.dumps(evaluation, default=lambda o: getattr(o, "value", str(o)))),
    }
    text = json.dumps(report, indent=1, default=lambda o: o.tolist() if isinstance(o, np.ndarray) else str(o))
    if args.out:
        args.out.write_text(text + "\n", encoding="utf-8")
    print(text)
    return 0 if evaluation["success"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
