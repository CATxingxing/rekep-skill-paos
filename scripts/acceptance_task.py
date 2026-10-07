#!/usr/bin/env python3
"""One explicit sim_tasks attempt through PAOS; records evidence, never retries an Action.

get_context -> plan_task (VLM) -> preflight (solver + sequential IK) ->
execute_task, exactly one POST each. The verdict comes from the task's
ground-truth evaluator replaying the patched mujoco_sim state log of this
attempt's time window (sim_tasks.replay_eval); no ReKep node sees it.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))

from acceptance_baseline import Gateway, preflight, write_video  # noqa: E402
from rekep_core.contracts import atomic_json, runtime_root  # noqa: E402

SIM_PYTHON = "/data0/wangxinran/.paos-rekep/envs/rekep-sim/bin/python"
# mujoco_sim bundles MuJoCo 3.3.7; replaying its states under another version (rekep-sim has 3.13)
# shifts contact equilibria, so the stability probe would measure a replay artefact.
REPLAY_MUJOCO = "/data0/wangxinran/.paos-rekep/envs/mujoco-3.3.7"


def ground_truth(task: str, seed: int, start_ns: int, end_ns: int, output: Path) -> dict:
    logs = sorted((runtime_root() / "run" / "rekep" / "ground-truth").glob(f"{task}-*.jsonl"), key=lambda path: path.stat().st_mtime)
    if not logs:
        return {"error": "no ground-truth state log; is the patched mujoco_sim installed?"}
    command = [SIM_PYTHON, "-m", "sim_tasks.replay_eval", "--task", task, "--seed", str(seed), "--log", str(logs[-1]),
               "--scene", str(ROOT / "assets/dobot-nova2-robotiq/mjcf/tasks" / f"{task}.xml"),
               "--start-wall-ns", str(start_ns), "--end-wall-ns", str(end_ns), "--out", str(output / "ground-truth-eval.json")]
    completed = subprocess.run(command, cwd=ROOT, capture_output=True, text=True, timeout=900, env={"MUJOCO_GL": "egl", "PATH": "/usr/bin:/bin", "PYTHONPATH": REPLAY_MUJOCO})
    (output / "ground-truth-eval.log").write_text(completed.stdout[-20000:] + completed.stderr[-20000:])
    if not (output / "ground-truth-eval.json").is_file():
        return {"error": f"replay_eval failed with rc={completed.returncode}"}
    return json.loads((output / "ground-truth-eval.json").read_text())


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", required=True, choices=("general_pickup", "stack_blocks", "push_t"))
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--gateway", default="http://127.0.0.1:19021")
    parser.add_argument("--motion-source", type=Path, default=ROOT.parent / "operator-motion/packages/motion_server/src")
    parser.add_argument("--allow-motion", action="store_true")
    parser.add_argument("--settle-s", type=float, default=3.0, help="hands-off time after execution before the window closes")
    parser.add_argument("--recordings", type=Path, default=runtime_root() / "run" / "rekep" / "recordings")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    sys.path.insert(0, str(args.motion_source.resolve()))
    scene = json.loads((ROOT / "assets/dobot-nova2-robotiq/mjcf/tasks" / f"{args.task}.json").read_text())
    profile = ROOT / "profiles" / f"sim-dobot-nova2-robotiq-{args.task.replace('_', '-')}"
    instruction = scene["instruction"]
    atomic_json(args.output / "task.json", {**scene, "profile": profile.name})
    gateway = Gateway(args.gateway, args.output)
    started_ns = time.time_ns()
    program, terminal, motion_admitted = None, None, False
    verdict: dict = {"task": args.task, "seed": scene["seed"], "instruction": instruction}
    try:
        for tool in ("rekep.get_context", "rekep.plan_task", "rekep.execute_task"):
            context = gateway.request(f"/tools/{tool}/context")
            atomic_json(args.output / f"{tool}-context.json", context)
            if not context["ready"]:
                raise RuntimeError(f"tool not ready: {tool}")
        before = gateway.snapshot("before")
        plan = gateway.action("rekep.plan_task", {
            "instruction": instruction, "session_id": before["session_id"], "observation_id": before["observation_id"],
        }, "plan")
        if plan["phase"] != "completed" or plan["result"]["status"] != "succeeded":
            raise RuntimeError(f"planning failed: {plan}")
        program = plan["result"]["outputs"]
        atomic_json(args.output / "program.json", program)
        checked = preflight(program, before, profile)
        atomic_json(args.output / "preflight.json", checked)
        print("preflight passed", len(checked["ik_evidence"]), "IK waypoints", flush=True)
        if not args.allow_motion:
            return 0
        motion_admitted = True  # Also covers a POST with a lost admission response.
        terminal = gateway.action("rekep.execute_task", {
            **checked["binding"], "allow_motion": True, "deadline_ms": 600000,
        }, "execute")
    except Exception as exc:
        atomic_json(args.output / "error.json", {"type": type(exc).__name__, "message": str(exc), "motion_post_may_have_been_sent": motion_admitted})
        print(type(exc).__name__, str(exc), flush=True)
    finally:
        if motion_admitted:
            time.sleep(args.settle_s)
            try:
                gateway.snapshot("after")
            except Exception as exc:
                atomic_json(args.output / "after-error.json", {"type": type(exc).__name__, "message": str(exc)})
        end_ns = time.time_ns()
        try:
            recording = write_video(args.output, args.recordings, started_ns, end_ns)
        except Exception as exc:  # the verdict never depends on the video
            recording = {"error": f"{type(exc).__name__}: {exc}"}
        atomic_json(args.output / "recording.json", recording)
        if motion_admitted:
            evaluation = ground_truth(args.task, int(scene["seed"]), started_ns, end_ns, args.output)
            execution_ok = bool(terminal and terminal["phase"] == "completed" and terminal.get("result", {}).get("status") == "succeeded")
            success = bool(evaluation.get("evaluation", {}).get("success"))
            verdict.update({"execution_succeeded": execution_ok, "ground_truth_success": success,
                            "ground_truth": evaluation.get("evaluation", evaluation), "task_passed": execution_ok and success})
        atomic_json(args.output / "verification.json", verdict)
        print("verdict", json.dumps({k: v for k, v in verdict.items() if k != "ground_truth"}), flush=True)
    return 0 if verdict.get("task_passed") else 1


if __name__ == "__main__":
    raise SystemExit(main())
