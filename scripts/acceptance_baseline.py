#!/usr/bin/env python3
"""One explicit baseline attempt; records evidence, never retries an Action."""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "nodes/common"), str(ROOT / "nodes/rekep-executor"), str(ROOT / "nodes/rekep-perception")]

from recorder import cut_video
from rekep_core.contracts import atomic_json, runtime_root, validate_program, validate_snapshot, verify_binding


class Gateway:
    def __init__(self, url: str, output: Path):
        self.url, self.output = url.rstrip("/"), output

    def request(self, path: str, body: dict | None = None) -> dict:
        req = urllib.request.Request(
            self.url + path, data=None if body is None else json.dumps(body).encode(),
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=610 if body is not None else 15) as response:
            value = json.load(response)
        if not value.get("ok"):
            raise RuntimeError(value)
        return value["data"]

    def snapshot(self, label: str) -> dict:
        data = self.request("/tools/rekep.perception/get_context:invoke", {
            "arguments": {}, "caller_id": "rekep-baseline", "timeout_ms": 600000,
        })
        atomic_json(self.output / f"{label}-response.json", data)
        result = data["response"]["result"]
        if result["status"] != "succeeded":
            raise RuntimeError(result)
        snapshot = validate_snapshot(result["outputs"]["perception"])
        atomic_json(self.output / f"{label}.json", snapshot)
        shutil.copy2(snapshot["overlay_ref"], self.output / f"{label}-overlay.jpg")
        return snapshot

    def action(self, tool: str, arguments: dict, label: str) -> dict:
        atomic_json(self.output / f"{label}-request.json", arguments)
        # Exactly one POST. A lost response must never cause an automatic retry.
        admission = self.request(f"/tools/{tool}:invoke", {
            "arguments": arguments, "caller_id": "rekep-baseline", "timeout_ms": 600000,
        })
        atomic_json(self.output / f"{label}-admission.json", admission)
        invocation = admission["invocation_id"]
        print(label, "admitted", invocation, admission["phase"], flush=True)
        previous, history = None, []
        deadline = time.monotonic() + 610
        while time.monotonic() < deadline:
            status = self.request(f"/invocations/{invocation}")
            phase = status["phase"]
            if phase != previous:
                history.append(status)
                atomic_json(self.output / f"{label}-status-history.json", history)
                print(label, phase, flush=True)
                previous = phase
            if phase not in {"dispatching", "accepted", "running", "cancelling"}:
                atomic_json(self.output / f"{label}-terminal.json", status)
                # Preserve the final endpoint response as well as the status resource.
                result = self.request(f"/invocations/{invocation}/result")
                atomic_json(self.output / f"{label}-result.json", result)
                return status
            time.sleep(1)  # First GET is immediate; this only bounds later polling.
        raise TimeoutError(f"unresolved invocation {invocation}; do not retry motion")


def preflight(program: dict, snapshot: dict, profile: Path) -> dict:
    import yaml
    from execution import solve_program
    from forge_msgs import Pose
    from forge_motion_server.config import load_config
    from forge_motion_server.kinematics import ForgeKinematicsAdapter

    validate_program(program, snapshot)
    binding = {k: program[k] for k in ("session_id", "scene_revision", "observation_id", "plan_id", "plan_digest")}
    verify_binding(program, **binding)
    config = yaml.safe_load((profile / "executor.yaml").read_text())
    solved = solve_program(program, snapshot, config, lambda _: None)
    mc = load_config(profile / "motion-server.yaml")
    ik = ForgeKinematicsAdapter(mc)
    positions = dict(zip(snapshot["robot_state"]["joint_names"], snapshot["robot_state"]["positions"], strict=True))
    seed = tuple(positions[name] for name in mc.groups["nova2_arm"].joint_names)
    evidence = []
    for stage in solved["stages"]:
        for segment in stage["segments"]:
            if segment["type"] != "move_pose":
                continue
            q = segment["target_pose"]["quaternion_xyzw"]
            # Same sequential seeding as motion_server: via poses, then target.
            points = [*segment.get("via_positions", []), segment["target_pose"]["position_m"]]
            for index, p in enumerate(points):
                outcome = ik.inverse("nova2_arm", Pose(x=p[0], y=p[1], z=p[2], qx=q[0], qy=q[1], qz=q[2], qw=q[3]), seed, mc.planning_timeout_ns, .0025, .02)
                label = segment["segment_id"] if index == len(points) - 1 else f"{segment['segment_id']}.via.{index}"
                if outcome.positions is None:
                    raise RuntimeError(f"IK_FAILED {label}: {outcome.status}: {outcome.message}")
                evidence.append({"segment_id": label, "joint_positions": outcome.positions,
                                 "max_joint_step_rad": max(abs(a-b) for a, b in zip(seed, outcome.positions, strict=True))})
                seed = outcome.positions
    return {"binding": binding, "solved": solved, "ik_evidence": evidence}


def placement_evidence(program: dict, before: dict, after: dict) -> dict:
    object_ids = {e["object_id"] for stage in program["stages"] for e in stage["events"] if e["type"] == "release"}
    regions = before["regions"]
    if len(object_ids) != 1 or len(regions) != 1:
        raise ValueError("baseline verifier requires one released object and one target region")
    obj = next(iter(object_ids))
    region = regions[0]
    points = [p for p in after["keypoints"] if p["object_id"] == obj]
    outside = {p["keypoint_id"]: max(max(lo-x, x-hi, 0.) for x, lo, hi in zip(
        p["position_m"], region["bounds_min_m"], region["bounds_max_m"], strict=True)) for p in points}
    passed = bool(points) and all(v <= 0.001 for v in outside.values())
    return {"passed": passed, "object_id": obj, "region_id": region["region_id"],
            "outside_distance_m": outside, "observation_id": after["observation_id"]}


def write_video(output: Path, recordings: Path, start_ns: int, end_ns: int) -> dict:
    """Cut this attempt's window out of the perception node's recorded video into video.mp4."""
    return cut_video(recordings, start_ns, end_ns, output / "video.mp4")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--gateway", default="http://127.0.0.1:19021")
    parser.add_argument("--profile", type=Path, default=ROOT / "profiles/sim-dobot-nova2-robotiq")
    parser.add_argument("--motion-source", type=Path, default=ROOT.parent / "operator-motion/packages/motion_server/src")
    parser.add_argument("--instruction", default="Pick up object_000 using its semantic object-center keypoint, lift it clear of obstacles, transport it, and place it inside region_000.")
    parser.add_argument("--allow-motion", action="store_true")
    parser.add_argument("--recordings", type=Path, default=runtime_root() / "run" / "rekep" / "recordings")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    sys.path.insert(0, str(args.motion_source.resolve()))
    gateway = Gateway(args.gateway, args.output)
    started_ns = time.time_ns()
    before, program, terminal = None, None, None
    motion_admitted = False
    try:
        for tool in ("rekep.get_context", "rekep.plan_task", "rekep.execute_task"):
            context = gateway.request(f"/tools/{tool}/context")
            atomic_json(args.output / f"{tool}-context.json", context)
            if not context["ready"]:
                raise RuntimeError(f"tool not ready: {tool}")
        before = gateway.snapshot("before")
        plan = gateway.action("rekep.plan_task", {
            "instruction": args.instruction, "session_id": before["session_id"], "observation_id": before["observation_id"],
        }, "plan")
        if plan["phase"] != "completed" or plan["result"]["status"] != "succeeded":
            raise RuntimeError(f"planning failed: {plan}")
        program = plan["result"]["outputs"]
        atomic_json(args.output / "program.json", program)
        checked = preflight(program, before, args.profile)
        atomic_json(args.output / "preflight.json", checked)
        print("preflight passed", len(checked["ik_evidence"]), "IK waypoints", flush=True)
        if not args.allow_motion:
            return 0
        motion_admitted = True  # Also covers a POST with a lost admission response.
        terminal = gateway.action("rekep.execute_task", {
            **checked["binding"], "allow_motion": True, "deadline_ms": 300000,
        }, "execute")
    except Exception as exc:
        atomic_json(args.output / "error.json", {"type": type(exc).__name__, "message": str(exc), "motion_post_may_have_been_sent": motion_admitted})
        print(type(exc).__name__, str(exc), flush=True)
        return 1
    finally:
        if motion_admitted:
            try:
                after = gateway.snapshot("after")
                evidence = placement_evidence(program, before, after)
                execution_ok = bool(terminal and terminal["phase"] == "completed" and terminal.get("result", {}).get("status") == "succeeded")
                evidence["execution_succeeded"] = execution_ok
                evidence["baseline_passed"] = execution_ok and evidence["passed"]
                atomic_json(args.output / "verification.json", evidence)
                print("verification", json.dumps(evidence), flush=True)
            except Exception as exc:
                atomic_json(args.output / "after-error.json", {"type": type(exc).__name__, "message": str(exc)})
                print("post-observation error", str(exc), flush=True)
        # Every attempt, including failed or motion-free ones, keeps a video.
        try:
            recording = write_video(args.output, args.recordings, started_ns, time.time_ns())
        except Exception as exc:  # the attempt's verdict never depends on the video
            recording = {"error": f"{type(exc).__name__}: {exc}"}
        atomic_json(args.output / "recording.json", recording)
        print("recording", json.dumps({k: v for k, v in recording.items() if k != "error"}), recording.get("error", ""), flush=True)
    verification = json.loads((args.output / "verification.json").read_text()) if (args.output / "verification.json").exists() else {}
    return 0 if verification.get("baseline_passed") else 1


if __name__ == "__main__":
    raise SystemExit(main())
