#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import threading
from pathlib import Path

import yaml

from execution import ChildActionBridge, execute, solve_program
from rekep_core.contracts import ContractError, atomic_json, read_json, runtime_root, validate_program, validate_snapshot, verify_binding
from rekep_core.provider import Provider, ProviderFailure


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    arguments = parser.parse_args()
    config = yaml.safe_load(arguments.config.read_text(encoding="utf-8")) or {}
    snapshots: dict[str, dict] = {}
    programs: dict[str, dict] = {}
    cache_lock = threading.RLock()
    holder: dict[str, object] = {}

    def execute_task(call, cancel, progress):
        if call.get("allow_motion") is not True:
            raise ProviderFailure("REKEP_MOTION_NOT_ALLOWED", "allow_motion must be true")
        required = ("session_id", "observation_id", "plan_id", "plan_digest")
        if any(not isinstance(call.get(field), str) for field in required):
            raise ProviderFailure("REKEP_INVALID_ARGUMENT", ", ".join(required) + " are required")
        revision, deadline_ms = call.get("scene_revision"), call.get("deadline_ms")
        if isinstance(revision, bool) or not isinstance(revision, int) or revision < 0:
            raise ProviderFailure("REKEP_INVALID_ARGUMENT", "scene_revision must be a non-negative integer")
        if isinstance(deadline_ms, bool) or not isinstance(deadline_ms, int) or not 1 <= deadline_ms <= 600000:
            raise ProviderFailure("REKEP_INVALID_ARGUMENT", "deadline_ms must be in [1, 600000]")
        with cache_lock:
            snapshot = snapshots.get(call["observation_id"])
            program = programs.get(call["plan_id"])
            current = max(
                (item for item in snapshots.values() if item["session_id"] == call["session_id"]),
                key=lambda item: int(item["timestamp_ns"]),
                default=None,
            )
        try:
            if snapshot is None:
                snapshot = validate_snapshot(read_json(runtime_root() / "run" / "rekep" / "snapshots" / f"{call['observation_id'].removeprefix('sha256:')}.json"))
            if program is None:
                program = read_json(runtime_root() / "run" / "rekep" / "constraints" / f"{call['plan_id']}.json")
            validate_program(program, snapshot)
            verify_binding(program, session_id=call["session_id"], scene_revision=revision, observation_id=call["observation_id"], plan_id=call["plan_id"], plan_digest=call["plan_digest"])
            if current is None:
                raise ContractError("current observation stream is unavailable")
            if current["scene_revision"] != revision:
                raise ContractError("scene_revision is stale")
            solved = solve_program(program, snapshot, config, progress)
        except ContractError as exc:
            raise ProviderFailure("REKEP_STALE_OR_TAMPERED_PLAN", str(exc)) from exc
        active = runtime_root() / "run" / "rekep" / "active-execution.json"
        atomic_json(active, {"plan_id": program["plan_id"], "plan_digest": program["plan_digest"]})
        try:
            result = execute(program, solved, holder["bridge"], cancel, progress, deadline_ms=deadline_ms, config=config, snapshot=snapshot)
        finally:
            active.unlink(missing_ok=True)
        if result["status"] == "failed":
            raise ProviderFailure("REKEP_EXECUTION_FAILED", result["failure_reason"] or "execution failed", details=result)
        return result

    def on_input(input_id, value, metadata):
        holder["bridge"].handle_input(input_id, value, metadata)
        if input_id not in {"snapshot_in", "constraint_in"}:
            return
        try:
            raw = value[0].as_py() if hasattr(value, "__getitem__") else bytes(value).decode("utf-8")
            parsed = json.loads(raw)
            with cache_lock:
                if input_id == "snapshot_in":
                    snapshot = validate_snapshot(parsed)
                    snapshots[snapshot["observation_id"]] = snapshot
                    if len(snapshots) > 64:
                        snapshots.pop(next(iter(snapshots)))
                else:
                    programs[parsed["plan_id"]] = parsed
                    if len(programs) > 64:
                        programs.pop(next(iter(programs)))
        except (TypeError, ValueError, KeyError, UnicodeDecodeError, json.JSONDecodeError, ContractError):
            return

    provider = Provider(endpoint_id="rekep.executor", operation="execute_task", semantics="action", handler=execute_task, input_handler=on_input)
    holder["bridge"] = ChildActionBridge(provider.send_output)
    provider.run()


if __name__ == "__main__":
    main()
