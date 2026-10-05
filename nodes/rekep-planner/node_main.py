#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import threading
from pathlib import Path

from constraint_schema import bind_program
from rekep_core.contracts import ContractError, atomic_json, runtime_root, validate_snapshot
from rekep_core.provider import Provider, ProviderFailure
from vlm_client import VLMError, generate


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    arguments = parser.parse_args()
    prompt = Path(__file__).with_name("prompt_template.txt")
    snapshots: dict[str, dict] = {}
    lock = threading.RLock()
    holder: dict[str, Provider] = {}

    def plan(call, _cancel, progress):
        instruction = call.get("instruction")
        observation_id = call.get("observation_id")
        session_id = call.get("session_id")
        if not isinstance(instruction, str) or not instruction.strip():
            raise ProviderFailure("REKEP_INVALID_ARGUMENT", "instruction must be a non-empty string")
        if not isinstance(observation_id, str) or not isinstance(session_id, str):
            raise ProviderFailure("REKEP_INVALID_ARGUMENT", "session_id and observation_id are required")
        with lock:
            snapshot = snapshots.get(observation_id)
        if snapshot is None:
            recovery = runtime_root() / "run" / "rekep" / "snapshots" / f"{observation_id.removeprefix('sha256:')}.json"
            try:
                snapshot = validate_snapshot(json.loads(recovery.read_text(encoding="utf-8")))
            except (OSError, json.JSONDecodeError, ContractError) as exc:
                raise ProviderFailure("REKEP_OBSERVATION_NOT_FOUND", "the exact observation is unavailable") from exc
        if snapshot["session_id"] != session_id:
            raise ProviderFailure("REKEP_STALE_OBSERVATION", "session_id does not match the observation")
        progress({"phase": "vlm_constraint_generation", "percent": 10})
        try:
            stages, provider_name, model = generate(instruction.strip(), snapshot, prompt)
            program = bind_program(stages, instruction=instruction.strip(), snapshot=snapshot, provider=provider_name, model=model)
        except VLMError as exc:
            raise ProviderFailure(exc.code, str(exc)) from exc
        except (ContractError, ValueError) as exc:
            raise ProviderFailure("REKEP_CONSTRAINT_REJECTED", str(exc)) from exc
        atomic_json(runtime_root() / "run" / "rekep" / "constraints" / f"{program['plan_id']}.json", program)
        import pyarrow as pa
        holder["provider"].send_output("constraint_out", pa.array([json.dumps(program, ensure_ascii=False, separators=(",", ":"))], type=pa.string()))
        progress({"phase": "constraint_program_ready", "percent": 100})
        return program

    def on_input(input_id, value, _metadata):
        if input_id != "snapshot_in":
            return
        try:
            raw = value[0].as_py() if hasattr(value, "__getitem__") else bytes(value).decode("utf-8")
            snapshot = validate_snapshot(json.loads(raw))
        except (TypeError, ValueError, UnicodeDecodeError, json.JSONDecodeError, ContractError):
            return
        with lock:
            snapshots[snapshot["observation_id"]] = snapshot
            if len(snapshots) > 32:
                snapshots.pop(next(iter(snapshots)))

    provider = Provider(endpoint_id="rekep.planner", operation="plan_task", semantics="action", handler=plan, input_handler=on_input)
    holder["provider"] = provider
    provider.run()


if __name__ == "__main__":
    main()
