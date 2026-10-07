#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from perception import Perception, decode_text
from rekep_core.provider import Provider, ProviderFailure


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    arguments = parser.parse_args()
    perception = Perception(arguments.config)
    holder: dict[str, Provider] = {}

    def emit(snapshot: dict) -> None:
        import pyarrow as pa
        holder["provider"].send_output("snapshot_out", pa.array([json.dumps(snapshot, ensure_ascii=False, separators=(",", ":"))], type=pa.string()))

    def query(_arguments, _cancel, _progress):
        try:
            snapshot = perception.capture()
        except RuntimeError as exc:
            raise ProviderFailure("REKEP_PERCEPTION_UNAVAILABLE", str(exc)) from exc
        emit(snapshot)
        return {"profile": "sim-dobot-nova2-robotiq", "readiness": perception.readiness(), "perception": snapshot}

    def on_input(input_id, value, _metadata):
        try:
            if input_id == "rgb":
                perception.update_rgb(value)
            elif input_id == "depth":
                perception.update_depth(value)
            elif input_id == "joint_state":
                perception.update_joint_state(value)
            elif input_id == "simulator_status":
                perception.update_status(json.loads(decode_text(value)))
            elif input_id == "observe_request":
                emit(perception.capture())
        except (RuntimeError, TypeError, ValueError, UnicodeDecodeError, json.JSONDecodeError):
            return

    provider = Provider(endpoint_id="rekep.perception", operation="get_context", semantics="query", handler=query, input_handler=on_input)
    holder["provider"] = provider
    provider.run()


if __name__ == "__main__":
    main()
