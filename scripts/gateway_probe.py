"""Probe the running rekep-sim Forge gateway: tools -> query -> action -> session."""

from __future__ import annotations

import json
import sys
import time
import urllib.request

BASE = sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:19021"
OPENER = urllib.request.build_opener(urllib.request.ProxyHandler({}))


def req(method: str, path: str, body=None, timeout: float = 1800.0):
    data = None if body is None else json.dumps(body).encode()
    request = urllib.request.Request(
        BASE + path, data=data, method=method, headers={"Content-Type": "application/json"}
    )
    with OPENER.open(request, timeout=timeout) as response:
        return json.loads(response.read())


def list_tools():
    payload = req("GET", "/tools")
    data = payload.get("data", {})
    tools = data.get("tools") or data.get("items") or []
    return [t.get("tool_id") for t in tools]


def query(endpoint: str, operation: str, arguments: dict) -> dict:
    payload = req(
        "POST",
        f"/tools/{endpoint}/{operation}:invoke",
        {"arguments": arguments, "caller_id": "rekep-acceptance", "timeout_ms": 1800000},
    )
    data = payload["data"]
    response = data.get("response")
    if isinstance(response, dict):
        result = response.get("result", {})
        if response.get("outcome") != "completed" or result.get("status") != "succeeded":
            raise RuntimeError(f"query failed: {json.dumps(response)[:500]}")
        return result.get("outputs", {})
    return data


def admit(tool_id: str, arguments: dict) -> dict:
    payload = req(
        "POST",
        f"/tools/{tool_id}:invoke",
        {"arguments": arguments, "caller_id": "rekep-acceptance", "timeout_ms": 1800000},
    )
    invocation = payload["data"]["invocation_id"]
    for _ in range(2400):
        result = req("GET", f"/invocations/{invocation}/result")
        status = result.get("data", {}).get("status")
        if status == "available":
            terminal = result["data"]["result"]
            if terminal.get("status") != "succeeded":
                raise RuntimeError(f"{tool_id} failed: {json.dumps(terminal)[:800]}")
            return terminal.get("outputs", {})
        time.sleep(1)
    raise TimeoutError(f"invocation timed out: {invocation}")


def main() -> int:
    tools = list_tools()
    print("TOOLS:", tools)
    assert "rekep.get_context" in tools and "rekep.plan_task" in tools and "rekep.execute_task" in tools

    ctx = query("rekep.perception", "get_context", {"refresh": True})
    perception = ctx["perception"]
    print("PERCEPTION observation_id:", perception["observation_id"])
    print("PERCEPTION keypoints:", json.dumps(perception["keypoints"])[:400])

    plan = admit(
        "rekep.plan_task",
        {
            "instruction": "pick the orange cube and place it in the green target zone",
            "observation_id": perception["observation_id"],
        },
    )
    print("PLAN:", json.dumps(plan))

    result = admit(
        "rekep.execute_task",
        {
            "plan_id": plan["plan_id"],
            "plan_digest": plan["plan_digest"],
            "allow_motion": True,
            "deadline_ms": 1800000,
        },
    )
    print("EXECUTE:", json.dumps({k: result.get(k) for k in ("status", "video", "place_error_m")}))
    assert result.get("status") == "succeeded"
    print("ACCEPTANCE_OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
