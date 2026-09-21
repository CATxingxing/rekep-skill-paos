"""P4 acceptance probe: positive lifecycle + negative contracts via the gateway."""

from __future__ import annotations

import json
import sys
import time

import gateway_probe as gp


def expect_failure(tool_id: str, arguments: dict, *, label: str) -> bool:
    """Admit an invocation that must NOT succeed; return True if it failed closed."""
    try:
        payload = gp.req(
            "POST",
            f"/tools/{tool_id}:invoke",
            {"arguments": arguments, "caller_id": "rekep-acceptance", "timeout_ms": 600000},
        )
    except Exception as exc:  # admission rejected
        print(f"NEG {label}: admission rejected ({type(exc).__name__})")
        return True
    data = payload.get("data", {})
    invocation = data.get("invocation_id")
    if not invocation:
        print(f"NEG {label}: rejected without invocation ({json.dumps(data)[:200]})")
        return True
    for _ in range(600):
        result = gp.req("GET", f"/invocations/{invocation}/result")
        status = result.get("data", {}).get("status")
        if status == "available":
            terminal = result["data"]["result"]
            ok = terminal.get("status") != "succeeded"
            err = terminal.get("error", {})
            print(f"NEG {label}: terminal={terminal.get('status')} code={err.get('code')} -> {'PASS' if ok else 'FAIL'}")
            return ok
        time.sleep(1)
    print(f"NEG {label}: timeout -> FAIL")
    return False


def main() -> int:
    checks: list[bool] = []

    tools = gp.list_tools()
    checks.append(set(tools) == {"rekep.get_context", "rekep.plan_task", "rekep.execute_task"})
    print("A/C tools:", tools)

    ctx = gp.query("rekep.perception", "get_context", {"refresh": True})
    perception = ctx["perception"]
    obs = perception["observation_id"]
    checks.append(obs.startswith("sha256:"))
    checks.append(len(perception["keypoints"]) >= 1)
    obj_labels = {k["object"] for k in perception["keypoints"]}
    checks.append("pick_cube" in obj_labels)
    print("C perception observation_id:", obs, "objects:", obj_labels)

    plan = gp.admit("rekep.plan_task", {"instruction": "pick the cube and place it in the target", "observation_id": obs})
    checks.append(plan["num_stages"] >= 1 and plan["plan_digest"].startswith("sha256:"))
    print("C plan:", {k: plan[k] for k in ("plan_id", "num_stages", "grasp_keypoints", "release_keypoints")})

    # negative: stale observation
    checks.append(expect_failure("rekep.plan_task", {"instruction": "x", "observation_id": "sha256:" + "0" * 64}, label="stale_observation"))

    # negative: execute with allow_motion=false
    checks.append(expect_failure("rekep.execute_task", {"plan_id": plan["plan_id"], "plan_digest": plan["plan_digest"], "allow_motion": False, "deadline_ms": 600000}, label="motion_not_allowed"))

    # negative: tampered plan digest
    checks.append(expect_failure("rekep.execute_task", {"plan_id": plan["plan_id"], "plan_digest": "sha256:" + "1" * 64, "allow_motion": True, "deadline_ms": 600000}, label="plan_tampered"))

    # negative: unknown plan id
    checks.append(expect_failure("rekep.execute_task", {"plan_id": "plan_missing", "plan_digest": "sha256:" + "2" * 64, "allow_motion": True, "deadline_ms": 600000}, label="plan_not_found"))

    # positive end-to-end
    result = gp.admit("rekep.execute_task", {"plan_id": plan["plan_id"], "plan_digest": plan["plan_digest"], "allow_motion": True, "deadline_ms": 1800000})
    ok = result.get("status") == "succeeded" and result.get("place_error_m", 1.0) < 0.06
    checks.append(ok)
    print("F execute:", {k: result.get(k) for k in ("status", "video", "place_error_m")})

    passed = all(checks)
    print("P4_CHECKS:", checks)
    print("P4_ACCEPTANCE:", "PASS" if passed else "FAIL")
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
