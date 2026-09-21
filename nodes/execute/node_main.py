"""Execution node: run the digest-bound ReKep plan on the MuJoCo simulator."""

from __future__ import annotations

import os

os.environ.setdefault("MUJOCO_GL", "egl")

from rekep_sim.env.mujoco_env import MujocoReKepEnv  # noqa: E402
from rekep_sim.provider import Provider, ProviderFailure  # noqa: E402
from rekep_sim.runtime import ReKepRuntime  # noqa: E402
from rekep_sim.state import digest, read_json, scene_path, state_dir  # noqa: E402


def main() -> None:
    holder: dict = {}

    def _runtime():
        # Heavy init (MuJoCo + solver warmup) is deferred so the dataflow starts
        # promptly and the gateway can bind before the runtime health deadline.
        if "runtime" not in holder:
            env = MujocoReKepEnv(
                scene_path(),
                height=int(os.environ.get("REKEP_SIM_HEIGHT", "240")),
                width=int(os.environ.get("REKEP_SIM_WIDTH", "320")),
            )
            holder["env"] = env
            holder["runtime"] = ReKepRuntime(env)
        return holder["env"], holder["runtime"]

    def handler(arguments, cancel, progress):
        if arguments.get("allow_motion") is not True:
            raise ProviderFailure("REKEP_MOTION_NOT_ALLOWED", "allow_motion must be true")
        plan_id = arguments.get("plan_id")
        expected = arguments.get("plan_digest")
        if not isinstance(plan_id, str) or not isinstance(expected, str):
            raise ProviderFailure("REKEP_INVALID_ARGUMENT", "plan_id and plan_digest are required")
        path = state_dir() / "plans" / f"{plan_id}.json"
        try:
            plan = read_json(path)
        except FileNotFoundError as exc:
            raise ProviderFailure("REKEP_PLAN_NOT_FOUND", str(exc)) from exc
        stored = plan.pop("plan_digest", None)
        actual = digest(plan)
        plan["plan_digest"] = stored
        if stored != expected or actual != expected:
            raise ProviderFailure("REKEP_PLAN_TAMPERED", "plan digest does not match")
        env, runtime = _runtime()
        env.reset()
        result = runtime.execute(plan, cancel=cancel, progress=progress, allow_motion=True)
        if result.get("status") == "cancelled":
            return result
        if result.get("status") != "succeeded":
            raise ProviderFailure(
                "REKEP_SIMULATION_FAILED", "task outcome check failed", details=result
            )
        return result

    Provider(
        endpoint_id="rekep.executor",
        operation="execute_task",
        semantics="session",
        handler=handler,
    ).run()


if __name__ == "__main__":
    main()
