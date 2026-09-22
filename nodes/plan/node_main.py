"""Planning node: VLM (or deterministic template) ReKep constraint generation.

No simulator is loaded here; it consumes the perception snapshot and produces a
digest-bound execution plan. This mirrors the reference constraint_generation
step plus the plan contract of the Skill.
"""

from __future__ import annotations

import os

from rekep_sim.provider import Provider, ProviderFailure  # noqa: E402
from rekep_sim.runtime import ReKepRuntime  # noqa: E402
from rekep_sim.state import load_service_env, read_json, state_dir, write_json  # noqa: E402


def main() -> None:
    load_service_env()
    planner = ReKepRuntime(env=None)

    def handler(arguments, _cancel, _progress):
        instruction = arguments.get("instruction")
        if not isinstance(instruction, str) or not instruction.strip():
            raise ProviderFailure("REKEP_INVALID_ARGUMENT", "instruction must be a non-empty string")
        sd = state_dir()
        try:
            snapshot = read_json(sd / "latest_perception.json")
        except FileNotFoundError as exc:
            raise ProviderFailure(
                "REKEP_NO_PERCEPTION", "call rekep.perception.get_context first"
            ) from exc
        observation_id = arguments.get("observation_id")
        if observation_id and observation_id != snapshot.get("observation_id"):
            raise ProviderFailure(
                "REKEP_OBSERVATION_STALE",
                "observation_id does not match the latest perception snapshot",
            )
        source = arguments.get("source") or os.environ.get("REKEP_PLAN_SOURCE")
        if not source:
            # use the reference VLM path automatically once a key is configured
            source = "vlm" if os.environ.get("REKEP_VLM_API_KEY") else "template"
        program = planner.plan(instruction, snapshot, source=source)
        plans = sd / "plans"
        plans.mkdir(parents=True, exist_ok=True)
        write_json(plans / f"{program['plan_id']}.json", program)
        write_json(sd / "latest_plan.json", program)
        return {
            "plan_id": program["plan_id"],
            "plan_digest": program["plan_digest"],
            "instruction": instruction,
            "observation_id": program["observation_id"],
            "num_stages": program["num_stages"],
            "grasp_keypoints": program["grasp_keypoints"],
            "release_keypoints": program["release_keypoints"],
            "source": program["source"],
        }

    Provider(
        endpoint_id="rekep.plan",
        operation="plan_task",
        semantics="action",
        handler=handler,
    ).run()


if __name__ == "__main__":
    main()
