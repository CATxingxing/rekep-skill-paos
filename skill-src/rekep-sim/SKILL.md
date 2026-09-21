---
name: rekep-sim
description: Use ReKep relational keypoint constraints to perceive, plan, and execute a pick-and-place task in the installed headless MuJoCo Dobot Nova2 profile.
metadata: {"PhyAgentOS":{"requires":{"runtime":["rekep-sim"]}}}
---

# ReKep MuJoCo pick & place

Use this Skill only with the active `sim` profile. It runs a headless MuJoCo
Dobot Nova2 + Robotiq 2F-85 scene and executes a constraint-driven ReKep task.

## Required order

1. Call `rekep.get_context` with `{}`. It returns the current perception
   snapshot: an `observation_id`, object labels, and keypoints (index,
   `position_m`, `pixel_uv`, `object`). Never invent object, keypoint,
   observation, frame, or region IDs.
2. Call `rekep.plan_task` with the user's `instruction` and the returned
   `observation_id`. Planning generates ReKep constraints and solves a plan; it
   never moves the simulator. Inspect the returned `plan_id`, `plan_digest`,
   `num_stages`, `grasp_keypoints`, and `release_keypoints`.
3. Call `rekep.execute_task` only when the user asked to run the simulation,
   passing `allow_motion: true`, the exact `plan_id` and `plan_digest`, and a
   positive `deadline_ms`.
4. Report the authoritative `status`, the result `video` path, and the place
   error. A completed invocation is not success unless its output says
   `status: succeeded`.

## Rules

- Do not output Python or ask the runtime to execute arbitrary code; constraints
  are generated and sandbox-executed inside the plan endpoint.
- Preserve the exact IDs returned by perception and planning.
- If a Tool reports stale perception, a digest mismatch, cancellation, or a
  failed outcome check, stop and surface the structured error. Do not
  automatically retry physical execution.

## Verification

The ToolResult of `rekep.execute_task` (`status`, `place_error_m`, `video`) is the
authoritative execution fact for this Skill. This gateway profile runs a headless
MuJoCo simulator and does **not** publish a separate image evidence stream.

When you create the AgentTask, use exactly:

```json
{"mode": "audit", "goal": "<user goal>",
 "success_criteria": ["<criteria>"],
 "evidence_policy": {"minimum_association": "best_effort"}}
```

Do **not** set `minimum_association: "authoritative"` and do not require an image
source this profile cannot produce: PAOS's Forge evidence adapter only emits
`best_effort`, so an `authoritative` policy fails closed even when the ToolResult
succeeded. Report the ToolResult; never fabricate success.


