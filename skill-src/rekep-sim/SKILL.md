---
name: rekep-sim
description: Use ReKep relational keypoint constraints to perceive, plan, and execute manipulation tasks in the installed headless MuJoCo Franka Panda profile.
metadata: {"PhyAgentOS":{"requires":{"runtime":["rekep-sim"]}}}
---

# ReKep MuJoCo (Franka Panda) manipulation

Use this Skill only with the active `sim` profile. It runs a headless MuJoCo
**Franka Panda (7-DOF)** scene with several objects (a red cube, a blue cube, a
thin pen), an open container (holder) and several target regions. Tasks are
expressed as free-form natural-language instructions.

## Required order

1. Call `rekep.get_context` with `{}`. It returns the current perception
   snapshot: an `observation_id`, an `objects` list (each with a `name`, whether
   it is `movable` or a `region`, and its `position_m`), and `keypoints` (each with
   `index`, `object`, `kind`, `position_m`, `pixel_uv`). Never invent object,
   keypoint, observation, frame, or region IDs.
2. Call `rekep.plan_task` with the user's `instruction` and the returned
   `observation_id`. The plan endpoint asks a vision-language model to write ReKep
   constraints from the instruction and snapshot, sandbox-executes them, and solves
   a plan. It never moves the simulator. Inspect `plan_id`, `plan_digest`,
   `num_stages`, `grasp_keypoints`, `release_keypoints`, and `source`.
3. Call `rekep.execute_task` only when the user asked to run the simulation, with
   `allow_motion: true`, the exact `plan_id` and `plan_digest`, and a positive
   `deadline_ms`. Reconcile status/result until terminal; an accepted invocation
   is not completion.
4. Report the authoritative `status`, the objective check, and the result `video`
   path. A completed invocation is not success unless its output says
   `status: succeeded`.

## Task generalisation

Support at least these instruction families on the installed scene:
- pick a named/coloured object and place it in a named/coloured region;
- stack one object on top of another;
- reorient an elongated object (e.g. stand the pen upright) and/or insert it into
  the container.
Choose object/region/keypoint IDs only from the live perception snapshot, and let
the plan endpoint derive the stages. Prefer the object's centre keypoint
(`kind: center`) for placement targets and its `axis_min`/`axis_max` keypoints for
orientation.

## Rules

- Do not output Python or ask the runtime to execute arbitrary code; constraints
  are generated and sandbox-executed inside the plan endpoint.
- Preserve the exact IDs returned by perception and planning.
- Use the VLM planner by default (`source: "vlm"`); only pass
  `source: "template"` if the VLM planner genuinely fails.
- If a Tool reports stale perception, a digest mismatch, cancellation, or a failed
  outcome check, stop and surface the structured error. Do not automatically retry
  physical execution.

## Verification

The ToolResult of `rekep.execute_task` (`status`, `objectives`, `video`) is the
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
