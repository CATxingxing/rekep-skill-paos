---
name: rekep
description: Perceive, plan, and execute observation-bound relational manipulation with the installed ReKep profile. Use for supported simulation or hardware profiles; simulator authorization never authorizes real hardware motion.
metadata:
  PhyAgentOS:
    requires:
      runtime: [rekep]
---

# ReKep manipulation

1. Call `rekep.get_context`. Preserve the returned `session_id`, `scene_revision`, and `observation_id`; use only object, region, and keypoint IDs present in that snapshot.
2. Call `rekep.plan_task` with the original instruction and exact `session_id` and `observation_id`. Planning uses the VLM configured for the current PAOS Agent and produces a restricted JSON constraint program. It does not move the robot. If the provider is unsupported or unavailable, stop and report the error; do not substitute a template or another model.
3. Inspect the stages and constraint program. Call `rekep.execute_task` only when the user has authorized motion for the active profile. Pass the exact `session_id`, `scene_revision`, `observation_id`, `plan_id`, and `plan_digest`, plus `allow_motion: true` and a positive deadline.
4. Treat only terminal `status: succeeded` as success. On stale identity, digest mismatch, invalid constraints, solver/IK/collision failure, timeout, cancellation, or failed post-stage verification, stop and surface the structured failure. Never automatically retry a physical segment.

The executor sends one segment at a time and waits for its terminal child result before continuing. Never invent identifiers, execute generated Python, change profiles silently, or treat an accepted invocation as completion.
