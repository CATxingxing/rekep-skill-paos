# PAOS end-to-end acceptance (VLM planner, Franka Panda)

Scope: prove the `rekep-sim` Skill runs the **full PhyAgentOS flow**
(`paos agent -m ...` → Agent LLM → `get_context` → `plan_task`(VLM) → `execute_task`)
and reproduces the task-generalisation families, on the installed headless
MuJoCo **Franka Panda** scene.

Basis: `需求.md` (criterion), the PAOS Node/Forge reference manual (integration
protocol), and `baiyu858/PhyAgentOS-rekep-real-plugin` (algorithm/behaviour
blueprint: constraints, subgoal/path solvers, keypoints) — re-integrated on the
current protocol rather than the plugin's old HAL.

## Method

- Agent LLM: the configured `deepseek-flash` in `~/.PhyAgentOS/config.json`.
- ReKep VLM (constraint generation): `deepseek-v4-flash-vision-exp` via
  `REKEP_VLM_API_KEY=$XERA_API_KEY_1`, `REKEP_VLM_BASE_URL=https://newapi.x-era.com/v1`.
  The key is supplied through `~/.PhyAgentOS/run/rekep-sim/service.env` (loaded by
  the nodes at startup; never committed to the repo).
- Each run: `paos agent -m '<instruction>'` with an explicit request to use the
  **VLM planner** (`source="vlm"`), then execute, then verify the ToolResult and
  the produced video.

## Results (VLM planner through PAOS)

| # | Instruction | Task | Plan id / digest | Success spec | Status | Objective | Video |
|---|-------------|------|------------------|--------------|--------|-----------|-------|
| E1 | pick red cube → red region | `task_40315f5b91ce4e57` | `plan_9365ebb9…` / `sha256:cd8e7620…` | `in_region red_cube→region_red` | **succeeded** | ok, dx0.008 | `red_cube_to_red_region_vlm_plan_9365ebb9.mp4` |
| E2 | pick blue cube → blue region | `task_666f48ebc9d243ef` | `plan_630b8b42…` / `sha256:f9e1f1a0…` | `in_region blue_cube→region_blue` | **succeeded** | ok | `blue_cube_to_blue_region_vlm_plan_630b8b42.mp4` |
| E3 | stack red cube on blue cube | `task_21abe059b8f14128` | `plan_f23e6c72…` / `sha256:6a050dcf…` | `on_top red_cube←blue_cube` | **succeeded** | ok, dx0.0195 dy0.0132 top0.0399 | `red_on_blue_vlm_plan_f23e6c72.mp4` |
| E4 | stand pen upright in the holder | `task_2fadc394b2c6428d` | `plan_6ad90e9e…` / `sha256:89144b58…` | `upright pen in region_holder` | **succeeded** | ok, angle 6.75° off vertical | `pen_upright_in_holder_vlm_plan_6ad90e9e.mp4` |

Logs: `evidence/P5v/agent_e1d.log`, `agent_e2b.log`, `agent_e3b.log`, `agent_e4.log`.
Videos: `~/.PhyAgentOS/workspace/artifacts/rekep-sim/`.

Every run: `source="vlm"` (template never selected), constraints sandbox-executed,
AgentTask finalised terminal, no fabricated evidence.

## Frozen generality matrix (no API; executor + success evaluators)

`evidence/P5g/run_final.log` (`scripts/generality_experiments.py`), same executor
used by the nodes:

| GEN | Case | Result |
|-----|------|--------|
| GEN1 | red cube → region_red | succeeded (dx0.034) |
| GEN2 | blue cube → region_blue | succeeded (dx0.017) |
| GEN3 | stack red on blue | succeeded (on_top) |
| GEN4 | reorient pen upright | succeeded (10.47°) |
| GEN5 | no-op (tampered) constraints | **failed** (fail-closed) |
| GEN6 | objective mismatch | **failed** (fail-closed) |
| GEN7 | GEN1 repeat | succeeded (dx0.049) |

`tests`: 26 passed.

## Fixes applied this round

1. Perception node now persists `objects` (regions/movables) and reports
   `robot=franka_panda` / `gripper=franka_hand` — the VLM index guard and
   multi-object grounding now see the full scene.
2. `SKILL.md` rewritten for the Panda scene (objects, regions, instruction
   families, verification contract).
3. Task-level success derived from the instruction (`in_region` / `on_top` /
   `upright` / `moved`) instead of a weak "moved" check — PAOS results are
   meaningful and fail-closed.
4. IK robustness: yaw variants + small safe-approach fallback.
5. Assisted grasp now approaches the **intended** object (ignoring the
   real-robot Cartesian standoff the VLM emits), fixing stack/reorient grasp
   failures without changing which object is grasped.
6. Secrets kept out of the bundle via an external `service.env` loaded at node
   startup (Dora does not forward the launching shell's env to nodes).
