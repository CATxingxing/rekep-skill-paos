# sim_tasks — RoboDojo-inspired MuJoCo evaluation scenes

Three manipulation scenes on the Dobot Nova2 + Robotiq 2F-85 MuJoCo model of this repo, each with a
programmatic evaluator.  They are **RoboDojo-inspired**, not the official RoboDojo benchmark; scores are
not comparable with the RoboDojo leaderboard.

| task | instruction (example) | success |
|---|---|---|
| `general_pickup` | "Pick up the blue block and lift it off the table." | target z − initial z ≥ 0.10 m, held 0.7 s |
| `stack_blocks`   | "Stack the three cubes into one tower on the green square: the yellow cube at the bottom, …" | 2 correct, stable pairs; base on the zone; gripper released; survives a 1.5 s hands-off probe |
| `push_t`         | "Push the blue T-shaped block onto the green T-shaped outline …" | xy ≤ 2 cm, yaw ≤ 0.2 rad, never lifted (≤ 1 cm), at rest |

All thresholds, randomization ranges and physics numbers are in `config/tasks.yaml`.

## Layout

```
sim_tasks/config/tasks.yaml   every threshold / range / physics parameter
sim_tasks/scene.py            builds MJCF from the baseline Nova2+2F-85 scene (baseline file is only read)
sim_tasks/env.py              TaskEnv: reset(seed), step, observe, evaluate, is_success, get_metrics, probe
sim_tasks/evaluator.py        generic layer: EvalResult, FailureReason, is_inside_region, orientation_error, probe_stats
sim_tasks/tasks/*.py          per task: scene sampling + Evaluator class + instruction
sim_tasks/oracle.py, demos.py scripted ground-truth-aware drivers (TEST TOOLS ONLY, never a policy)
sim_tasks/run.py              CLI
sim_tasks/make_profiles.py    writes profiles/sim-dobot-nova2-robotiq-<task>/ and sim_tasks/generated/<task>.xml
tests/test_sim_tasks.py       scene / physics / evaluator tests
```

## Use

```bash
conda activate paoswx                 # has mujoco 3.13; the repo .venv does not
export MUJOCO_GL=egl
python -m sim_tasks.run --task stack_blocks --seed 42 --out /tmp/out --oracle --video
pytest tests/test_sim_tasks.py
```

```python
from sim_tasks import make_env
env = make_env("stack_blocks")
obs = env.reset(seed=42)              # policy-facing: rgb, depth, joint_names/positions, sim_time, instruction
env.get_instruction()
env.step(arm_ctrl=q, gripper_ctrl=255.0, ticks=1)   # 50 Hz control ticks; position targets, gripper 0..255
env.evaluate()   # {"success", "partial", "score", "metrics", "failure_reason"}  (ground truth, evaluator only)
env.is_success(); env.get_metrics()
```

`evaluate(final=True)` (default) rolls a **copy** of the simulation forward with the arm held still for the
task's stability probe; the live simulation is not advanced.  `evaluate(final=False)` skips the probe, so
stack / push-T cannot report success without it.

## Ground truth vs. policy

`observe()` returns only RGB, depth, joint state, sim time and the instruction.  Evaluators read
`SceneState` (body poses, velocities, contacts) built from MuJoCo.  Do not pass `scene_info()`, `plan`,
`snapshot()` or evaluator output to ReKep.

## Failure reasons

`WRONG_OBJECT, GRASP_FAILED, OBJECT_DROPPED, PLACEMENT_ERROR, ORIENTATION_ERROR, STACK_UNSTABLE,
NOT_RELEASED, PUSH_LIFT_VIOLATION, INSERTION_FAILED (unused so far), SEQUENCE_ERROR, TIMEOUT, UNKNOWN`.
Specific physical causes win over `TIMEOUT`, which wins over the generic geometric reason.

## Known limits

* The Dora `mujoco_sim` node does not export object poses, so an evaluator cannot read the live PAOS
  simulation yet; today the evaluators run inside this pure-MuJoCo harness (same MJCF, same robot).
* Perception requires one green region per scene: `general_pickup` has a small green reference mat
  (a distractor); `stack_blocks` uses the green stack zone; `push_t` uses the green T outline.
  The perception keypoint offsets in the task profiles are the baseline's cube calibration.
* `oracle_push_t` only solves the aligned scenario `demos.push_t_scenario()`; random Push-T starts need
  rotation, which the scripted driver does not implement.
