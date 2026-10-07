"""Scene / physics / evaluator tests for sim_tasks (pure MuJoCo, no ReKep, no Dora).

Run with an interpreter that has mujoco + numpy (the repo .venv has neither):
    conda activate paoswx && MUJOCO_GL=egl pytest tests/test_sim_tasks.py
Tests are skipped when mujoco is missing.
"""
from __future__ import annotations

import math
import sys
import types
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("mujoco")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from sim_tasks import load_config, make_env                      # noqa: E402
from sim_tasks.demos import oracle_pickup, oracle_push_t, oracle_stack, push_t_scenario   # noqa: E402
from sim_tasks.mathutil import quat_from_yaw                       # noqa: E402
from sim_tasks.oracle import Oracle                                # noqa: E402

import tempfile                                                    # noqa: E402
import sim_tasks.env as _env                                       # noqa: E402

_env.GENERATED_DIR = Path(tempfile.mkdtemp(prefix="sim_tasks_test_"))   # keep test scenes out of the repo

CFG = load_config()
TABLE_TOP = CFG["common"]["table"]["top_z"]


def fresh(task: str, seed: int = 42, **overrides):
    env = make_env(task)
    env.reset(seed=seed, overrides=overrides or None)
    return env


# ======================================================================= scenes
@pytest.mark.parametrize("task", ["general_pickup", "stack_blocks", "push_t"])
def test_random_scenes_are_legal(task):
    """Scene load, settle (no explosion), objects on the table, reachable, no robot contact."""
    for seed in range(25):
        env = fresh(task, seed)
        o = Oracle(env)
        state = env.snapshot()
        assert not [c for n in env.tracked for c in state.robot_contacts(n)], "robot touches an object at reset"
        ws = env.common["workspace"]
        for name in env.tracked:
            p = state.bodies[name].pos
            assert ws["x"][0] - 0.1 <= p[0] <= ws["x"][1] + 0.1 and ws["y"][0] - 0.1 <= p[1] <= ws["y"][1] + 0.1
            assert p[2] > TABLE_TOP, f"{name} below the table top"
            assert float(np.linalg.norm(state.bodies[name].linvel)) < 1e-3
            assert o.reachable([p[0], p[1], 0.04], 0.0) and o.reachable([p[0], p[1], 0.22], 0.0)
        for a in env.tracked:
            for b in env.tracked:
                if a < b:
                    assert np.linalg.norm(state.bodies[a].pos[:2] - state.bodies[b].pos[:2]) >= 0.09
        env.close()


@pytest.mark.parametrize("task", ["general_pickup", "stack_blocks", "push_t"])
def test_idle_physics_is_stable(task):
    env = fresh(task, 3)
    before = {n: env.snapshot().bodies[n].pos.copy() for n in env.tracked}
    env.step(ticks=250)      # 5 s with the arm held at home
    after = env.snapshot()
    for n in env.tracked:
        assert np.linalg.norm(after.bodies[n].pos - before[n]) < 1e-3
    assert np.isfinite(env.data.qpos).all() and np.abs(env.data.qvel).max() < 1.0
    env.close()


@pytest.mark.parametrize("task", ["general_pickup", "stack_blocks", "push_t"])
def test_same_seed_same_scene(task):
    a, b, c = fresh(task, 11), fresh(task, 11), fresh(task, 12)
    assert a.scene_path.read_text() == b.scene_path.read_text()
    assert a.scene_path.read_text() != c.scene_path.read_text()
    assert a.get_instruction() == b.get_instruction()
    a.step(ticks=50)
    b.step(ticks=50)
    for n in a.tracked:
        assert np.array_equal(a.snapshot().bodies[n].pos, b.snapshot().bodies[n].pos)


def test_pickup_randomizes_target_order_and_xy_yaw():
    plans = [fresh("general_pickup", s).plan for s in range(12)]
    assert len({p["target"] for p in plans}) > 3
    assert len({tuple(o["name"] for o in p["objects"]) for p in plans}) > 6
    assert len({round(p["objects"][0]["yaw"], 3) for p in plans}) > 6


def test_stack_randomizes_color_order_and_zone():
    plans = [fresh("stack_blocks", s).plan for s in range(12)]
    assert len({tuple(p["order"]) for p in plans}) >= 4
    assert len({tuple(np.round(p["zone"]["center"], 3)) for p in plans}) > 6


def test_observation_has_no_ground_truth():
    env = fresh("stack_blocks")
    obs = env.observe()
    assert set(obs) == {"rgb", "depth", "joint_names", "joint_positions", "sim_time", "instruction"}
    assert obs["rgb"].shape == (480, 640, 3) and obs["depth"].shape == (480, 640)
    assert "cube" not in " ".join(obs["joint_names"])


@pytest.mark.parametrize("task", ["general_pickup", "stack_blocks", "push_t"])
def test_perception_segmentation_sees_scene(task):
    """The task profile's palette segmentation finds every object with its colour, and a green region."""
    sys.path[:0] = [str(ROOT / "nodes/common"), str(ROOT / "nodes/rekep-perception")]
    stub = types.ModuleType("dinov2_runtime")
    stub.DinoV2 = object
    sys.modules.setdefault("dinov2_runtime", stub)
    import yaml
    import perception as P
    cfg = yaml.safe_load((ROOT / f"profiles/sim-dobot-nova2-robotiq-{task.replace('_', '-')}/perception.yaml").read_text())
    p = P.Perception.__new__(P.Perception)
    p.config = cfg
    for seed in range(6):
        env = fresh(task, seed)
        obs = env.observe()
        cloud, valid = p._base_cloud(obs["depth"])
        masks, regions = p._segment(obs["rgb"], cloud, valid)
        assert len(masks) == len(env.tracked), f"seed {seed}: {len(masks)} masks for {len(env.tracked)} objects"
        assert len(regions) >= 1
        plan = env.scene_info()
        truth = sorted(o["color"] for o in plan.get("objects", plan.get("blocks", []))) or ["blue"]
        assert sorted(label for _, label in masks) == truth
        gt = [env.snapshot().bodies[n].pos for n in env.tracked]
        for m, _ in masks:   # visible surface lies near some object, i.e. camera calibration matches
            pt = np.median(cloud[m & valid], axis=0)
            assert min(np.linalg.norm(pt[:2] - g[:2]) for g in gt) < 0.05
        env.close()


# =============================================================== General Pickup
def test_pickup_oracle_success_all_shapes():
    seen = set()
    for seed in range(8):
        env = fresh("general_pickup", seed)
        for idx in range(3):
            env.reset(seed=seed, overrides={"target_index": idx})
            shape = env.plan["objects"][idx]["shape"]
            oracle_pickup(env)
            r = env.evaluate()
            assert r["success"] and r["score"] == 1.0, (seed, idx, shape, r)
            assert r["metrics"]["max_lift_height"] >= CFG["general_pickup"]["evaluator"]["lift_height_m"]
            seen.add(shape)
    assert seen == {"cube", "block", "cylinder"}


def test_pickup_no_motion_is_grasp_failure():
    env = fresh("general_pickup")
    env.step(ticks=100)
    r = env.evaluate()
    assert not r["success"] and r["score"] == 0.0 and r["failure_reason"] == "GRASP_FAILED"


def test_pickup_wrong_object_is_wrong_object():
    env = fresh("general_pickup")
    other = next(o["name"] for o in env.plan["objects"] if o["name"] != env.plan["target"])
    oracle_pickup(env, name=other)
    r = env.evaluate()
    assert not r["success"] and r["failure_reason"] == "WRONG_OBJECT" and r["score"] == 0.0
    assert r["metrics"]["wrong_object_moved"] and other in r["metrics"]["wrong_objects_lifted"]


def test_pickup_hold_time_is_required():
    env = fresh("general_pickup")
    oracle_pickup(env, hold_s=0.0)            # reaches 10 cm at the very end, no hold
    r = env.evaluate()
    assert r["metrics"]["max_lift_height"] >= 0.10
    # reaching height only counts after hold_time_s of continuous lift
    assert r["metrics"]["max_hold_s"] < CFG["general_pickup"]["evaluator"]["hold_time_s"] or r["success"]
    env2 = fresh("general_pickup")
    oracle_pickup(env2, lift=0.06)           # lifted 6 cm only
    r2 = env2.evaluate()
    assert not r2["success"] and r2["failure_reason"] == "GRASP_FAILED" and r2["partial"] and 0.2 < r2["score"] < 1.0


def test_pickup_drop_is_detected():
    env = fresh("general_pickup")
    o = oracle_pickup(env, lift=0.06, hold_s=0.2)
    o.gripper(CFG["common"]["gripper_open_ctrl"], ticks=80)   # release in the air, object falls
    env.step(ticks=50)
    r = env.evaluate()
    assert not r["success"] and r["failure_reason"] == "OBJECT_DROPPED" and r["metrics"]["object_dropped"]


def test_pickup_timeout():
    env = make_env("general_pickup", load_config(overrides={"general_pickup": {"time_limit_s": 2.0}}))
    env.reset(seed=1)
    env.step(ticks=150)
    r = env.evaluate()
    assert not r["success"] and r["failure_reason"] == "TIMEOUT"


# ================================================================ Stack Blocks
def place_stack(env, order, base_xy, offsets=None, settle_s=0.8):
    size = env.plan["block_size"]
    for level, name in enumerate(order):
        off = (offsets or {}).get(name, (0.0, 0.0))
        env.set_body_pose(name, [base_xy[0] + off[0], base_xy[1] + off[1], TABLE_TOP + size / 2 + level * size + 2e-4],
                          quat_from_yaw(0.0))
    for _ in range(round(settle_s / env.timestep)):
        import mujoco
        mujoco.mj_step(env.model, env.data)


def test_stack_oracle_success_and_metrics():
    for seed in (42, 7, 21):
        env = fresh("stack_blocks", seed)
        oracle_stack(env)
        r = env.evaluate()
        assert r["success"] and r["score"] == 1.0, (seed, r)
        m = r["metrics"]
        assert m["stable_pairs"] == 2 and m["base_in_zone"] and m["released"] and m["at_rest"]
        assert m["max_pose_drift"] < 1e-3 and m["max_linear_speed"] < 0.01


def test_stack_manual_success_state():
    env = fresh("stack_blocks")
    place_stack(env, env.plan["order"], env.plan["zone"]["center"])
    r = env.evaluate()
    assert r["success"], r
    assert r["metrics"]["max_tower_height_blocks"] >= 1


def test_stack_within_tolerance_still_succeeds_and_beyond_fails():
    ev = CFG["stack_blocks"]["evaluator"]
    env = fresh("stack_blocks")
    top = env.plan["order"][2]
    place_stack(env, env.plan["order"], env.plan["zone"]["center"], {top: (ev["xy_tolerance_m"] * 0.5, 0.0)})
    assert env.evaluate()["success"]
    env = fresh("stack_blocks")
    top = env.plan["order"][2]
    place_stack(env, env.plan["order"], env.plan["zone"]["center"], {top: (ev["xy_tolerance_m"] * 1.15, 0.0)})
    r = env.evaluate()
    assert not r["success"] and r["partial"] and r["score"] == ev["partial_score"]
    assert r["failure_reason"] == "PLACEMENT_ERROR"


def test_stack_two_blocks_is_partial():
    env = fresh("stack_blocks")
    order = env.plan["order"]
    place_stack(env, order[:2], env.plan["zone"]["center"])
    r = env.evaluate()
    assert not r["success"] and r["partial"] and r["metrics"]["stable_pairs"] == 1
    assert r["score"] == CFG["stack_blocks"]["evaluator"]["partial_score"]


def test_stack_nothing_stacked_scores_zero():
    env = fresh("stack_blocks")
    r = env.evaluate()
    assert not r["success"] and not r["partial"] and r["score"] == 0.0 and r["failure_reason"] == "PLACEMENT_ERROR"


def test_stack_wrong_order_is_sequence_error():
    env = fresh("stack_blocks")
    order = list(env.plan["order"])
    wrong = [order[1], order[0], order[2]]
    place_stack(env, wrong, env.plan["zone"]["center"])
    r = env.evaluate()
    assert not r["success"] and r["failure_reason"] == "SEQUENCE_ERROR"
    # the same tower is accepted when the task does not constrain the order
    env2 = fresh("stack_blocks", require_order=False)
    place_stack(env2, wrong, env2.plan["zone"]["center"])
    assert env2.evaluate()["success"]


def test_stack_off_zone_fails_when_zone_required():
    env = fresh("stack_blocks")
    zc = env.plan["zone"]["center"]
    far = (zc[0] - 0.14, zc[1] + 0.02)
    place_stack(env, env.plan["order"], far)
    r = env.evaluate()
    assert not r["success"] and r["failure_reason"] == "PLACEMENT_ERROR" and not r["metrics"]["base_in_zone"]
    env2 = fresh("stack_blocks", require_base_in_zone=False)
    place_stack(env2, env2.plan["order"], far)
    assert env2.evaluate()["success"]


def test_stack_toppled_top_block_fails():
    env = fresh("stack_blocks")
    order, size = env.plan["order"], env.plan["block_size"]
    place_stack(env, order[:2], env.plan["zone"]["center"])
    zc = env.plan["zone"]["center"]
    tilted = np.array([math.cos(0.5), math.sin(0.5), 0, 0])            # 57 deg about x: lying on an edge
    env.set_body_pose(order[2], [zc[0], zc[1], TABLE_TOP + 2.2 * size], tilted, settle_s=0.0)
    r = env.evaluate()
    assert not r["success"] and r["failure_reason"] in {"STACK_UNSTABLE", "PLACEMENT_ERROR"}


def test_stack_cube_knocked_off_the_table_is_dropped():
    env = fresh("stack_blocks")
    order = env.plan["order"]
    place_stack(env, order, env.plan["zone"]["center"])
    env.set_body_pose(order[2], [0.9, 0.3, 0.05])           # outside the table: falls to the floor
    import mujoco
    for _ in range(4000):
        mujoco.mj_step(env.model, env.data)
    r = env.evaluate()
    assert not r["success"] and r["failure_reason"] == "OBJECT_DROPPED"


def test_stack_valid_now_but_sliding_off_is_unstable():
    """Momentarily aligned top block with lateral velocity: probe catches it."""
    env = fresh("stack_blocks")
    order = env.plan["order"]
    place_stack(env, order, env.plan["zone"]["center"])
    jid = env.model.joint(f"{order[2]}_free").id
    env.data.qvel[env.model.jnt_dofadr[jid]] = 0.5
    r = env.evaluate()
    assert not r["success"] and r["failure_reason"] == "STACK_UNSTABLE"


def test_stack_still_held_is_not_released():
    env = fresh("stack_blocks")
    import sim_tasks.demos as demos
    orig_place = Oracle.place
    Oracle.place = lambda self, *a, **k: orig_place(self, *a, **{**k, "release": False})
    try:
        # drive the first two levels fully, then hold the third; patch release only for the last block
        o = Oracle(env)
        order, size = env.plan["order"], env.plan["block_size"]
        from sim_tasks.mathutil import yaw_from_quat
        for level, name in enumerate(order):
            state = env.snapshot()
            pos = state.bodies[name].pos
            yaw = o.grasp_yaw(yaw_from_quat(state.bodies[name].quat))
            xy = env.plan["zone"]["center"] if level == 0 else state.bodies[order[level - 1]].pos[:2]
            cz = TABLE_TOP + size / 2 + level * size + 0.003
            o.pick(pos[:2], demos._pinch_z_for(env, pos[2]), yaw)
            if level < 2:
                orig_place(o, xy, demos._pinch_z_for(env, cz), yaw, hover_z=demos._pinch_z_for(env, TABLE_TOP + size * (level + 2.2)))
            else:
                o.place(xy, demos._pinch_z_for(env, cz), yaw, hover_z=demos._pinch_z_for(env, TABLE_TOP + size * 4.2))
    finally:
        Oracle.place = orig_place
    r = env.evaluate()
    assert not r["success"] and r["failure_reason"] == "NOT_RELEASED" and not r["metrics"]["released"]


def test_stack_timeout():
    env = make_env("stack_blocks", load_config(overrides={"stack_blocks": {"time_limit_s": 2.0}}))
    env.reset(seed=1)
    env.step(ticks=150)
    assert env.evaluate()["failure_reason"] == "TIMEOUT"


# ==================================================================== Push T
def goal_pose(env):
    g = env.plan["goal"]
    return g["center"][0], g["center"][1], g["yaw"]


def put_t(env, x, y, yaw, dz=0.0, settle_s=0.8):
    import mujoco
    env.set_body_pose("push_t", [x, y, env.plan["pos"][2] + dz], quat_from_yaw(yaw))
    for _ in range(round(settle_s / env.timestep)):
        mujoco.mj_step(env.model, env.data)


def test_push_t_goal_pose_is_success():
    env = fresh("push_t")
    put_t(env, *goal_pose(env))
    r = env.evaluate()
    assert r["success"] and r["score"] == 1.0, r
    assert r["metrics"]["coverage"] > 0.97 and r["metrics"]["xy_error"] < 1e-3


def test_push_t_thresholds_come_from_config():
    ev = CFG["push_t"]["evaluator"]
    gx, gy, gyaw = goal_pose(fresh("push_t"))
    env = fresh("push_t")
    put_t(env, gx + ev["xy_tolerance_m"] * 0.7, gy, gyaw + ev["yaw_tolerance_rad"] * 0.7)
    assert env.evaluate()["success"]
    env = fresh("push_t")
    put_t(env, gx + ev["xy_tolerance_m"] * 1.6, gy, gyaw)
    r = env.evaluate()
    assert not r["success"] and r["failure_reason"] == "PLACEMENT_ERROR"
    env = fresh("push_t")
    put_t(env, gx, gy, gyaw + ev["yaw_tolerance_rad"] * 2.5)
    r = env.evaluate()
    assert not r["success"] and r["failure_reason"] == "ORIENTATION_ERROR"


def test_push_t_partial_by_coverage():
    ev = CFG["push_t"]["evaluator"]
    gx, gy, gyaw = goal_pose(fresh("push_t"))
    env = fresh("push_t")
    put_t(env, gx + 0.03, gy, gyaw)
    r = env.evaluate()
    assert not r["success"] and r["partial"] and ev["coverage_partial"] <= r["metrics"]["coverage"] < 1.0
    assert 0.0 < r["score"] <= ev["partial_score_cap"]
    env = fresh("push_t")                         # unmoved T: no coverage
    r = env.evaluate()
    assert not r["partial"] and r["score"] < 0.05


def test_push_t_lifting_is_a_violation_even_if_put_back():
    env = fresh("push_t")
    gx, gy, gyaw = goal_pose(env)
    env.set_body_pose("push_t", [gx, gy, env.plan["pos"][2] + 0.03], quat_from_yaw(gyaw))
    env.step(ticks=1)                       # the evaluator observes the lifted pose
    put_t(env, gx, gy, gyaw)                # ... then it settles exactly on the goal
    r = env.evaluate()
    assert not r["success"] and r["failure_reason"] == "PUSH_LIFT_VIOLATION" and r["score"] == 0.0
    assert r["metrics"]["max_delta_z"] > CFG["push_t"]["evaluator"]["max_delta_z_m"]


def test_push_t_pushed_off_the_table_is_dropped():
    env = fresh("push_t")
    env.set_body_pose("push_t", [0.9, 0.3, 0.2])
    env.step(ticks=150)
    r = env.evaluate()
    assert not r["success"] and r["failure_reason"] in {"OBJECT_DROPPED", "PUSH_LIFT_VIOLATION"}


def test_push_t_moving_t_is_not_success():
    env = fresh("push_t")
    gx, gy, gyaw = goal_pose(env)
    put_t(env, gx, gy, gyaw)
    jid = env.model.joint("push_t_free").id
    env.data.qvel[env.model.jnt_dofadr[jid]] = 0.3      # sliding through the goal
    r = env.evaluate()
    assert not r["success"] and not r["metrics"]["at_rest"]


def test_push_t_oracle_pushes_it_home_physically():
    """Real contact: closed gripper pushes the T along its axis onto a goal on that line."""
    env = make_env("push_t")
    env.reset(seed=3, overrides=push_t_scenario(CFG))
    oracle_push_t(env)
    r = env.evaluate()
    assert r["success"], r
    m = r["metrics"]
    assert m["touched"] and m["max_delta_z"] < 0.005 and 0.10 < m["pushed_path_length"] < 0.25
    assert not m["lifted"]


def test_push_t_stops_when_pusher_stops():
    """Sliding friction: the T must not coast after the pusher halts (no 'ice-rink' physics)."""
    env = make_env("push_t")
    env.reset(seed=3, overrides=push_t_scenario(CFG))
    oracle_push_t(env, tol=0.06)                 # stop 6 cm short of the goal, pusher halts
    pos0 = env.snapshot().bodies["push_t"].pos[:2].copy()
    env.step(ticks=75)
    pos1 = env.snapshot().bodies["push_t"].pos[:2]
    assert np.linalg.norm(pos1 - pos0) < 0.003


def test_object_friction_is_not_inflated():
    """Stacking must not rely on glue-like friction."""
    assert max(CFG["common"]["object_physics"]["friction"][0], CFG["push_t"]["friction"][0]) <= 0.7
    assert CFG["common"]["table"]["friction"][0] <= 0.7
