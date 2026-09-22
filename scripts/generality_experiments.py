"""Generality experiments on the multi-object scene (frozen ReKep programs, no VLM/API).

Each experiment runs the full mechanism: perception -> constraints (frozen) ->
AST sandbox -> reference subgoal/path solvers -> MuJoCo execution -> task-level
objective check. Results + videos are written under evidence/P5g/.
"""

from __future__ import annotations

import json
import os
import shutil
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")
ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("REKEP_DINOV2_REPO", "/data7/home/linjiongxiao/.paos-rekep/models/dinov2")
os.environ.setdefault("REKEP_DINOV2_WEIGHTS", "/data7/home/linjiongxiao/.paos-rekep/models/weights/dinov2_vits14.pth")

import numpy as np  # noqa: E402

from rekep_sim.env.mujoco_env import MujocoReKepEnv  # noqa: E402
from rekep_sim.perception import perceive  # noqa: E402
from rekep_sim.runtime import ReKepRuntime, RuntimeConfig  # noqa: E402
from rekep_sim.state import env_kwargs  # noqa: E402

SCENE = ROOT / "skill-src/rekep-sim/assets/franka-panda/tabletop_panda.xml"
PROPOSER = {"num_candidates_per_mask": 10, "min_dist_bt_keypoints": 0.02}
OUT = ROOT / "evidence/P5g"


def _idx(kps, name):
    for k in kps:
        if k.get("object") == name and k.get("kind") == "center":
            return int(k["index"])
    for k in kps:
        if k.get("object") == name:
            return int(k["index"])
    raise RuntimeError(f"no keypoint for object {name!r}")


def _ids(kps, name):
    return [int(k["index"]) for k in kps if k.get("object") == name]


def _pen_axis_pair(kps, name):
    """Prefer the object's geometric long-axis endpoints; else two far proposals."""
    lo = [k for k in kps if k.get("object") == name and k.get("kind") == "axis_min"]
    hi = [k for k in kps if k.get("object") == name and k.get("kind") == "axis_max"]
    if lo and hi:
        return int(lo[0]["index"]), int(hi[0]["index"])
    proposals = [k for k in kps if k.get("object") == name and k.get("kind") == "proposal"]
    if len(proposals) < 2:
        return None, None
    best = None
    for i in range(len(proposals)):
        for j in range(i + 1, len(proposals)):
            d = float(np.linalg.norm(np.asarray(proposals[i]["position_m"]) - np.asarray(proposals[j]["position_m"])))
            if best is None or d > best[0]:
                best = (d, int(proposals[i]["index"]), int(proposals[j]["index"]))
    return best[1], best[2]


def pick_place(obj, region):
    o, r = "OBJ", "REG"
    text = (
        "num_stages = 3\n"
        "def stage1_subgoal_constraint1(end_effector, keypoints):\n"
        f"    return np.linalg.norm(end_effector - keypoints[{o}])\n"
        "def stage2_subgoal_constraint1(end_effector, keypoints):\n"
        f"    return np.linalg.norm(keypoints[{o}] - keypoints[{r}])\n"
        "def stage2_path_constraint1(end_effector, keypoints):\n"
        f"    return get_grasping_cost_by_keypoint_idx({o})\n"
        "def stage3_subgoal_constraint1(end_effector, keypoints):\n"
        f"    return np.linalg.norm(keypoints[{o}] - keypoints[{r}])\n"
        f"grasp_keypoints = [{o}, -1, -1]\n"
        f"release_keypoints = [-1, -1, {o}]\n"
    )
    return text, {"object": obj, "mode": "in_region", "target_region": region, "tol": 0.05}


def stack(a_name, b_name):
    """Return (text with OA/OB placeholders, success spec) for stack a on b."""
    text = (
        "num_stages = 3\n"
        "def stage1_subgoal_constraint1(end_effector, keypoints):\n"
        "    return np.linalg.norm(end_effector - keypoints[OA])\n"
        "def stage2_subgoal_constraint1(end_effector, keypoints):\n"
        "    off = keypoints[OB] + np.array([0.0, 0.0, 0.03])\n"
        "    return np.linalg.norm(keypoints[OA] - off)\n"
        "def stage2_path_constraint1(end_effector, keypoints):\n"
        "    return get_grasping_cost_by_keypoint_idx(OA)\n"
        "def stage3_subgoal_constraint1(end_effector, keypoints):\n"
        "    off = keypoints[OB] + np.array([0.0, 0.0, 0.03])\n"
        "    return np.linalg.norm(keypoints[OA] - off)\n"
        "grasp_keypoints = [OA, -1, -1]\n"
        "release_keypoints = [-1, -1, OA]\n"
    )
    return text, {"object": a_name, "mode": "on_top", "target_object": b_name, "tol": 0.03}


def upright(center, a, b, region, region_name, obj_name):
    """Reorient an elongated object so its long axis becomes vertical (3 stages)."""
    text = (
        "num_stages = 3\n"
        "def stage1_subgoal_constraint1(end_effector, keypoints):\n"
        f"    return np.linalg.norm(end_effector - keypoints[{a}])\n"
        "def stage2_subgoal_constraint1(end_effector, keypoints):\n"
        f"    v = keypoints[{a}] - keypoints[{b}]\n"
        "    return 1.0 - abs(v[2]) / (np.linalg.norm(v) + 1e-9)\n"
        "def stage2_path_constraint1(end_effector, keypoints):\n"
        f"    return get_grasping_cost_by_keypoint_idx({a})\n"
        "def stage3_subgoal_constraint1(end_effector, keypoints):\n"
        f"    v = keypoints[{a}] - keypoints[{b}]\n"
        "    return 1.0 - abs(v[2]) / (np.linalg.norm(v) + 1e-9)\n"
        f"grasp_keypoints = [{a}, -1, -1]\n"
        f"release_keypoints = [-1, -1, {a}]\n"
    )
    return text, {"object": obj_name, "mode": "upright", "tol_deg": 30.0}


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    env = MujocoReKepEnv(SCENE, height=320, width=420, **env_kwargs(SCENE))
    runtime = ReKepRuntime(env, RuntimeConfig())
    snapshot, _proj, kp = perceive(env, proposer_config=PROPOSER)
    kps = snapshot["keypoints"]
    names = sorted({k["object"] for k in kps})
    print("keypoints by object:", {n: len(_ids(kps, n)) for n in names})
    only = __import__("os").environ.get("GEN_ONLY", "")

    def run(name, text_template, success, obj=None, region=None):
        if only and only not in name:
            return {}
        text = text_template
        if obj is not None:
            text = text.replace("OBJ", str(_idx(kps, obj)))
        if region is not None:
            text = text.replace("REG", str(_idx(kps, region)))
        env.reset()
        program = runtime._program_from_text(text, snapshot, name, source="frozen", success=success)
        result = runtime.execute(program, allow_motion=True)
        video = env.last_video_path if hasattr(env, "last_video_path") else ""
        if video and Path(video).exists():
            shutil.copy(video, OUT / f"{name}.mp4")
        summary = {
            "experiment": name,
            "status": result.get("status"),
            "objectives": result.get("objectives"),
            "stages": result.get("stages"),
            "elapsed_s": result.get("elapsed_s"),
            "video": str(OUT / f"{name}.mp4"),
        }
        (OUT / f"{name}.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
        print(f"[{name}] {result.get('status')} objectives={result.get('objectives')}")
        return result

    results = {}
    # GEN1/GEN7: pick red cube -> region_a (run twice for reproducibility)
    t, s = pick_place("red_cube", "region_a")
    results["GEN1_pick_red_to_region_a"] = run("GEN1_pick_red_to_region_a", t, s, "red_cube", "region_a")
    results["GEN7_repeat"] = run("GEN7_repeat", t, s, "red_cube", "region_a")

    # GEN2: pick blue cube -> region_b
    t, s = pick_place("blue_cube", "region_b")
    results["GEN2_pick_blue_to_region_b"] = run("GEN2_pick_blue_to_region_b", t, s, "blue_cube", "region_b")

    # GEN3: stack red on blue
    t, s = stack("red_cube", "blue_cube")
    t = t.replace("OA", str(_idx(kps, "red_cube"))).replace("OB", str(_idx(kps, "blue_cube")))
    results["GEN3_stack_red_on_blue"] = run("GEN3_stack_red_on_blue", t, s)

    # GEN4: reorient pen upright into holder
    pa, pb = _pen_axis_pair(kps, "pen")
    if pb is not None:
        t, s = upright(_idx(kps, "pen"), pa, pb, _idx(kps, "region_holder"), "region_holder", "pen")
        results["GEN4_reorient_pen"] = run("GEN4_reorient_pen", t, s)
    else:
        print("GEN4 skipped: pen has <2 usable keypoints")

    # GEN5: constraint authenticity -- same task but constraints are no-ops (return 0)
    noop = (
        "num_stages = 3\n"
        "def stage1_subgoal_constraint1(end_effector, keypoints):\n    return 0.0\n"
        "def stage2_subgoal_constraint1(end_effector, keypoints):\n    return 0.0\n"
        "def stage2_path_constraint1(end_effector, keypoints):\n    return 0.0\n"
        "def stage3_subgoal_constraint1(end_effector, keypoints):\n    return 0.0\n"
        "grasp_keypoints = [-1, -1, -1]\nrelease_keypoints = [-1, -1, -1]\n"
    )
    t, s = pick_place("red_cube", "region_a")
    results["GEN5_noop_constraints"] = run(
        "GEN5_noop_constraints", noop, {"object": "red_cube", "mode": "in_region", "target_region": "region_a", "tol": 0.05}
    )

    # GEN6: fail-closed -- objective mismatched with constraints
    t, s = pick_place("red_cube", "region_a")
    results["GEN6_objective_mismatch"] = run(
        "GEN6_objective_mismatch", t, {"object": "red_cube", "mode": "in_region", "target_region": "region_holder", "tol": 0.03}, "red_cube", "region_a"
    )

    summary = {
        "scene": str(SCENE),
        "proposer": PROPOSER,
        "results": {k: v.get("status") for k, v in results.items()},
        "objectives": {k: v.get("objectives") for k, v in results.items()},
    }
    (OUT / "SUMMARY.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print("\nSUMMARY:", json.dumps(summary["results"], indent=2))
    env.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
