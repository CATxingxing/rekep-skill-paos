# ReKep PAOS Skill

This repository builds a relocatable PhyAgentOS Skill with three custom nodes:

- `rekep_perception` consumes one external RGB-D/state stream and emits immutable `PerceptionSnapshot` values.
- `rekep_planner` asks the VLM selected by the current PAOS configuration for a restricted JSON constraint program.
- `rekep_executor` validates and solves the exact program, then serially drives standard motion and gripper actions.

The ReKep nodes never create or mutate a simulator. The simulation profile uses one downloaded `mujoco_sim` node as the only scene owner. Inter-node data travels through Dora; immutable files are restart recovery records, not a `latest`-file transport.

The `sim-dobot-nova2-robotiq` profile models one Dobot Nova 2 arm with a Robotiq 2F-85 gripper. It is not a full Dobot X-Trainer model: the commercial X-Trainer is a dual-arm platform built around two Nova 2 slave arms and uses its own slave grippers and cameras. Results from this profile therefore validate the Nova 2 single-arm simulation chain only.

## Layout

```text
skill-src/rekep/              only the model-facing Skill document and manifest
profiles/sim-dobot-nova2-robotiq/ launch topology and runtime configuration
nodes/                        the three custom nodes and their shared contracts
assets/                       robot assets staged into the built Skill bundle
locks/                        node, model, and migrated-upstream provenance
scripts/                      deterministic fetch/build/package/runtime helpers
tests/                        contract, binding, solver, serialization, and profile tests
```

## Build and validate

```bash
uv sync --extra build --extra dev
uv run python scripts/fetch_nodes.py
uv run python scripts/rebuild_reused_nodes.py --mode container --max-glibc 2.31
uv run python scripts/build_nodes.py
uv run python scripts/package_skill.py
uv run python scripts/validate_dist.py
uv run pytest
```

The five generic Forge nodes are rebuilt from pinned official source instead of trusting downloaded executables. `--mode container` reproduces the upstream Ubuntu 20.04 / glibc 2.31 release baseline; `--mode local --max-glibc 2.35` is the direct build path for an Ubuntu 22.04 host. See `REFACTOR_AND_TEST_GUIDE.md` for the complete file-by-file runtime and test guide.

`prepare_runtime.py` copies only the active PAOS model/provider settings into a mode-0600 runtime file. The planner fails closed if that provider is unavailable or cannot accept OpenAI-compatible multimodal requests. There is no template planner or generated-Python execution path.

The repository does not contain generated `dist/` output, credentials, runtime state, model weights, or package environments.
