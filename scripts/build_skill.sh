#!/usr/bin/env bash
# Build the rekep-sim Skill bundle: 3 single-file node archives + generated
# skill.yaml (with registry-style node locks) + packaged tar.gz.
#
# Local/dev build: node entrypoints are shell wrappers that exec the node
# runtime python. Host-specific paths are injected via the profile environment.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "$ROOT/scripts/env.sh"
PAOS_ENV="${PAOS_ENV:-/data7/home/linjiongxiao/miniconda3/envs/paos}"
PAOS_BIN="${PAOS_BIN:-$PAOS_ENV/bin/paos}"
PAOS_PY="${PAOS_PY:-$PAOS_ENV/bin/python}"
VP="${REKEP_SIM_PYTHON:-$PAOS_REKEP_ROOT/envs/rekep-sim/bin/python}"
VERSION="${REKEP_SIM_VERSION:-0.1.0}"
ARCH=x86_64
PLATFORM=linux
GATEWAY_ARCHIVE="${REKEP_GATEWAY_ARCHIVE:-/data7/home/linjiongxiao/.tmp/opencode/gw/gateway.tar.gz}"
REF_RUNTIME="$ROOT/reference/rekep-real-plugin/runtime"
SCENE="$ROOT/skill-src/rekep-sim/assets/dobot-nova2-robotiq/mjcf/dobot_nova2_robotiq_2f85_pick_place.xml"
STATE_DIR="${REKEP_SIM_STATE_DIR:-$PAOS_REKEP_ROOT/run/rekep-sim}"

DIST="$ROOT/dist"
rm -rf "$DIST"
mkdir -p "$DIST/nodes" "$DIST/skills"

sha() { sha256sum "$1" | awk '{print $1}'; }

# --- node archives: single file at archive root named = entrypoint -------------
for role in perception plan execute; do
  entry="rekep_${role}"
  archive="$DIST/nodes/${entry}-${VERSION}-${PLATFORM}-${ARCH}.tar.gz"
  tar -czf "$archive" -C "$ROOT/nodes/$role" "$entry"
  # archive must contain exactly one file
  test "$(tar -tzf "$archive" | wc -l)" = "1"
done

gateway_sha="$(sha "$GATEWAY_ARCHIVE")"

# --- generate skill.yaml with node locks --------------------------------------
lock() { # node_id artifact_id version entrypoint sha
  cat <<EOF
    $1:
      artifact_id: $2
      version: "$3"
      platform: $PLATFORM
      arch: $ARCH
      artifact_type: executable_tar_gz
      entrypoint: $4
      sha256: $5
EOF
}

{
cat <<EOF
manifest_version: 2
name: rekep-sim
version: "$VERSION"
description: Constraint-driven ReKep pick-and-place in headless MuJoCo (Dobot Nova2 + Robotiq 2F-85).
skill_document: SKILL.md
gateway_url: http://127.0.0.1:19021
required_tools:
  - rekep.get_context
  - rekep.plan_task
  - rekep.execute_task
profiles:
  sim:
    dataflow: profiles/sim/dataflow.yaml
    required_binaries:
      - gateway
      - rekep_perception
      - rekep_plan
      - rekep_execute
    required_assets:
      - profiles/sim/gateway.yaml
    required_environment: []
    environment:
      MUJOCO_GL: egl
      PYTHONDONTWRITEBYTECODE: "1"
      TMPDIR: "$PAOS_REKEP_ROOT/tmp"
      TMP: "$PAOS_REKEP_ROOT/tmp"
      TEMP: "$PAOS_REKEP_ROOT/tmp"
      XDG_CACHE_HOME: "$PAOS_REKEP_ROOT/cache"
      HF_HOME: "$PAOS_REKEP_ROOT/hf"
      TORCH_HOME: "$PAOS_REKEP_ROOT/torch"
      PYTHONPYCACHEPREFIX: "$PAOS_REKEP_ROOT/pycache"
      OMP_NUM_THREADS: "4"
      MKL_NUM_THREADS: "4"
      OPENBLAS_NUM_THREADS: "4"
      NUMEXPR_NUM_THREADS: "4"
      REKEP_TORCH_THREADS: "4"
      REKEP_SIM_PYTHON: "$VP"
      REKEP_SIM_SRC: "$ROOT"
      REKEP_SIM_EXTRA_PYTHONPATH: "$REF_RUNTIME"
      REKEP_SIM_SCENE: "$SCENE"
      REKEP_SIM_STATE_DIR: "$STATE_DIR"
      REKEP_DINOV2_REPO: "$PAOS_REKEP_ROOT/models/dinov2"
      REKEP_DINOV2_WEIGHTS: "$PAOS_REKEP_ROOT/models/weights/dinov2_vits14.pth"
      REKEP_DINOV2_MODEL: dinov2_vits14
artifacts:
  resolver: local
  nodes:
EOF
lock gateway "gateway-1.1.0-linux-x86_64" "1.1.0" "gateway" "$gateway_sha"
for role in perception plan execute; do
  entry="rekep_${role}"
  archive="$DIST/nodes/${entry}-${VERSION}-${PLATFORM}-${ARCH}.tar.gz"
  lock "$entry" "${entry}-${VERSION}-${PLATFORM}-${ARCH}" "$VERSION" "$entry" "$(sha "$archive")"
done
} > "$ROOT/skill-src/rekep-sim/skill.yaml"

# --- package the Skill bundle -------------------------------------------------
export PYTHONPATH="$ROOT/scripts/shim${PYTHONPATH:+:$PYTHONPATH}"
PKG="$(find /data7/home/linjiongxiao/projects -maxdepth 3 -name package_skill.py -path '*PhyAgentOS-core/scripts*' | head -1)"
if [[ -z "$PKG" ]]; then echo "package_skill.py not found" >&2; exit 1; fi
"$PAOS_PY" "$PKG" "$ROOT/skill-src/rekep-sim" --output-dir "$DIST/skills"

echo "built:"
ls -l "$DIST/nodes" "$DIST/skills"
