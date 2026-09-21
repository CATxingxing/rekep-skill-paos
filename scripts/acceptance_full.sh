#!/usr/bin/env bash
# P4 acceptance: run inside the private namespace (scripts/nsrun.sh).
# Covers B/D (reference parity + sandbox via pytest) and A/C/E/F (Skill runtime
# contract, positive lifecycle, negative fail-closed contracts) via the gateway.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "$ROOT/scripts/env.sh"
PAOS_ENV="${PAOS_ENV:-/data7/home/linjiongxiao/miniconda3/envs/paos}"
export PATH="$PAOS_ENV/bin:$PATH"
export PYTHONPATH="$ROOT/scripts/shim:$ROOT/scripts${PYTHONPATH:+:$PYTHONPATH}"
export NO_PROXY=127.0.0.1,localhost
export no_proxy=127.0.0.1,localhost
export REKEP_SIM_VERSION=0.1.2
VP="$PAOS_REKEP_ROOT/envs/rekep-sim"
EV="$ROOT/evidence/P4"
mkdir -p "$EV"

echo "=== B/D: reference parity + sandbox (pytest) ==="
REKEP_DINOV2_REPO="$PAOS_REKEP_ROOT/models/dinov2" \
REKEP_DINOV2_WEIGHTS="$PAOS_REKEP_ROOT/models/weights/dinov2_vits14.pth" \
  "$VP/bin/python" -m pytest "$ROOT/tests" -q 2>&1 | tee "$EV/pytest.log" | tail -3

echo "=== build + install bundle ==="
python "$ROOT/scripts/paos_dev_shim.py" skill stop rekep-sim >/dev/null 2>&1 || true
bash "$ROOT/scripts/build_skill.sh" >/dev/null
python "$ROOT/scripts/paos_dev_shim.py" skill install "$ROOT/dist/skills/rekep-sim-0.1.2.tar.gz" --local --yes 2>&1 | tail -3
bash "$ROOT/scripts/install_nodes_local.sh" 2>&1 | tail -4

echo "=== start + A/C/E/F probe ==="
python "$ROOT/scripts/paos_dev_shim.py" skill stop rekep-sim >/dev/null 2>&1 || true
python "$ROOT/scripts/paos_dev_shim.py" skill start rekep-sim --profile sim 2>&1 | tail -3
python "$ROOT/scripts/paos_dev_shim.py" skill status rekep-sim 2>&1 | tail -8
python "$ROOT/scripts/acceptance_full.py" 2>&1 | tee "$EV/probe.log" | tail -30
