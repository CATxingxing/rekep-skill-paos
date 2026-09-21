#!/usr/bin/env bash
# Local acceptance run (must be executed inside the private network namespace).
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "$ROOT/scripts/env.sh"
PAOS_ENV="${PAOS_ENV:-/data7/home/linjiongxiao/miniconda3/envs/paos}"
export PATH="$PAOS_ENV/bin:$PATH"
export PYTHONPATH="$ROOT/scripts/shim${PYTHONPATH:+:$PYTHONPATH}"
export NO_PROXY=127.0.0.1,localhost
export no_proxy=127.0.0.1,localhost
P="python $ROOT/scripts/paos_dev_shim.py"
EV="$ROOT/evidence/P3"
mkdir -p "$EV"

$P skill stop rekep-sim >/dev/null 2>&1 || true
echo "=== start ==="
$P skill start rekep-sim --profile sim 2>&1 | tail -5
echo "=== status ==="
$P skill status rekep-sim 2>&1 | tail -15
echo "=== probe ==="
python3 "$ROOT/scripts/gateway_probe.py" 2>&1 | tail -20
