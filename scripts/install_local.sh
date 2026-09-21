#!/usr/bin/env bash
# Install (locally) and start the rekep-sim Skill using the dev shim for glibc<2.32.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "$ROOT/scripts/env.sh"
PAOS_ENV="${PAOS_ENV:-/data7/home/linjiongxiao/miniconda3/envs/paos}"
export PATH="$PAOS_ENV/bin:$PATH"
export PYTHONPATH="$ROOT/scripts/shim${PYTHONPATH:+:$PYTHONPATH}"
PAOS="python $ROOT/scripts/paos_dev_shim.py"
VERSION="${REKEP_SIM_VERSION:-0.1.0}"
GATEWAY_ARCHIVE="${REKEP_GATEWAY_ARCHIVE:-/data7/home/linjiongxiao/.tmp/opencode/gw/gateway.tar.gz}"

$PAOS skill install "$ROOT/dist/skills/rekep-sim-${VERSION}.tar.gz" --local --yes
$PAOS forge-node install rekep-sim gateway --archive "$GATEWAY_ARCHIVE"
for role in perception plan execute; do
  entry="rekep_${role}"
  $PAOS forge-node install rekep-sim "$entry" \
    --archive "$ROOT/dist/nodes/${entry}-${VERSION}-linux-x86_64.tar.gz"
done
$PAOS skill inspect rekep-sim
