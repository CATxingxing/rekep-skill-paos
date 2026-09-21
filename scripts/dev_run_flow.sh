#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "$ROOT/scripts/env.sh"
PAOS_ENV=/data7/home/linjiongxiao/miniconda3/envs/paos
export PATH="$PAOS_ENV/bin:$PATH"
export NO_PROXY=127.0.0.1,localhost no_proxy=127.0.0.1,localhost
D="$(readlink -f /data7/home/linjiongxiao/.PhyAgentOS/forge_runtime/environments/rekep-sim/sim/current)/launch/profiles/sim"
cd "$D"
for p in $(pgrep -x dora 2>/dev/null || true); do kill "$p" 2>/dev/null || true; done
sleep 1
nohup dora up > "$PAOS_REKEP_ROOT/logs/dev-dora-up.log" 2>&1 &
sleep 5
nohup dora run dataflow.yaml > "$PAOS_REKEP_ROOT/logs/dev-dora-run.log" 2>&1 &
sleep 50
echo "=== gateway lines ==="; grep -a "gateway" "$PAOS_REKEP_ROOT/logs/dev-dora-run.log" | grep -av "spawner\|daemon\|WARN" | tail -30
echo "=== ss 19021 ==="; ss -ltnp 2>/dev/null | grep 19021 || echo "not listening"
echo "=== gateway procs ==="; ps -eo pid,lstart,cmd | grep -E "versions/gateway" | grep -v grep | head
