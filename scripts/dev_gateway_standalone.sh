#!/usr/bin/env bash
set -euo pipefail
source /data7/home/linjiongxiao/projects2/rekep-skill-paos/scripts/env.sh
export NO_PROXY=127.0.0.1,localhost
ENVDIR="$(readlink -f /data7/home/linjiongxiao/.PhyAgentOS/forge_runtime/environments/rekep-sim/sim/current)"
D="$ENVDIR/launch/profiles/sim"
GW="$ENVDIR/bin/gateway"
cd "$D"
echo "gw=$GW"; ls -l "$GW"; echo "gateway.yaml -> $(readlink -f gateway.yaml)"
timeout 15 "$GW" --config gateway.yaml > "$PAOS_REKEP_ROOT/logs/gw-standalone.log" 2>&1 &
sleep 8
echo "=== gw log ==="; cat "$PAOS_REKEP_ROOT/logs/gw-standalone.log" | head -30
echo "=== listening? ==="; ss -ltn 2>/dev/null | grep 19021 || echo "not listening"
echo "=== curl ==="; curl -s --noproxy '*' -m 4 http://127.0.0.1:19021/tools | head -c 600 || echo "curl failed"
