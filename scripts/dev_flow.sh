#!/usr/bin/env bash
set -u
ROOT="/data7/home/linjiongxiao/projects2/rekep-skill-paos"
source "$ROOT/scripts/env.sh"
PAOS_ENV=/data7/home/linjiongxiao/miniconda3/envs/paos
export PATH="$PAOS_ENV/bin:$PATH"
export NO_PROXY=127.0.0.1,localhost no_proxy=127.0.0.1,localhost
EV="$ROOT/evidence/P3/dev"
D="$(readlink -f /data7/home/linjiongxiao/.PhyAgentOS/forge_runtime/environments/rekep-sim/sim/current)/launch/profiles/sim"
cd "$D"
pkill -x dora 2>/dev/null; sleep 1
( dora up > "$EV/up.log" 2>&1 & )
sleep 6
( dora run dataflow.yaml > "$EV/run.log" 2>&1 & )
sleep 90
ss -ltn > "$EV/ss.log" 2>&1
ps -eo pid,ppid,cmd > "$EV/ps.log" 2>&1
curl -s --noproxy '*' -m 5 http://127.0.0.1:19021/tools > "$EV/tools.json" 2>&1
echo "done"
