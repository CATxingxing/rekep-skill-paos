#!/usr/bin/env bash
# E acceptance: the OS LLM reads SKILL.md and drives the tool chain via paos agent.
# Run inside scripts/nsrun.sh. No VLM key is exported (planning uses the template
# path) so this isolates the Agent/SKILL.md orchestration under test.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "$ROOT/scripts/env.sh"
PAOS_ENV="${PAOS_ENV:-/data7/home/linjiongxiao/miniconda3/envs/paos}"
export PATH="$PAOS_ENV/bin:$PATH"
export PYTHONPATH="$ROOT/scripts/shim${PYTHONPATH:+:$PYTHONPATH}"
export NO_PROXY=127.0.0.1,localhost
export no_proxy=127.0.0.1,localhost
P="python $ROOT/scripts/paos_dev_shim.py"

$P skill stop rekep-sim >/dev/null 2>&1 || true
echo "=== start ==="
$P skill start rekep-sim --profile sim 2>&1 | tail -3
$P skill status rekep-sim 2>&1 | tail -8
echo "=== paos agent ==="
timeout 900 $P agent -m '使用 rekep-sim：抓取橙色方块并将其放入绿色目标区域；先感知和规划，再明确执行仿真，最后返回视频路径。' 2>&1 | tail -80
echo "=== status after ==="
$P skill status rekep-sim 2>&1 | tail -8
