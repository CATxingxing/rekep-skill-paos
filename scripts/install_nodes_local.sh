#!/usr/bin/env bash
# Place the locked node executables + receipts directly (local resolver dev path).
# Avoids Registry downloads; mirrors what NodeInstaller would produce.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "$ROOT/scripts/env.sh"
PAOS_HOME="${PAOS_HOME:-/data7/home/linjiongxiao/.PhyAgentOS}"
VERSION="${REKEP_SIM_VERSION:-0.1.0}"
GATEWAY_ARCHIVE="${REKEP_GATEWAY_ARCHIVE:-/data7/home/linjiongxiao/.tmp/opencode/gw/gateway.tar.gz}"
NODES_ROOT="$PAOS_HOME/forge_runtime/nodes"

place() { # node_id artifact_id entrypoint archive
  local node_id="$1" artifact_id="$2" entrypoint="$3" archive="$4"
  local target="$NODES_ROOT/$node_id/versions/$artifact_id"
  rm -rf "$target"; mkdir -p "$target"
  tar -xzf "$archive" -C "$target"
  chmod +x "$target/$entrypoint"
  local asha bsha
  asha="$(sha256sum "$archive" | awk '{print $1}')"
  bsha="$(sha256sum "$target/$entrypoint" | awk '{print $1}')"
  cat > "$target/.paos-node.json" <<EOF
{"schema_version":1,"node_id":"$node_id","artifact_id":"$artifact_id","artifact_type":"executable_tar_gz","entrypoint":"$entrypoint","archive_sha256":"$asha","binary_sha256":"$bsha"}
EOF
  echo "installed node $node_id ($artifact_id)"
}

place gateway gateway-1.1.0-linux-x86_64 gateway "$GATEWAY_ARCHIVE"
for role in perception plan execute; do
  entry="rekep_${role}"
  place "$entry" "${entry}-${VERSION}-linux-x86_64" "$entry" \
    "$ROOT/dist/nodes/${entry}-${VERSION}-linux-x86_64.tar.gz"
done
