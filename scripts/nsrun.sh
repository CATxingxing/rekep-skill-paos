#!/usr/bin/env bash
# Run a command inside a private network namespace with loopback up.
# Isolates Dora's fixed ports (6012 / 53290 / 53291) from other users on the host.
set -euo pipefail
exec unshare -rn bash -c 'ip link set lo up 2>/dev/null || ifconfig lo up 2>/dev/null || true; exec "$@"' _ "$@"
