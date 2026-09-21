#!/usr/bin/env python
"""Dev-only shim for running the `paos` CLI on glibc < 2.32 (e.g. Ubuntu 20.04).

Why: PAOS 1.0.0 `ArchiveValidator.extract` calls
`os.chmod(target, mode, follow_symlinks=False)`. On glibc 2.31 the kernel/glibc
`fchmodat(AT_SYMLINK_NOFOLLOW)` path is unavailable, so CPython reports
`os.chmod` as not supporting `follow_symlinks`, and the call raises
`NotImplementedError`. Archive extraction targets are regular files (links are
rejected upstream), so falling back to a plain chmod is safe for local testing.

This shim does NOT modify PhyAgentOS-core; it only patches the process at
runtime. Production install/run happens on the glibc >= 2.32 target host.

Usage:
    python scripts/paos_dev_shim.py <paos args...>
"""
from __future__ import annotations

import os
import sys

_orig_chmod = os.chmod


def _chmod(path, mode, *, dir_fd=None, follow_symlinks=True):
    try:
        return _orig_chmod(path, mode, dir_fd=dir_fd, follow_symlinks=follow_symlinks)
    except NotImplementedError:
        return _orig_chmod(path, mode)


os.chmod = _chmod
try:
    os.supports_follow_symlinks = set(os.supports_follow_symlinks) | {os.chmod}
except Exception:
    pass

from PhyAgentOS.cli.commands import app  # noqa: E402

if __name__ == "__main__":
    sys.argv[0] = "paos"
    raise SystemExit(app())
