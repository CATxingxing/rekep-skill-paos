"""sitecustomize shim: make `os.chmod(..., follow_symlinks=False)` work on glibc < 2.32.

Loaded automatically by CPython when this directory is on PYTHONPATH. It patches
only the current process and does not modify PhyAgentOS-core. See
scripts/paos_dev_shim.py for the rationale.

Enable:  export PYTHONPATH=<repo>/scripts/shim:$PYTHONPATH
"""
from __future__ import annotations

import os

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
