#!/usr/bin/env python3
from __future__ import annotations

import argparse
import shutil
import time
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description="Remove only disposable ReKep cache/tmp and expired immutable run records.")
    parser.add_argument("--runtime-root", type=Path, default=Path.home() / ".paos-rekep")
    parser.add_argument("--max-age-days", type=int, default=14)
    parser.add_argument("--keep-latest", type=int, default=100)
    arguments = parser.parse_args()
    root = arguments.runtime_root.expanduser().resolve()
    active = root / "run/rekep/active-execution.json"
    if active.exists():
        raise SystemExit(f"refusing cleanup while an execution is active: {active}")
    for name in ("cache", "tmp"):
        target = root / name
        if target.is_dir():
            shutil.rmtree(target)
        target.mkdir(parents=True, exist_ok=True)
    cutoff = time.time() - max(0, arguments.max_age_days) * 86400
    for name in ("snapshots", "constraints", "executions"):
        directory = root / "run/rekep" / name
        if not directory.is_dir():
            continue
        files = sorted((path for path in directory.rglob("*.json") if path.is_file()), key=lambda path: path.stat().st_mtime, reverse=True)
        for path in files[max(0, arguments.keep_latest):]:
            if path.stat().st_mtime < cutoff:
                path.unlink()
    print(f"cleaned disposable runtime data under {root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
