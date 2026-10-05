#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def run(*command: str, cwd: Path | None = None) -> str:
    result = subprocess.run(command, cwd=cwd, check=True, text=True, stdout=subprocess.PIPE)
    return result.stdout.strip()


def main() -> int:
    parser = argparse.ArgumentParser(description="Fetch pinned Forge Node source checkouts; no prebuilt binary is trusted.")
    parser.add_argument("--checkout-dir", type=Path, default=ROOT / ".build/upstream-sources")
    parser.add_argument("--clean", action="store_true", help="Remove only the selected checkout directory before cloning.")
    arguments = parser.parse_args()
    checkout_dir = arguments.checkout_dir.resolve()
    if arguments.clean and checkout_dir.is_dir():
        shutil.rmtree(checkout_dir)
    checkout_dir.mkdir(parents=True, exist_ok=True)
    lock = json.loads((ROOT / "locks/nodes.lock.json").read_text(encoding="utf-8"))
    repositories: dict[str, dict] = {}
    for node in lock["nodes"]:
        current = repositories.setdefault(node["repository"], node)
        if current["source"] != node["source"] or current["revision"] != node["revision"]:
            raise SystemExit(f"repository {node['repository']} has conflicting source pins")
    for name, node in repositories.items():
        destination = checkout_dir / name
        if not (destination / ".git").is_dir():
            if destination.exists():
                raise SystemExit(f"refusing to overwrite non-Git path: {destination}")
            run("git", "clone", "--filter=blob:none", "--no-checkout", node["source"], str(destination))
        remote = run("git", "remote", "get-url", "origin", cwd=destination).removesuffix(".git")
        if remote != node["source"].removesuffix(".git"):
            raise SystemExit(f"unexpected origin for {destination}: {remote}")
        run("git", "fetch", "--filter=blob:none", "origin", node["revision"], cwd=destination)
        run("git", "checkout", "--detach", node["revision"], cwd=destination)
        revision = run("git", "rev-parse", "HEAD", cwd=destination)
        if revision != node["revision"]:
            raise SystemExit(f"revision mismatch for {destination}: {revision}")
        print(f"fetched {name} at {revision}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
