#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import tempfile
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def sha256(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description="Prepare the locked DINOv2 source and checkpoint.")
    parser.add_argument("--runtime-root", type=Path, default=Path.home() / ".paos-rekep")
    parser.add_argument("--source-dir", type=Path, help="Use this existing DINOv2 checkout instead of cloning.")
    parser.add_argument("--weights", type=Path, help="Use this existing checkpoint instead of downloading.")
    parser.add_argument("--device", default="cuda")
    arguments = parser.parse_args()
    lock = json.loads((ROOT / "locks/models.lock.json").read_text(encoding="utf-8"))["models"]["dinov2_vits14"]
    target = arguments.runtime_root.expanduser().resolve() / "models/dinov2"
    target.mkdir(parents=True, exist_ok=True)
    source_target = target / "source"
    revision_marker = source_target / "SOURCE_REVISION"
    if source_target.exists():
        revision = revision_marker.read_text(encoding="utf-8").strip() if revision_marker.is_file() else ""
        if revision != lock["source_revision"]:
            raise SystemExit(f"existing DINOv2 source has unexpected revision marker {revision!r}; remove only {source_target} and retry")
    elif arguments.source_dir:
        revision = subprocess.run(["git", "-C", str(arguments.source_dir.resolve()), "rev-parse", "HEAD"], check=True, text=True, stdout=subprocess.PIPE).stdout.strip()
        if revision != lock["source_revision"]:
            raise SystemExit(f"provided DINOv2 checkout has unexpected revision {revision}")
        shutil.copytree(arguments.source_dir.resolve(), source_target, ignore=shutil.ignore_patterns(".git"))
    else:
        with tempfile.TemporaryDirectory(prefix="rekep-dinov2-") as temporary:
            clone = Path(temporary) / "source"
            subprocess.run(["git", "clone", "--no-checkout", lock["source_url"], str(clone)], check=True)
            subprocess.run(["git", "-C", str(clone), "checkout", "--detach", lock["source_revision"]], check=True)
            shutil.copytree(clone, source_target, ignore=shutil.ignore_patterns(".git"))
    revision_marker.write_text(lock["source_revision"] + "\n", encoding="utf-8")
    weights_target = target / "dinov2_vits14.pth"
    if arguments.weights:
        shutil.copyfile(arguments.weights.resolve(), weights_target)
    elif not weights_target.is_file():
        with urllib.request.urlopen(lock["weights_url"], timeout=300) as response, weights_target.open("wb") as stream:
            shutil.copyfileobj(response, stream)
    if sha256(weights_target) != lock["weights_sha256"]:
        weights_target.unlink(missing_ok=True)
        raise SystemExit("DINOv2 checkpoint SHA-256 mismatch")
    config = {"source_path": str(source_target), "weights_path": str(weights_target), "model_name": lock["model_name"], "device": arguments.device}
    config_path = target / "config.json"
    config_path.write_text(json.dumps(config, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    config_path.chmod(0o600)
    print(f"prepared locked DINOv2 under {target}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
