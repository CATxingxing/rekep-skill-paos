#!/usr/bin/env python3
from __future__ import annotations

import gzip
import hashlib
import json
import shutil
import stat
import tarfile
import tempfile
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
DIST = ROOT / "dist"


def sha256(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def archive_tree(root: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("wb") as raw:
        with gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0) as compressed:
            with tarfile.open(fileobj=compressed, mode="w", format=tarfile.PAX_FORMAT) as archive:
                for path in sorted(root.rglob("*")):
                    if not path.is_file():
                        continue
                    info = archive.gettarinfo(str(path), arcname=path.relative_to(root).as_posix())
                    info.uid = info.gid = 0
                    info.uname = info.gname = "root"
                    info.mtime = 0
                    with path.open("rb") as stream:
                        archive.addfile(info, stream)


def locks() -> list[dict]:
    reused_path = ROOT / ".build/reused-nodes.lock.json"
    if not reused_path.is_file():
        raise SystemExit("reused node build lock is missing; run scripts/rebuild_reused_nodes.py")
    standard = json.loads(reused_path.read_text(encoding="utf-8"))["nodes"]
    source = json.loads((ROOT / "locks/nodes.lock.json").read_text(encoding="utf-8"))["nodes"]
    generated_by_id = {item["node_id"]: item for item in standard}
    if set(generated_by_id) != {item["node_id"] for item in source}:
        raise SystemExit("reused node build lock does not match the source lock node set")
    immutable_fields = ("artifact_id", "entrypoint", "version", "source", "revision")
    for expected in source:
        actual = generated_by_id[expected["node_id"]]
        if any(actual.get(field) != expected.get(field) for field in immutable_fields):
            raise SystemExit(f"reused node build lock provenance mismatch: {expected['node_id']}")
    custom_path = ROOT / ".build/custom-nodes.lock.json"
    if not custom_path.is_file():
        raise SystemExit("custom node lock is missing; run scripts/build_nodes.py")
    custom = json.loads(custom_path.read_text(encoding="utf-8"))["nodes"]
    return standard + custom


def main() -> int:
    source_entries = sorted(path.name for path in (ROOT / "skill-src/rekep").iterdir())
    if source_entries != ["SKILL.md", "skill.yaml"]:
        raise SystemExit(f"skill-src/rekep must contain only SKILL.md and skill.yaml: {source_entries}")
    node_locks = locks()
    expected_node_archives = {f"{lock['artifact_id']}.tar.gz" for lock in node_locks}
    for stale in (DIST / "nodes").glob("*.tar.gz"):
        if stale.name not in expected_node_archives:
            stale.unlink()
    for lock in node_locks:
        archive = DIST / "nodes" / f"{lock['artifact_id']}.tar.gz"
        if not archive.is_file() or sha256(archive) != lock["sha256"]:
            raise SystemExit(f"locked node archive is missing or changed: {archive}")
    with tempfile.TemporaryDirectory(prefix="rekep-skill-") as temporary:
        stage = Path(temporary) / "rekep"
        stage.mkdir()
        shutil.copy2(ROOT / "skill-src/rekep/SKILL.md", stage / "SKILL.md")
        shutil.copy2(ROOT / "skill-src/rekep/skill.yaml", stage / "skill.yaml")
        shutil.copytree(ROOT / "profiles", stage / "profiles")
        shutil.copytree(ROOT / "assets", stage / "assets")
        shutil.copy2(ROOT / "scripts/prepare_runtime.py", stage / "prepare_runtime.py")
        start = stage / "start.sh"
        start.write_text(
            "#!/usr/bin/env bash\nset -euo pipefail\n"
            "ROOT=\"$(cd \"$(dirname \"${BASH_SOURCE[0]}\")\" && pwd)\"\n"
            "exec python3 \"${ROOT}/prepare_runtime.py\"\n",
            encoding="utf-8",
        )
        start.chmod(start.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
        manifest_path = stage / "skill.yaml"
        manifest = yaml.safe_load(manifest_path.read_text(encoding="utf-8"))
        manifest["artifacts"] = {"resolver": "registry", "nodes": {}}
        for lock in node_locks:
            manifest["artifacts"]["nodes"][lock["node_id"]] = {
                "artifact_id": lock["artifact_id"],
                "version": lock["version"],
                "platform": lock["platform"],
                "arch": lock["arch"],
                "artifact_type": "executable_tar_gz",
                "entrypoint": lock["entrypoint"],
                "sha256": lock["sha256"],
            }
        manifest_path.write_text(yaml.safe_dump(manifest, sort_keys=False, allow_unicode=True), encoding="utf-8")
        files = []
        for path in sorted(item for item in stage.rglob("*") if item.is_file()):
            files.append({"path": path.relative_to(stage).as_posix(), "sha256": sha256(path), "size": path.stat().st_size})
        archive_manifest = {"manifest_version": 1, "kind": "skill", "name": manifest["name"], "version": manifest["version"], "files": files}
        (stage / "archive-manifest.json").write_text(json.dumps(archive_manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        destination = DIST / "skills" / f"{manifest['name']}-{manifest['version']}.tar.gz"
        for stale in (DIST / "skills").glob("rekep-*.tar.gz"):
            if stale != destination:
                stale.unlink()
        archive_tree(stage, destination)
    outputs = sorted((DIST / "nodes").glob("*.tar.gz")) + sorted((DIST / "skills").glob("*.tar.gz"))
    (DIST / "SHA256SUMS").write_text("".join(f"{sha256(path)}  {path.relative_to(DIST).as_posix()}\n" for path in outputs), encoding="utf-8")
    print(f"packaged {destination}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
