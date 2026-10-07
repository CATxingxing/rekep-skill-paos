#!/usr/bin/env python3
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import os
import platform
import shutil
import stat
import subprocess
import sys
import tarfile
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
VERSION = "0.4.4"
PLATFORM = platform.system().lower()
ARCH = {"amd64": "x86_64", "x64": "x86_64", "arm64": "aarch64"}.get(platform.machine().lower(), platform.machine().lower())
NODES = {
    "rekep-perception": "rekep_perception",
    "rekep-planner": "rekep_planner",
    "rekep-executor": "rekep_executor",
}


def sha256(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def archive_executable(executable: Path, destination: Path, entrypoint: str) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("wb") as raw:
        with gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0) as compressed:
            with tarfile.open(fileobj=compressed, mode="w", format=tarfile.PAX_FORMAT) as archive:
                info = archive.gettarinfo(str(executable), arcname=entrypoint)
                info.uid = info.gid = 0
                info.uname = info.gname = "root"
                info.mtime = 0
                info.mode = 0o755
                with executable.open("rb") as stream:
                    archive.addfile(info, stream)


def main() -> int:
    parser = argparse.ArgumentParser(description="Build three one-file ReKep node executables with PyInstaller.")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "dist/nodes")
    parser.add_argument("--clean", action="store_true")
    parser.add_argument(
        "--node",
        choices=tuple(NODES),
        action="append",
        help="Build only the selected node; may be repeated. Existing lock records for other nodes are preserved.",
    )
    parser.add_argument(
        "--reuse-executable",
        action="store_true",
        help="Re-archive an already built selected executable without invoking PyInstaller again.",
    )
    parser.add_argument(
        "--pyinstaller-lib",
        type=Path,
        help="Optional site-packages directory containing PyInstaller only; dependencies still come from the active Python environment.",
    )
    arguments = parser.parse_args()
    if PLATFORM != "linux" or ARCH != "x86_64":
        raise SystemExit("the locked release target is Linux x86_64")
    if arguments.clean:
        shutil.rmtree(ROOT / ".build/nodes", ignore_errors=True)
    lock_path = ROOT / ".build/custom-nodes.lock.json"
    records_by_id: dict[str, dict] = {}
    if arguments.node and lock_path.is_file():
        previous = json.loads(lock_path.read_text(encoding="utf-8"))
        records_by_id = {item["node_id"]: item for item in previous.get("nodes", [])}
    selected = set(arguments.node or NODES)
    for source_name, entrypoint in NODES.items():
        if source_name not in selected:
            continue
        source = ROOT / "nodes" / source_name
        build_root = ROOT / ".build/nodes" / source_name
        executable = build_root / "dist" / entrypoint
        if not arguments.reuse_executable:
            shutil.rmtree(build_root, ignore_errors=True)
            build_root.mkdir(parents=True)
        pyinstaller_arguments = [
            "--noconfirm", "--clean", "--onefile",
            "--name", entrypoint,
            "--distpath", str(build_root / "dist"),
            "--workpath", str(build_root / "work"),
            "--specpath", str(build_root),
            "--paths", str(ROOT / "nodes/common"),
            "--paths", str(source),
        ]
        if source_name == "rekep-planner":
            separator = ";" if os.name == "nt" else ":"
            pyinstaller_arguments += ["--add-data", f"{source / 'prompt_template.txt'}{separator}."]
        pyinstaller_arguments.append(str(source / "node_main.py"))
        if not arguments.reuse_executable:
            if arguments.pyinstaller_lib:
                bootstrap = "import runpy,sys; sys.path.append(sys.argv.pop(1)); runpy.run_module('PyInstaller',run_name='__main__')"
                command = [sys.executable, "-c", bootstrap, str(arguments.pyinstaller_lib.resolve()), *pyinstaller_arguments]
            else:
                command = [sys.executable, "-m", "PyInstaller", *pyinstaller_arguments]
            subprocess.run(command, check=True)
        elif not executable.is_file():
            raise SystemExit(f"cannot reuse missing executable: {executable}")
        executable.chmod(executable.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
        subprocess.run(
            [str(executable), "--help"],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=60,
        )
        artifact_id = f"{entrypoint}-{VERSION}-{PLATFORM}-{ARCH}"
        archive = arguments.output_dir / f"{artifact_id}.tar.gz"
        archive_executable(executable, archive, entrypoint)
        records_by_id[entrypoint] = {
            "node_id": entrypoint,
            "artifact_id": artifact_id,
            "version": VERSION,
            "platform": PLATFORM,
            "arch": ARCH,
            "artifact_type": "executable_tar_gz",
            "entrypoint": entrypoint,
            "sha256": sha256(archive),
        }
        print(f"built {archive.name}")
    if set(records_by_id) != set(NODES.values()):
        missing = sorted(set(NODES.values()) - set(records_by_id))
        raise SystemExit(f"custom node lock is incomplete after partial build: {missing}")
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    records = [records_by_id[entrypoint] for entrypoint in NODES.values()]
    lock_path.write_text(json.dumps({"schema_version": 1, "nodes": records}, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
