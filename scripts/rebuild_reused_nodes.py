#!/usr/bin/env python3
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import os
import platform
import re
import shutil
import subprocess
import sys
import tarfile
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

SOURCE_PATCHES = {
    "motion": (
        ROOT / "patches/motion/0001-reserve-final-pose-error-budget.patch",
    ),
    "gateway": (
        ROOT / "patches/gateway/0001-action-admission-status-barrier.patch",
    ),
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def version_tuple(value: str) -> tuple[int, ...]:
    return tuple(int(part) for part in value.split("."))


def host_glibc() -> str:
    family, version = platform.libc_ver()
    if family != "glibc" or not version:
        raise SystemExit("this build path requires a glibc Linux host")
    return version


def required_glibc(executable: Path) -> str:
    output = subprocess.run(
        ["readelf", "--version-info", str(executable)], check=True, text=True,
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
    ).stdout
    versions = re.findall(r"GLIBC_(\d+\.\d+)", output)
    return max(versions, key=version_tuple) if versions else "0.0"


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


def build_local(checkout: Path, script: str) -> None:
    subprocess.run(["bash", script], cwd=checkout, check=True)


def build_container(checkout: Path, work: Path, script: str, image: str) -> Path:
    if shutil.which("docker") is None:
        raise SystemExit("docker is required for --mode container")
    if work.exists():
        shutil.rmtree(work)
    work.mkdir(parents=True)
    command = (
        f"trap 'chown -R {os.getuid()}:{os.getgid()} /work' EXIT; "
        "apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y "
        "binutils ca-certificates curl file git libegl1 libgl1 libglfw3 libgomp1 libosmesa6 "
        "&& rm -rf /var/lib/apt/lists/* "
        "&& curl -LsSf https://astral.sh/uv/0.12.5/install.sh | env UV_INSTALL_DIR=/usr/local/bin sh "
        "&& uv python install 3.12 "
        "&& cp -a /src/. /work/repo && cd /work/repo "
        f"&& bash {script}"
    )
    subprocess.run([
        "docker", "run", "--rm", "--platform", "linux/amd64",
        "-v", f"{checkout.resolve()}:/src:ro", "-v", f"{work.resolve()}:/work",
        image, "bash", "-lc", command,
    ], check=True)
    return work / "repo"


def apply_source_patches(repository: str, checkout: Path) -> list[dict[str, str]]:
    records = []
    for patch in SOURCE_PATCHES.get(repository, ()):
        if not patch.is_file():
            raise SystemExit(f"source patch is missing: {patch}")
        check = subprocess.run(
            ["git", "apply", "--check", str(patch)],
            cwd=checkout,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        if check.returncode == 0:
            subprocess.run(["git", "apply", str(patch)], cwd=checkout, check=True)
        else:
            reverse = subprocess.run(
                ["git", "apply", "--reverse", "--check", str(patch)],
                cwd=checkout,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )
            if reverse.returncode != 0:
                detail = check.stderr.strip() or reverse.stderr.strip()
                raise SystemExit(f"source patch does not apply cleanly to {repository}: {detail}")
        records.append({
            "path": patch.relative_to(ROOT).as_posix(),
            "sha256": sha256(patch),
        })
    return records


def main() -> int:
    parser = argparse.ArgumentParser(description="Rebuild reused Forge Nodes from pinned source on a compatible glibc baseline.")
    parser.add_argument("--checkout-dir", type=Path, default=ROOT / ".build/upstream-sources")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "dist/nodes")
    parser.add_argument("--mode", choices=("local", "container"), default="local")
    parser.add_argument("--container-image", default="ubuntu:20.04", help="Official Forge release baseline (glibc 2.31).")
    parser.add_argument("--max-glibc", help="Reject binaries requiring a newer GLIBC symbol; defaults to the host version.")
    parser.add_argument("--build-glibc", help="Explicit container glibc baseline when using a non-standard image.")
    parser.add_argument("--skip-fetch", action="store_true")
    parser.add_argument("--reuse-built", action="store_true", help="Reuse existing official dist executables after revision and smoke checks.")
    arguments = parser.parse_args()
    if platform.system() != "Linux" or platform.machine() not in {"x86_64", "AMD64"}:
        raise SystemExit("locked node artifacts target Linux x86_64")
    if not arguments.skip_fetch:
        subprocess.run([
            sys.executable, os.fspath(Path(__file__).with_name("fetch_nodes.py")),
            "--checkout-dir", os.fspath(arguments.checkout_dir),
        ], check=True)
    maximum = arguments.max_glibc or host_glibc()
    if arguments.mode == "local":
        build_glibc = host_glibc()
    else:
        known_images = {"ubuntu:20.04": "2.31", "ubuntu:22.04": "2.35"}
        build_glibc = arguments.build_glibc or known_images.get(arguments.container_image)
        if not build_glibc:
            raise SystemExit("--build-glibc is required for an unknown container image")
    if version_tuple(build_glibc) > version_tuple(maximum):
        raise SystemExit(f"build environment glibc {build_glibc} exceeds allowed target baseline {maximum}")
    lock = json.loads((ROOT / "locks/nodes.lock.json").read_text(encoding="utf-8"))
    repositories: dict[str, dict] = {}
    for node in lock["nodes"]:
        repositories.setdefault(node["repository"], node)
    built_roots: dict[str, Path] = {}
    applied_patches: dict[str, list[dict[str, str]]] = {}
    for repository, node in repositories.items():
        checkout = arguments.checkout_dir.resolve() / repository
        revision = subprocess.run(["git", "rev-parse", "HEAD"], cwd=checkout, check=True, text=True, stdout=subprocess.PIPE).stdout.strip()
        if revision != node["revision"]:
            raise SystemExit(f"source revision mismatch for {repository}: {revision}")
        applied_patches[repository] = apply_source_patches(repository, checkout)
        repository_nodes = [item for item in lock["nodes"] if item["repository"] == repository]
        already_built = all((checkout / item["source_executable"]).is_file() for item in repository_nodes)
        if arguments.reuse_built and already_built:
            built_roots[repository] = checkout
            continue
        if arguments.mode == "local":
            build_local(checkout, node["build_script"])
            built_roots[repository] = checkout
        else:
            built_roots[repository] = build_container(
                checkout, ROOT / ".build/container-builds" / repository,
                node["build_script"], arguments.container_image,
            )
    arguments.output_dir.mkdir(parents=True, exist_ok=True)
    records = []
    for node in lock["nodes"]:
        executable = built_roots[node["repository"]] / node["source_executable"]
        if not executable.is_file() or executable.stat().st_mode & 0o111 == 0:
            raise SystemExit(f"official build did not produce executable: {executable}")
        with tempfile.TemporaryDirectory(prefix="rekep-node-smoke-") as temporary:
            subprocess.run([str(executable), "--help"], cwd=temporary, check=True, timeout=120, stdout=subprocess.DEVNULL)
        bootloader_glibc = required_glibc(executable)
        if version_tuple(bootloader_glibc) > version_tuple(maximum):
            raise SystemExit(f"{node['entrypoint']} bootloader requires GLIBC_{bootloader_glibc}, above allowed GLIBC_{maximum}")
        archive = arguments.output_dir / f"{node['artifact_id']}.tar.gz"
        archive_executable(executable, archive, node["entrypoint"])
        record = {
            **node,
            "platform": lock["platform"],
            "arch": lock["arch"],
            "artifact_type": "executable_tar_gz",
            "sha256": sha256(archive),
            "bootloader_required_glibc": bootloader_glibc,
            "build_glibc_baseline": build_glibc,
            "build_mode": arguments.mode,
            "build_image": arguments.container_image if arguments.mode == "container" else None,
            "source_patches": applied_patches[node["repository"]],
        }
        records.append(record)
        print(f"rebuilt {archive.name}: build glibc {build_glibc}, bootloader GLIBC_{bootloader_glibc}, sha256={record['sha256']}")
    generated = ROOT / ".build/reused-nodes.lock.json"
    generated.parent.mkdir(parents=True, exist_ok=True)
    generated.write_text(json.dumps({"schema_version": 1, "max_glibc": maximum, "nodes": records}, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {generated}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
