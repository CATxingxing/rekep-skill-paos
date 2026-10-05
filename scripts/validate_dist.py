#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import re
import tarfile
import tempfile
from pathlib import Path, PurePosixPath

import yaml

ROOT = Path(__file__).resolve().parents[1]
ABSOLUTE_PATH = re.compile(rb"/(?:data\d*|home|Users)/[^\x00\r\n\"']+")


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def safe_name(name: str) -> str:
    path = PurePosixPath(name)
    if path.is_absolute() or ".." in path.parts or "\\" in name or not path.parts:
        raise ValueError(f"unsafe archive path: {name}")
    return path.as_posix()


def validate_node(archive: Path, entrypoint: str, expected_sha: str) -> None:
    if hashlib.sha256(archive.read_bytes()).hexdigest() != expected_sha:
        raise ValueError(f"node archive digest mismatch: {archive}")
    with tarfile.open(archive, "r:gz") as bundle:
        files = [item for item in bundle.getmembers() if item.isfile()]
        if len(files) != 1 or safe_name(files[0].name) != entrypoint or "/" in files[0].name:
            raise ValueError(f"node archive must contain one root executable named {entrypoint}: {archive}")
        if files[0].mode & 0o111 == 0:
            raise ValueError(f"node archive entrypoint is not executable: {archive}")


def validate_skill(archive: Path) -> None:
    with tarfile.open(archive, "r:gz") as bundle:
        members = [item for item in bundle.getmembers() if item.isfile()]
        names = {safe_name(item.name) for item in members}
        if "archive-manifest.json" not in names:
            raise ValueError("Skill archive lacks archive-manifest.json")
        payload = {}
        for member in members:
            source = bundle.extractfile(member)
            assert source is not None
            payload[safe_name(member.name)] = source.read()
        inventory = json.loads(payload["archive-manifest.json"])["files"]
        expected = {item["path"]: (item["sha256"], item["size"]) for item in inventory}
        actual = set(payload) - {"archive-manifest.json"}
        if actual != set(expected):
            raise ValueError("Skill archive inventory does not match payload")
        for name in actual:
            if (sha256_bytes(payload[name]), len(payload[name])) != expected[name]:
                raise ValueError(f"Skill archive inventory mismatch: {name}")
            if ABSOLUTE_PATH.search(payload[name]):
                raise ValueError(f"local absolute path leaked into Skill archive: {name}")
        manifest = yaml.safe_load(payload["skill.yaml"])
        if manifest["artifacts"]["resolver"] != "registry" or len(manifest["artifacts"]["nodes"]) != 8:
            raise ValueError("Skill manifest must lock exactly eight node artifacts")
        flow = yaml.safe_load(payload["profiles/sim-dobot-nova2-robotiq/dataflow.yaml"])
        if len(flow["nodes"]) != 8 or sum(item["id"] == "mujoco_sim" for item in flow["nodes"]) != 1:
            raise ValueError("simulation profile must contain eight nodes and one simulator")


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate ReKep distribution invariants.")
    parser.add_argument("--dist", type=Path, default=ROOT / "dist")
    arguments = parser.parse_args()
    if sorted(path.name for path in (ROOT / "skill-src/rekep").iterdir()) != ["SKILL.md", "skill.yaml"]:
        raise SystemExit("skill-src/rekep contains forbidden extra files")
    reused_path = ROOT / ".build/reused-nodes.lock.json"
    if not reused_path.is_file():
        raise SystemExit("reused node build lock is missing; run scripts/rebuild_reused_nodes.py")
    reused = json.loads(reused_path.read_text(encoding="utf-8"))
    standard = reused["nodes"]
    source_nodes = json.loads((ROOT / "locks/nodes.lock.json").read_text(encoding="utf-8"))["nodes"]
    generated_by_id = {item["node_id"]: item for item in standard}
    if set(generated_by_id) != {item["node_id"] for item in source_nodes}:
        raise ValueError("reused node build lock node set differs from source lock")
    for source in source_nodes:
        generated = generated_by_id[source["node_id"]]
        for field in ("artifact_id", "entrypoint", "version", "source", "revision"):
            if generated.get(field) != source.get(field):
                raise ValueError(f"reused node provenance mismatch: {source['node_id']} {field}")
    for lock in standard:
        if tuple(map(int, lock["build_glibc_baseline"].split("."))) > tuple(map(int, reused["max_glibc"].split("."))):
            raise ValueError(f"{lock['node_id']} exceeds the recorded glibc baseline")
    custom = json.loads((ROOT / ".build/custom-nodes.lock.json").read_text(encoding="utf-8"))["nodes"]
    expected_archives = {f"{lock['artifact_id']}.tar.gz" for lock in standard + custom}
    actual_archives = {path.name for path in (arguments.dist / "nodes").glob("*.tar.gz")}
    if actual_archives != expected_archives:
        raise ValueError(f"dist/nodes must contain exactly the eight locked archives: {sorted(actual_archives ^ expected_archives)}")
    for lock in standard + custom:
        validate_node(arguments.dist / "nodes" / f"{lock['artifact_id']}.tar.gz", lock["entrypoint"], lock["sha256"])
    skills = list((arguments.dist / "skills").glob("rekep-*.tar.gz"))
    if len(skills) != 1:
        raise SystemExit("expected exactly one ReKep Skill archive")
    validate_skill(skills[0])
    print("distribution validation passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
