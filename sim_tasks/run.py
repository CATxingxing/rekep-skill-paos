"""CLI: build a seeded scene, save what a policy sees, optionally run the scripted oracle, evaluate.

    MUJOCO_GL=egl python -m sim_tasks.run --task stack_blocks --seed 42 --out /tmp/out [--oracle] [--video]

Outputs in --out:
    <task>_seed<N>.xml   generated MJCF (same seed -> identical file)
    observation.png/.npy policy-facing RGB / depth at reset
    instruction.txt      policy-facing instruction
    scene_info.json      GROUND TRUTH layout (evaluator/logging only; never give to the policy)
    result.json          evaluator output (success, partial, score, metrics, failure_reason)
    video.mp4            with --video
"""
from __future__ import annotations

import argparse
import json
import struct
import subprocess
import zlib
from pathlib import Path

import numpy as np

from . import demos, make_env
from .config import load_config


def write_png(path: Path, rgb: np.ndarray) -> None:
    h, w, _ = rgb.shape
    raw = b"".join(b"\x00" + rgb[y].tobytes() for y in range(h))

    def chunk(tag: bytes, data: bytes) -> bytes:
        body = tag + data
        return struct.pack(">I", len(data)) + body + struct.pack(">I", zlib.crc32(body) & 0xFFFFFFFF)

    path.write_bytes(b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, 2, 0, 0, 0))
                     + chunk(b"IDAT", zlib.compress(raw)) + chunk(b"IEND", b""))


class VideoRecorder:
    def __init__(self, env, path: Path, every: int = 5, fps: int = 10):
        self.env, self.every, self.count = env, every, 0
        r = env.common["render"]
        self.proc = subprocess.Popen(
            ["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{r['width']}x{r['height']}",
             "-r", str(fps), "-i", "-", "-pix_fmt", "yuv420p", "-c:v", "libx264", "-crf", "23", str(path)],
            stdin=subprocess.PIPE)
        self._orig = env.step
        env.step = self.step

    def step(self, *args, **kwargs):
        ticks = kwargs.get("ticks", 1)
        for _ in range(ticks):
            kw = dict(kwargs, ticks=1)
            self._orig(*args, **kw)
            self.count += 1
            if self.count % self.every == 0:
                self.proc.stdin.write(self.env.render_rgbd()[0].tobytes())

    def close(self):
        self.env.step = self._orig
        self.proc.stdin.close()
        self.proc.wait()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--task", required=True, choices=["general_pickup", "stack_blocks", "push_t"])
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=None, help="alternative tasks.yaml")
    parser.add_argument("--oracle", action="store_true", help="run the ground-truth-aware scripted demo (test tool, not a policy); for push_t the scene is the aligned scenario the demo can solve")
    parser.add_argument("--video", action="store_true")
    args = parser.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    env = make_env(args.task, load_config(args.config))
    overrides = demos.push_t_scenario(env.config) if args.task == "push_t" and args.oracle else None
    obs = env.reset(seed=args.seed, overrides=overrides, out_dir=args.out)
    write_png(args.out / "observation.png", obs["rgb"])
    np.save(args.out / "observation_depth.npy", obs["depth"])
    (args.out / "instruction.txt").write_text(env.get_instruction() + "\n", encoding="utf-8")
    (args.out / "scene_info.json").write_text(json.dumps({"GROUND_TRUTH_DO_NOT_GIVE_TO_POLICY": env.scene_info()}, indent=2))
    recorder = VideoRecorder(env, args.out / "video.mp4") if args.video else None
    if args.oracle:
        {"general_pickup": demos.oracle_pickup, "stack_blocks": demos.oracle_stack, "push_t": demos.oracle_push_t}[args.task](env)
        write_png(args.out / "final.png", env.render_rgbd()[0])
    if recorder:
        recorder.close()
    result = env.evaluate(final=True)
    (args.out / "result.json").write_text(json.dumps(result, indent=2))
    print(json.dumps({"task": args.task, "seed": args.seed, "instruction": env.get_instruction(),
                      "success": result["success"], "partial": result["partial"], "score": result["score"],
                      "failure_reason": result["failure_reason"]}))
    env.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
