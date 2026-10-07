from __future__ import annotations

import os
import queue
import shutil
import subprocess
import tempfile
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import IO, Any

NS = 1_000_000_000
_CLOSE = "close-segment"


def ffmpeg_executable() -> str | None:
    """The ffmpeg bundled with imageio-ffmpeg (kept inside the node build), else the host one."""
    try:
        import imageio_ffmpeg

        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        return shutil.which("ffmpeg")


@dataclass
class _Segment:
    session_id: str
    start_ns: int
    shape: tuple[int, ...]
    process: subprocess.Popen
    timeline: IO[str]
    next_slot: int = 0
    last_ns: int = 0
    last_frame: Any = None
    last_timeline_ns: int = field(default=-NS)


class VideoRecorder:
    """Rate-limited, non-blocking recorder that encodes RGB frames straight into mp4 video.

    No image files are written. Each segment is ``<root>/<session_id>/<start_wall_ns>.mp4``
    encoded at a constant ``fps``: a frame goes into the slot matching its wall time and the
    previous frame is repeated over short gaps, so a frame's presentation time is its wall
    time minus the segment start. A gap longer than ``max_gap_s``, a frame-size change or
    ``segment_s`` of video starts a new segment. Fragmented MP4 keeps a segment readable
    while it is still being written and valid if the node is killed. A small sidecar
    ``<start_wall_ns>.timeline`` holds one ``<wall_ns> <sim_time_us>`` line per second so a
    test harness can measure the simulator real-time factor. Encoding happens on a worker
    thread; when the queue is full a frame is dropped and counted instead of stalling the
    Dora input loop.

    With ``trigger_file`` set, frames are recorded only while that file exists (the executor
    writes ``active-execution.json`` for the duration of one ``execute_task``) and for
    ``stop_delay_s`` after it disappears; the segment is then finalized. A trigger file older
    than ``max_active_s`` is treated as stale (a crashed executor) and ignored.
    """

    def __init__(
        self,
        root: Path,
        *,
        fps: float,
        retention_s: float,
        segment_s: float = 600.0,
        max_gap_s: float = 2.0,
        crf: int = 23,
        max_queue: int = 64,
        ffmpeg: str | None = None,
        trigger_file: Path | None = None,
        stop_delay_s: float = 10.0,
        max_active_s: float = 1800.0,
    ):
        if fps <= 0 or retention_s <= 0 or segment_s <= 0:
            raise ValueError("recording fps, retention and segment length must be positive")
        self.root = root
        self.fps = fps
        self.period_ns = int(NS / fps)
        self.retention_ns = int(retention_s * NS)
        self.segment_ns = int(segment_s * NS)
        self.max_gap_ns = int(max_gap_s * NS)
        self.crf = crf
        self.ffmpeg = ffmpeg or ffmpeg_executable()
        self.trigger_file = trigger_file
        self.stop_delay_ns = int(stop_delay_s * NS)
        self.max_active_ns = int(max_active_s * NS)
        self._recording = trigger_file is None
        self._stop_at: int | None = None
        self.dropped = 0
        self._last_ns = 0
        self._last_prune_ns = 0
        self._segment: _Segment | None = None
        self._queue: queue.Queue[tuple[int, int, str, Any] | str | None] = queue.Queue(maxsize=max_queue)
        self._worker = threading.Thread(target=self._run, name="rekep-video-recorder", daemon=True)
        self._worker.start()

    def offer(self, frame: Any, session_id: str, sim_time_s: float | None, now_ns: int | None = None) -> bool:
        if self.ffmpeg is None:
            self.dropped += 1
            return False
        now_ns = time.time_ns() if now_ns is None else now_ns
        if now_ns - self._last_ns < self.period_ns:
            return False
        self._last_ns = now_ns
        if not self._gate(now_ns):
            return False
        sim_us = -1 if sim_time_s is None else int(round(sim_time_s * 1e6))
        try:
            self._queue.put_nowait((now_ns, sim_us, session_id, frame.copy()))
        except queue.Full:
            self.dropped += 1
            return False
        return True

    def _gate(self, now_ns: int) -> bool:
        """Whether a frame at now_ns is recorded under the execution trigger."""
        if self.trigger_file is None:
            return True
        try:
            active = time.time_ns() - self.trigger_file.stat().st_mtime_ns <= self.max_active_ns
        except OSError:
            active = False
        if active:
            self._recording, self._stop_at = True, None
            return True
        if not self._recording:
            return False
        if self._stop_at is None:
            self._stop_at = now_ns + self.stop_delay_ns
        if now_ns <= self._stop_at:
            return True
        self._recording, self._stop_at = False, None
        try:
            self._queue.put_nowait(_CLOSE)  # finalize the segment now instead of at the next start
        except queue.Full:
            pass  # the next execution starts a new segment anyway (gap > max_gap_s)
        return False

    def close(self, timeout: float = 30.0) -> None:
        """Encode the queued frames and finalize the open segment."""
        self._queue.put(None)
        self._worker.join(timeout)

    def _run(self) -> None:
        while True:
            item = self._queue.get()
            if item is None:
                self._close_segment()
                return
            if item is _CLOSE:
                self._close_segment()
                continue
            try:
                self._write(*item)
                if item[0] - self._last_prune_ns >= 60 * NS:
                    self._last_prune_ns = item[0]
                    self.prune(time.time_ns())
            except Exception:  # recording must never affect perception
                self.dropped += 1
                self._close_segment()

    def _write(self, wall_ns: int, sim_us: int, session_id: str, frame: Any) -> None:
        import numpy as np

        frame = np.ascontiguousarray(frame, dtype=np.uint8)
        segment = self._segment
        if segment is not None and (
            segment.session_id != session_id
            or segment.shape != frame.shape
            or wall_ns - segment.last_ns > self.max_gap_ns
            or wall_ns - segment.start_ns >= self.segment_ns
        ):
            self._close_segment()
            segment = None
        if segment is None:
            segment = self._segment = self._open_segment(session_id, wall_ns, frame.shape)
        slot = (wall_ns - segment.start_ns + self.period_ns // 2) // self.period_ns
        if slot < segment.next_slot:
            return  # two frames in one slot: keep the first
        stdin = segment.process.stdin
        while segment.next_slot < slot:  # repeat the previous frame over a short gap
            stdin.write(segment.last_frame.tobytes())
            segment.next_slot += 1
        stdin.write(frame.tobytes())
        segment.next_slot += 1
        segment.last_ns = wall_ns
        segment.last_frame = frame
        if wall_ns - segment.last_timeline_ns >= NS:
            segment.timeline.write(f"{wall_ns} {sim_us}\n")
            segment.timeline.flush()
            segment.last_timeline_ns = wall_ns

    def _open_segment(self, session_id: str, start_ns: int, shape: tuple[int, ...]) -> _Segment:
        if len(shape) != 3 or shape[2] != 3:
            raise ValueError(f"recording needs HxWx3 RGB frames, got {shape}")
        height, width = shape[:2]
        directory = self.root / session_id
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / f"{start_ns:020d}.mp4"
        command = [
            self.ffmpeg, "-hide_banner", "-loglevel", "error", "-y",
            "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{width}x{height}", "-r", f"{self.fps:g}", "-i", "-",
            "-an", "-vf", "pad=ceil(iw/2)*2:ceil(ih/2)*2", "-c:v", "libx264", "-preset", "veryfast",
            "-tune", "zerolatency", "-crf", str(self.crf), "-g", str(max(1, round(self.fps))), "-pix_fmt", "yuv420p",
            "-movflags", "+frag_keyframe+empty_moov+default_base_moof", str(path),
        ]
        process = subprocess.Popen(
            command, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
        )
        timeline = path.with_suffix(".timeline").open("w", encoding="utf-8")
        return _Segment(session_id, start_ns, tuple(shape), process, timeline, last_ns=start_ns)

    def _close_segment(self) -> None:
        segment, self._segment = self._segment, None
        if segment is None:
            return
        try:
            segment.timeline.close()
            segment.process.stdin.close()
            segment.process.wait(timeout=30)
        except Exception:
            segment.process.kill()

    def prune(self, now_ns: int) -> None:
        active = None
        if self._segment is not None:
            active = self.root / self._segment.session_id / f"{self._segment.start_ns:020d}"
        # *.jpg: frames left by recorders before 0.4.3, which wrote images instead of video.
        for pattern in ("*/*.mp4", "*/*.timeline", "*/*.jpg"):
            for item in self.root.glob(pattern):
                try:
                    if item.with_suffix("") == active:
                        continue
                    if now_ns - item.stat().st_mtime_ns > self.retention_ns:
                        item.unlink()
                except OSError:
                    continue
        for directory in self.root.glob("*"):
            try:
                directory.rmdir()  # only removes empty session directories
            except OSError:
                pass


# ---------------------------------------------------------------------------
# Test-harness helpers: cut one attempt's window out of the recorded segments.


def select_segments(root: Path, start_ns: int, end_ns: int) -> list[tuple[int, int, Path]]:
    """``(segment_start_ns, segment_end_ns, path)`` of segments overlapping the window.

    A segment ends at its last write (file mtime), which is also right while it is open.
    """
    segments = []
    for item in root.glob("*/*.mp4"):
        try:
            begin, end = int(item.stem), item.stat().st_mtime_ns
        except (ValueError, OSError):
            continue
        if begin <= end_ns and end >= start_ns:
            segments.append((begin, end, item))
    return sorted(segments)


def sim_span(root: Path, start_ns: int, end_ns: int) -> tuple[float, float] | None:
    """(wall seconds, simulator seconds) between the first and last timeline samples in the window."""
    samples = []
    for item in root.glob("*/*.timeline"):
        try:
            for line in item.read_text(encoding="utf-8").splitlines():
                wall, sim = (int(value) for value in line.split())
                if start_ns <= wall <= end_ns and sim >= 0:
                    samples.append((wall, sim))
        except (ValueError, OSError):
            continue
    if len(samples) < 2:
        return None
    samples.sort()
    return (samples[-1][0] - samples[0][0]) / NS, (samples[-1][1] - samples[0][1]) / 1e6


def cut_video(root: Path, start_ns: int, end_ns: int, output: Path, ffmpeg: str | None = None) -> dict:
    """Write the recorded video between start_ns and end_ns (wall clock) to ``output``."""
    ffmpeg = ffmpeg or ffmpeg_executable()
    report: dict = {"recordings": str(root), "start_ns": start_ns, "end_ns": end_ns}
    segments = select_segments(root, start_ns, end_ns)
    report["segments"] = [path.name for _, _, path in segments]
    if not segments:
        report["error"] = "no recorded video in the attempt window"
        return report
    if ffmpeg is None:
        report["error"] = "ffmpeg is not available"
        return report
    encode = ["-an", "-c:v", "libx264", "-crf", "23", "-pix_fmt", "yuv420p", "-movflags", "+faststart"]
    with tempfile.TemporaryDirectory(prefix="rekep-cut-") as directory:
        parts = []
        for index, (begin, end, path) in enumerate(segments):
            offset_s = max(0, start_ns - begin) / NS
            duration_s = (min(end_ns, end) - max(start_ns, begin)) / NS
            if duration_s <= 0:
                continue
            part = Path(directory) / f"part-{index:03d}.mp4"
            command = [ffmpeg, "-nostdin", "-hide_banner", "-loglevel", "error", "-y", "-ss", f"{offset_s:.3f}", "-i", str(path),
                       "-t", f"{duration_s:.3f}", *encode, str(part)]
            completed = subprocess.run(command, stdin=subprocess.DEVNULL, capture_output=True, text=True, timeout=600)
            if completed.returncode or not part.is_file():
                report["ffmpeg_returncode"] = completed.returncode
                report["error"] = completed.stderr[-2000:]
                return report
            parts.append(part)
        if not parts:
            report["error"] = "no recorded video in the attempt window"
            return report
        output.parent.mkdir(parents=True, exist_ok=True)
        if len(parts) == 1:
            shutil.move(str(parts[0]), output)
            returncode = 0
        else:
            listing = Path(directory) / "parts.txt"
            listing.write_text("".join(f"file '{part}'\n" for part in parts))
            command = [ffmpeg, "-nostdin", "-hide_banner", "-loglevel", "error", "-y", "-f", "concat", "-safe", "0",
                       "-i", str(listing), *encode, str(output)]
            completed = subprocess.run(command, stdin=subprocess.DEVNULL, capture_output=True, text=True, timeout=600)
            returncode = completed.returncode
            if returncode:
                report["error"] = completed.stderr[-2000:]
    report["ffmpeg_returncode"] = returncode
    report["wall_span_s"] = (min(end_ns, segments[-1][1]) - max(start_ns, segments[0][0])) / NS
    span = sim_span(root, start_ns, end_ns)
    if span is not None and span[0] > 0:
        report["sim_span_s"] = span[1]
        report["real_time_factor"] = span[1] / span[0]
    return report


def extract_frame(video: Path, output: Path, *, last: bool, ffmpeg: str | None = None) -> bool:
    """Save the first or last frame of ``video`` as a JPEG (before/after photos)."""
    ffmpeg = ffmpeg or ffmpeg_executable()
    if ffmpeg is None or not video.is_file():
        return False
    seek = ["-sseof", "-0.5"] if last else []
    keep = ["-update", "1"] if last else ["-frames:v", "1"]  # last: keep overwriting until the end
    command = [ffmpeg, "-nostdin", "-hide_banner", "-loglevel", "error", "-y", *seek, "-i", str(video),
               *keep, "-q:v", "2", str(output)]
    completed = subprocess.run(command, stdin=subprocess.DEVNULL, capture_output=True, text=True, timeout=120)
    return completed.returncode == 0 and output.is_file() and os.path.getsize(output) > 0
