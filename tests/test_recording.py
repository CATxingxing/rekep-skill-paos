from __future__ import annotations

import importlib.util
import json
import os
import shutil
import subprocess
import time
from pathlib import Path

import numpy as np
import pytest

from perception import Perception
from recorder import VideoRecorder, cut_video, extract_frame, ffmpeg_executable, select_segments, sim_span

spec = importlib.util.spec_from_file_location("acceptance_baseline", Path(__file__).resolve().parents[1] / "scripts/acceptance_baseline.py")
acceptance = importlib.util.module_from_spec(spec)
spec.loader.exec_module(acceptance)

needs_ffmpeg = pytest.mark.skipif(ffmpeg_executable() is None, reason="ffmpeg is required to encode recordings")
needs_ffprobe = pytest.mark.skipif(shutil.which("ffprobe") is None, reason="ffprobe is required to inspect recordings")


def _frames(video: Path) -> int:
    output = subprocess.run(
        ["ffprobe", "-v", "error", "-count_frames", "-select_streams", "v:0", "-show_entries", "stream=nb_read_frames",
         "-of", "json", str(video)],
        capture_output=True, text=True, check=True,
    ).stdout
    return int(json.loads(output)["streams"][0]["nb_read_frames"])


def _record(root: Path, walls_and_sims, *, shape=(48, 64, 3), session="session_a", **options) -> VideoRecorder:
    recorder = VideoRecorder(root, fps=10, retention_s=3600, max_queue=len(walls_and_sims) + 1, **options)
    for index, (wall, sim) in enumerate(walls_and_sims):
        frame = np.full(shape, (index * 20) % 255, dtype=np.uint8)
        assert recorder.offer(frame, session, sim, now_ns=wall)
    recorder.close()
    return recorder


def test_recorder_is_rate_limited():
    recorder = VideoRecorder(Path("/nonexistent"), fps=10, retention_s=3600, ffmpeg="/bin/true")
    frame = np.zeros((8, 8, 3), dtype=np.uint8)
    base = time.time_ns()
    recorder._queue.put = recorder._queue.put_nowait = lambda item: None  # nothing reaches the encoder
    assert recorder.offer(frame, "session_a", 1.25, now_ns=base)
    assert not recorder.offer(frame, "session_a", 1.26, now_ns=base + 50_000_000)
    assert recorder.offer(frame, "session_a", 1.35, now_ns=base + 100_000_000)


def test_recorder_without_ffmpeg_drops_frames(monkeypatch, tmp_path):
    import recorder as module

    monkeypatch.setattr(module, "ffmpeg_executable", lambda: None)
    recorder = VideoRecorder(tmp_path, fps=10, retention_s=3600)
    assert not recorder.offer(np.zeros((8, 8, 3), dtype=np.uint8), "session_a", None)
    assert recorder.dropped == 1 and not any(tmp_path.iterdir())


@needs_ffmpeg
@needs_ffprobe
def test_recorder_writes_only_video_with_wall_clock_slots(tmp_path):
    base = time.time_ns()
    # 10 fps for 3 s with one 0.5 s gap that is filled by repeating the previous frame.
    walls = [base + i * 100_000_000 for i in range(30) if not 10 <= i < 15]
    _record(tmp_path, [(wall, (wall - base) / 2e9) for wall in walls])
    files = sorted(path.name for path in tmp_path.rglob("*") if path.is_file())
    assert files == [f"{base:020d}.mp4", f"{base:020d}.timeline"]
    assert not list(tmp_path.rglob("*.jpg"))
    assert _frames(tmp_path / "session_a" / f"{base:020d}.mp4") == 30
    wall_s, sim_s = sim_span(tmp_path, base, base + 10 * 10**9)
    assert sim_s == pytest.approx(wall_s / 2, rel=1e-3)


@needs_ffmpeg
def test_recorder_starts_a_new_segment_after_a_long_gap_or_size_change(tmp_path):
    base = time.time_ns()
    recorder = VideoRecorder(tmp_path, fps=10, retention_s=3600, max_gap_s=1.0)
    small, large = np.zeros((16, 16, 3), dtype=np.uint8), np.zeros((32, 32, 3), dtype=np.uint8)
    recorder.offer(small, "session_a", None, now_ns=base)
    recorder.offer(small, "session_a", None, now_ns=base + 3 * 10**9)  # gap > max_gap_s
    recorder.offer(large, "session_a", None, now_ns=base + 3 * 10**9 + 100_000_000)  # new size
    recorder.close()
    assert len(list(tmp_path.glob("session_a/*.mp4"))) == 3


def test_recorder_prunes_old_segments_and_legacy_frames(tmp_path):
    recorder = VideoRecorder(tmp_path, fps=10, retention_s=1, ffmpeg="/bin/true")
    now = time.time_ns()
    (tmp_path / "old").mkdir()
    (tmp_path / "new").mkdir()
    for name in ("old/00000000000000000001.mp4", "old/00000000000000000001.timeline", "old/00000000000000000001_0.jpg"):
        (tmp_path / name).write_bytes(b"x")
        os.utime(tmp_path / name, ns=(now - 5 * 10**9, now - 5 * 10**9))
    (tmp_path / "new" / "00000000000000000002.mp4").write_bytes(b"x")
    recorder.prune(now)
    assert not (tmp_path / "old").exists()
    assert (tmp_path / "new" / "00000000000000000002.mp4").exists()


@needs_ffmpeg
@needs_ffprobe
def test_attempt_video_is_cut_from_segments_and_reports_real_time_factor(tmp_path):
    recordings = tmp_path / "recordings"
    base = time.time_ns() - 60 * 10**9
    # Two segments (forced by a 3 s gap); the simulator advances half as fast as the wall clock.
    walls = [base + i * 100_000_000 for i in range(40)] + [base + 7 * 10**9 + i * 100_000_000 for i in range(40)]
    _record(recordings, [(wall, (wall - base) / 2e9) for wall in walls], max_gap_s=1.0)
    for path in recordings.glob("*/*.mp4"):  # mtime marks the segment end
        begin = int(path.stem)
        end = begin + 39 * 100_000_000
        os.utime(path, ns=(end, end))
    assert len(select_segments(recordings, base, base + 11 * 10**9)) == 2
    output = tmp_path / "attempt"
    output.mkdir()
    report = acceptance.write_video(output, recordings, base + 2 * 10**9, base + 9 * 10**9)
    assert report["ffmpeg_returncode"] == 0 and len(report["segments"]) == 2
    assert report["real_time_factor"] == pytest.approx(0.5, rel=0.02)
    # 1.9 s from the first segment plus 2.0 s from the second, at 10 fps.
    assert _frames(output / "video.mp4") == pytest.approx(39, abs=2)
    assert extract_frame(output / "video.mp4", output / "before.jpg", last=False)
    assert extract_frame(output / "video.mp4", output / "after.jpg", last=True)


def test_cut_video_reports_an_empty_window(tmp_path):
    report = cut_video(tmp_path, 0, 10, tmp_path / "video.mp4")
    assert report["error"] == "no recorded video in the attempt window"


def test_perception_tracks_simulator_time_for_recordings():
    perception = object.__new__(Perception)
    perception._lock = __import__("threading").RLock()
    perception._episode = 0
    perception._sim_time_s = None
    perception.update_status({"episode_index": 0, "sim_time": 12.5})
    assert perception._sim_time_s == 12.5
    perception.update_status({"episode_index": 0, "sim_time": True})
    assert perception._sim_time_s == 12.5


def test_simulator_status_bytes_and_string_payloads_decode_to_json():
    import pyarrow as pa
    from perception import decode_text

    payload = json.dumps({"episode_index": 0, "sim_time": 1.5})
    assert json.loads(decode_text(pa.array(list(payload.encode()), type=pa.uint8()))) == {"episode_index": 0, "sim_time": 1.5}
    assert json.loads(decode_text(pa.array([payload]))) == {"episode_index": 0, "sim_time": 1.5}


@needs_ffmpeg
@needs_ffprobe
def test_execution_trigger_records_only_during_execution_plus_stop_delay(tmp_path):
    trigger = tmp_path / "active-execution.json"
    recordings = tmp_path / "recordings"
    recorder = VideoRecorder(recordings, fps=10, retention_s=3600, max_queue=256, trigger_file=trigger, stop_delay_s=1.0)
    frame = np.zeros((16, 16, 3), dtype=np.uint8)
    base = time.time_ns()
    offered = lambda index: recorder.offer(frame, "session_a", None, now_ns=base + index * 100_000_000)
    assert not any(offered(i) for i in range(0, 10))  # before execute_task: nothing recorded
    trigger.write_text("{}")
    assert all(offered(i) for i in range(10, 30))  # during execution
    trigger.unlink()
    assert all(offered(i) for i in range(30, 40))  # stop_delay_s after the execution ended
    assert not any(offered(i) for i in range(41, 60))  # stopped
    recorder.close()
    segments = list(recordings.glob("session_a/*.mp4"))
    assert len(segments) == 1 and int(segments[0].stem) == base + 10 * 100_000_000
    assert _frames(segments[0]) == 30
    assert recorder._segment is None


def test_stale_execution_trigger_is_ignored(tmp_path):
    trigger = tmp_path / "active-execution.json"
    trigger.write_text("{}")
    old = time.time_ns() - 3600 * 10**9
    os.utime(trigger, ns=(old, old))
    recorder = VideoRecorder(tmp_path / "recordings", fps=10, retention_s=3600, ffmpeg="/bin/true", trigger_file=trigger, max_active_s=1800)
    assert not recorder.offer(np.zeros((8, 8, 3), dtype=np.uint8), "session_a", None)
