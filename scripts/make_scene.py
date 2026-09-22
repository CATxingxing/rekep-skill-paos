"""Generate a multi-object / multi-region MuJoCo scene from the Nova2 base scene.

Adds (all auto-enumerated by the env):
  movable: red_cube, blue_cube, pen (capsule)
  regions: region_a (red zone), region_b (blue zone), region_holder (inside holder)
  static : holder (open-top bin: 4 walls)

Usage: python scripts/make_scene.py
"""
from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "skill-src/rekep-sim/assets/dobot-nova2-robotiq/mjcf/dobot_nova2_robotiq_2f85_pick_place.xml"
DST = ROOT / "skill-src/rekep-sim/assets/dobot-nova2-robotiq/mjcf/tabletop_general.xml"

OBJECTS = """
    <body name="red_cube" pos="0.22 -0.34 0.03">
      <freejoint name="red_cube_free" />
      <geom name="red_cube_geom" type="box" size="0.02 0.02 0.02" mass="0.05" rgba="0.9 0.12 0.12 1" friction="0.8 0.5 0.5" contype="2" conaffinity="5" />
    </body>
    <body name="blue_cube" pos="0.32 -0.44 0.03">
      <freejoint name="blue_cube_free" />
      <geom name="blue_cube_geom" type="box" size="0.02 0.02 0.02" mass="0.05" rgba="0.12 0.25 0.9 1" friction="0.8 0.5 0.5" contype="2" conaffinity="5" />
    </body>
    <body name="pen" pos="0.20 -0.44 0.02" euler="0 1.5708 0">
      <freejoint name="pen_free" />
      <geom name="pen_geom" type="capsule" size="0.008 0.05" mass="0.02" rgba="0.95 0.95 0.95 1" friction="0.6 0.5 0.5" contype="2" conaffinity="5" />
    </body>
    <body name="holder" pos="0.28 -0.30 0.0">
      <geom name="holder_wall_xp" type="box" pos="0.022 0 0.02" size="0.003 0.024 0.02" rgba="0.2 0.2 0.2 1" contype="2" conaffinity="5" />
      <geom name="holder_wall_xn" type="box" pos="-0.022 0 0.02" size="0.003 0.024 0.02" rgba="0.2 0.2 0.2 1" contype="2" conaffinity="5" />
      <geom name="holder_wall_yp" type="box" pos="0 0.024 0.02" size="0.024 0.003 0.02" rgba="0.2 0.2 0.2 1" contype="2" conaffinity="5" />
      <geom name="holder_wall_yn" type="box" pos="0 -0.024 0.02" size="0.024 0.003 0.02" rgba="0.2 0.2 0.2 1" contype="2" conaffinity="5" />
    </body>
    <geom name="region_a" type="box" pos="0.10 -0.44 0.006" size="0.035 0.035 0.003" rgba="0.9 0.1 0.1 0.45" contype="0" conaffinity="0" />
    <geom name="region_b" type="box" pos="0.10 -0.30 0.006" size="0.035 0.035 0.003" rgba="0.1 0.3 0.9 0.45" contype="0" conaffinity="0" />
    <geom name="region_holder" type="box" pos="0.28 -0.30 0.02" size="0.02 0.02 0.003" rgba="0.1 0.9 0.1 0.4" contype="0" conaffinity="0" />
"""


def main() -> int:
    text = SRC.read_text(encoding="utf-8")
    marker = "  </worldbody>"
    if marker not in text:
        raise SystemExit("worldbody marker not found")
    new = text.replace(
        "  <worldbody>",
        "  <worldbody>\n    <light name=\"table_fill\" pos=\"0 -0.2 1.2\" dir=\"0 0.2 -0.8\" directional=\"true\" />",
        1,
    ).replace(marker, OBJECTS + marker, 1)
    DST.write_text(new, encoding="utf-8")
    print("wrote", DST, len(new), "bytes")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
