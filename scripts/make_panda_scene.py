"""Generate a multi-object / multi-region scene for the Franka Panda (7-DOF).

Reads Menagerie ``franka-panda/panda.xml`` and appends objects/regions +
a floor, writing ``tabletop_panda.xml`` in the same directory (meshdir resolves).

Usage: python scripts/make_panda_scene.py
"""
from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DIR = ROOT / "skill-src/rekep-sim/assets/franka-panda"
SRC = DIR / "panda.xml"
DST = DIR / "tabletop_panda.xml"

OBJECTS = """
    <light name="fill" pos="0.4 -0.2 1.2" dir="0 0.2 -0.8" directional="true" />
    <geom name="table_floor" type="plane" size="0 0 0.05" rgba="0.35 0.35 0.4 1" />
    <body name="red_cube" pos="0.45 -0.15 0.025">
      <freejoint name="red_cube_free" />
      <geom name="red_cube_geom" type="box" size="0.02 0.02 0.02" mass="0.05" rgba="0.9 0.12 0.12 1" friction="0.8 0.5 0.5" contype="2" conaffinity="5" />
    </body>
    <body name="blue_cube" pos="0.55 0.05 0.025">
      <freejoint name="blue_cube_free" />
      <geom name="blue_cube_geom" type="box" size="0.02 0.02 0.02" mass="0.05" rgba="0.12 0.25 0.9 1" friction="0.8 0.5 0.5" contype="2" conaffinity="5" />
    </body>
    <body name="pen" pos="0.50 -0.05 0.02" euler="0 1.5708 0">
      <freejoint name="pen_free" />
      <geom name="pen_geom" type="capsule" size="0.008 0.05" mass="0.02" rgba="0.95 0.95 0.95 1" friction="0.6 0.5 0.5" contype="2" conaffinity="5" />
    </body>
    <body name="holder" pos="0.45 0.15 0.0">
      <geom name="holder_wall_xp" type="box" pos="0.022 0 0.02" size="0.003 0.024 0.02" rgba="0.2 0.2 0.2 1" contype="2" conaffinity="5" />
      <geom name="holder_wall_xn" type="box" pos="-0.022 0 0.02" size="0.003 0.024 0.02" rgba="0.2 0.2 0.2 1" contype="2" conaffinity="5" />
      <geom name="holder_wall_yp" type="box" pos="0 0.024 0.02" size="0.024 0.003 0.02" rgba="0.2 0.2 0.2 1" contype="2" conaffinity="5" />
      <geom name="holder_wall_yn" type="box" pos="0 -0.024 0.02" size="0.024 0.003 0.02" rgba="0.2 0.2 0.2 1" contype="2" conaffinity="5" />
    </body>
    <geom name="region_a" type="box" pos="0.30 0.20 0.006" size="0.04 0.04 0.003" rgba="0.9 0.1 0.1 0.45" contype="0" conaffinity="0" />
    <geom name="region_b" type="box" pos="0.62 0.20 0.006" size="0.04 0.04 0.003" rgba="0.1 0.3 0.9 0.45" contype="0" conaffinity="0" />
    <geom name="region_holder" type="box" pos="0.45 0.15 0.02" size="0.02 0.02 0.003" rgba="0.1 0.9 0.1 0.4" contype="0" conaffinity="0" />
"""


def main() -> int:
    text = SRC.read_text(encoding="utf-8")
    if "</worldbody>" not in text:
        raise SystemExit("no </worldbody> in panda.xml")
    new = text.replace("</worldbody>", OBJECTS + "  </worldbody>", 1)
    DST.write_text(new, encoding="utf-8")
    print("wrote", DST, len(new), "bytes")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
