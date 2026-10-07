"""Build task MJCF files from the baseline Nova2 + Robotiq scene.

The baseline file (``dobot_nova2_robotiq_2f85_pick_place.xml``) is read, never
modified.  Its demo cube, platforms and green zone are removed and replaced by
the task's table and objects.  The robot, actuators, gravity compensation,
timestep, lights and the ``rekep_overview`` camera (hence the perception
calibration) are inherited unchanged.
"""
from __future__ import annotations

import xml.etree.ElementTree as ET
from pathlib import Path

from .config import BASE_SCENE

REMOVE_FROM_WORLDBODY = {"pick_cube", "pick_platform", "place_platform", "place_zone"}


def fmt(values) -> str:
    return " ".join(f"{float(v):.6g}" for v in values)


def geom(parent: ET.Element, name: str, gtype: str, size, pos=(0, 0, 0), rgba=None, mass=None,
         friction=None, contype=None, conaffinity=None, condim=None, quat=None, extra=None) -> ET.Element:
    attrs = {"name": name, "type": gtype, "size": fmt(size), "pos": fmt(pos)}
    if quat is not None:
        attrs["quat"] = fmt(quat)
    if rgba is not None:
        attrs["rgba"] = fmt(rgba)
    if mass is not None:
        attrs["mass"] = f"{mass:.6g}"
    if friction is not None:
        attrs["friction"] = fmt(friction)
    if contype is not None:
        attrs["contype"] = str(contype)
    if conaffinity is not None:
        attrs["conaffinity"] = str(conaffinity)
    if condim is not None:
        attrs["condim"] = str(condim)
    if extra:
        attrs.update(extra)
    return ET.SubElement(parent, "geom", attrs)


def free_body(parent: ET.Element, name: str, pos, quat) -> ET.Element:
    body = ET.SubElement(parent, "body", {"name": name, "pos": fmt(pos), "quat": fmt(quat)})
    ET.SubElement(body, "freejoint", {"name": f"{name}_free"})
    return body


def build_mjcf(task_fill, common: dict, base_scene: Path = BASE_SCENE) -> str:
    """Return MJCF text.  ``task_fill(worldbody)`` appends the task's bodies/geoms."""
    tree = ET.parse(base_scene)
    root = tree.getroot()
    root.set("model", root.get("model", "scene") + "_task")
    root.find("compiler").set("meshdir", str(base_scene.parent.resolve()))
    world = root.find("worldbody")
    for child in list(world):
        if child.get("name") in REMOVE_FROM_WORLDBODY:
            world.remove(child)
    for key in root.findall("keyframe"):
        root.remove(key)

    table, col = common["table"], common["collision"]["table"]
    cx, cy = table["center_xy"]
    hx, hy = table["half_size_xy"]
    half_z = table["top_z"] / 2.0
    geom(world, "table", "box", (hx, hy, half_z), pos=(cx, cy, half_z), rgba=(0.30, 0.30, 0.35, 1.0),
         friction=table["friction"], contype=col["contype"], conaffinity=col["conaffinity"])
    task_fill(world)
    ET.indent(tree, space="  ")
    return ET.tostring(root, encoding="unicode")
