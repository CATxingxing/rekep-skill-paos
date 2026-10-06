from pathlib import Path
from xml.etree import ElementTree

import yaml


ROOT = Path(__file__).resolve().parents[1]
PROFILE = ROOT / "profiles" / "sim-dobot-nova2-robotiq"


def test_gripper_parent_deadline_exceeds_controller_deadline() -> None:
    executor = yaml.safe_load((PROFILE / "executor.yaml").read_text(encoding="utf-8"))
    controller = yaml.safe_load(
        (PROFILE / "gripper-action-controller.yaml").read_text(encoding="utf-8")
    )

    assert executor["gripper_deadline_ms"] > controller["goal_timeout_ms"]
    assert executor["max_cartesian_step_m"] <= 0.025


def test_scaled_sim_gripper_has_contact_stall_configuration() -> None:
    simulator = yaml.safe_load(
        (PROFILE / "mujoco-simulator.yaml").read_text(encoding="utf-8")
    )
    controller = yaml.safe_load(
        (PROFILE / "gripper-action-controller.yaml").read_text(encoding="utf-8")
    )
    gripper = next(joint for joint in simulator["joints"] if joint["name"] == "gripper")

    assert gripper["state_scale"] > 100.0
    assert controller["allow_stalling"] is True
    assert controller["stall_velocity_threshold"] >= 1.0


def test_target_volume_centers_visible_top_face_at_rest() -> None:
    perception = yaml.safe_load((PROFILE / "perception.yaml").read_text(encoding="utf-8"))
    assert perception["targets"]["volume_height_m"] == 0.10

    scene = (
        ROOT
        / "assets"
        / "dobot-nova2-robotiq"
        / "mjcf"
        / "dobot_nova2_robotiq_2f85_pick_place.xml"
    ).read_text(encoding="utf-8")
    assert 'name="pick_platform"' in scene and 'contype="4" conaffinity="2"' in scene


def test_joint_terminal_tolerance_supports_cartesian_terminal_check() -> None:
    executor = yaml.safe_load((PROFILE / "executor.yaml").read_text(encoding="utf-8"))
    motion = yaml.safe_load((PROFILE / "motion-server.yaml").read_text(encoding="utf-8"))
    controller = yaml.safe_load(
        (PROFILE / "joint-trajectory-controller.yaml").read_text(encoding="utf-8")
    )
    assert controller["goal_position_tolerance"] <= 0.01
    assert controller["goal_time_tolerance_ms"] >= 3000
    assert motion["execution_timeout_margin_ms"] >= (
        controller["goal_time_tolerance_ms"] + 500
    )
    assert executor["move_deadline_ms"] > motion["execution_timeout_margin_ms"]


def test_sim_arm_servo_gain_supports_joint_terminal_tolerance() -> None:
    scene = ROOT / "assets" / "dobot-nova2-robotiq" / "mjcf"
    for path in (
        scene / "dobot_nova2_robotiq_2f85_pick_place.xml",
        scene / "dobot_nova2" / "nova2.xml",
    ):
        root = ElementTree.parse(path).getroot()
        actuators = {
            actuator.attrib["name"]: actuator
            for group in root.findall("actuator")
            for actuator in group.findall("position")
            if actuator.attrib.get("name", "").startswith("joint")
        }
        assert set(actuators) == {f"joint{index}" for index in range(1, 7)}
        assert all(float(actuator.attrib["kp"]) >= 2000.0 for actuator in actuators.values())
        assert all(float(actuator.attrib["kv"]) >= 70.0 for actuator in actuators.values())
