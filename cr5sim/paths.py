"""Project file locations."""

from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
XACRO = ROOT / "Assets" / "cr5_robot.xacro"
DAE_DIR = ROOT / "Assets"
URDF = ROOT / "urdf" / "cr5_robot.urdf"
OBJ_DIR = ROOT / "meshes" / "obj"
GRIPPER_MJCF = ROOT / "third_party" / "robotiq_2f85" / "2f85.xml"
ROBOT_MJCF = ROOT / "mjcf" / "cr5.xml"
SCENE_MJCF = ROOT / "mjcf" / "scene.xml"
CUROBO_ROBOT_CFG = ROOT / "curobo_cfg" / "cr5.yml"
PLAN_DIR = ROOT / "plans"
