"""Convert the CR5 DAE meshes to OBJ and generate a MuJoCo MJCF from the URDF.

Outputs:
    meshes/obj/<Link>_<i>.obj        one OBJ per colored DAE part (visual)
    meshes/obj/<Link>_collision.obj  all parts merged (MuJoCo uses its convex hull)
    mjcf/cr5.xml                     arm + Robotiq 2F-85 (attached from third_party/)
    (mjcf/scene.xml is hand-written and attaches cr5.xml)

Run tools/build_urdf.py first.

Run:
    .venv/bin/python tools/build_mujoco_model.py
"""

from __future__ import annotations

import os
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import trimesh
from scipy.spatial.transform import Rotation

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from cr5sim.paths import DAE_DIR, GRIPPER_MJCF, OBJ_DIR, URDF as URDF_PATH  # noqa: E402
from cr5sim.paths import ROBOT_MJCF as MJCF_PATH  # noqa: E402

ARM_LINKS = ("base_link", "Link1", "Link2", "Link3", "Link4", "Link5", "Link6")


def convert_link(link: str) -> list[tuple[str, tuple[float, ...]]]:
    """Write OBJ files for one link; returns [(mesh_name, rgba), ...] for visual parts."""
    scene = trimesh.load(DAE_DIR / f"{link}.dae", force="scene")
    parts: list[tuple[str, tuple[float, ...]]] = []
    merged = []
    for i, node in enumerate(scene.graph.nodes_geometry):
        transform, geom_name = scene.graph[node]
        mesh = scene.geometry[geom_name].copy()
        mesh.apply_transform(transform)
        rgba = tuple(float(c) / 255.0 for c in mesh.visual.material.main_color)
        name = f"{link}_{i}"
        # Plain geometry only: MuJoCo ignores .mtl files, colors go on the geom.
        trimesh.Trimesh(mesh.vertices, mesh.faces, process=False).export(OBJ_DIR / f"{name}.obj")
        parts.append((name, rgba))
        merged.append(trimesh.Trimesh(mesh.vertices, mesh.faces))
    trimesh.util.concatenate(merged).export(OBJ_DIR / f"{link}_collision.obj")
    return parts


def fmt(values) -> str:
    return " ".join(f"{float(v):.6g}" for v in values)


def build_mjcf(visual_parts: dict[str, list[tuple[str, tuple[float, ...]]]]) -> ET.Element:
    urdf = ET.parse(URDF_PATH).getroot()
    links = {l.get("name"): l for l in urdf.findall("link")}
    child_joint = {j.find("child").get("link"): j for j in urdf.findall("joint")}

    mujoco = ET.Element("mujoco", model="cr5")
    ET.SubElement(mujoco, "compiler", angle="radian", meshdir="../meshes/obj", autolimits="true")
    # Same contact settings as the Robotiq model (needed for non-slip grasps).
    ET.SubElement(mujoco, "option", cone="elliptic", impratio="10")
    default = ET.SubElement(mujoco, "default")
    cr5 = ET.SubElement(default, "default", {"class": "cr5"})
    ET.SubElement(cr5, "joint", damping="2", armature="0.1", frictionloss="0.1")
    ET.SubElement(cr5, "position", kp="3000", kv="150", forcerange="-300 300")
    visual = ET.SubElement(cr5, "default", {"class": "visual"})
    ET.SubElement(visual, "geom", type="mesh", contype="0", conaffinity="0", group="2")
    collision = ET.SubElement(cr5, "default", {"class": "collision"})
    ET.SubElement(collision, "geom", type="mesh", group="3", rgba="0.2 0.6 1 0.3")

    asset = ET.SubElement(mujoco, "asset")
    gripper_file = os.path.relpath(GRIPPER_MJCF, MJCF_PATH.parent)
    ET.SubElement(asset, "model", name="robotiq_2f85", file=gripper_file)
    for link in ARM_LINKS:
        for name, _ in visual_parts[link]:
            # Some visual parts are flat trim rings; their mass comes from <inertial> anyway.
            ET.SubElement(asset, "mesh", name=name, file=f"{name}.obj", inertia="shell")
        ET.SubElement(asset, "mesh", name=f"{link}_collision", file=f"{link}_collision.obj")

    worldbody = ET.SubElement(mujoco, "worldbody")
    actuator = ET.SubElement(mujoco, "actuator")
    parent = worldbody
    for link in ARM_LINKS:
        # gravcomp mimics the gravity feed-forward of the real CR5 controller.
        attrs = {"name": link, "childclass": "cr5", "gravcomp": "1"}
        joint = child_joint.get(link)
        if joint is not None and joint.get("type") == "revolute":
            origin = joint.find("origin")
            attrs["pos"] = fmt(origin.get("xyz").split())
            rpy = [float(v) for v in origin.get("rpy").split()]
            # MuJoCo quats are wxyz; URDF rpy is extrinsic XYZ.
            x, y, z, w = Rotation.from_euler("xyz", rpy).as_quat()
            attrs["quat"] = fmt([w, x, y, z])
        body = ET.SubElement(parent, "body", attrs)

        inertial = links[link].find("inertial")
        I = inertial.find("inertia").attrib
        ET.SubElement(
            body,
            "inertial",
            pos=fmt(inertial.find("origin").get("xyz").split()),
            mass=inertial.find("mass").get("value"),
            fullinertia=fmt([I[k] for k in ("ixx", "iyy", "izz", "ixy", "ixz", "iyz")]),
        )
        if joint is not None and joint.get("type") == "revolute":
            limit = joint.find("limit")
            name = joint.get("name")
            rng = fmt([limit.get("lower"), limit.get("upper")])
            ET.SubElement(body, "joint", name=name, axis=fmt(joint.find("axis").get("xyz").split()), range=rng)
            ET.SubElement(actuator, "position", name=name, joint=name, ctrlrange=rng, **{"class": "cr5"})
        for name, rgba in visual_parts[link]:
            ET.SubElement(body, "geom", mesh=name, rgba=fmt(rgba), **{"class": "visual"})
        ET.SubElement(body, "geom", name=f"{link}_collision", mesh=f"{link}_collision", **{"class": "collision"})
        if link == "Link6":
            ET.SubElement(body, "site", name="flange", pos="0 0 0", size="0.005")
            # Flange face = Link6 origin, +z out: the gripper mounts with no offset.
            ET.SubElement(body, "attach", model="robotiq_2f85", body="base_mount", prefix="gripper/")
        parent = body

    # base_link is welded to the world, so MuJoCo does not filter its contacts with
    # Link1. The other pairs' convex hulls overlap in every configuration.
    contact = ET.SubElement(mujoco, "contact")
    for a, b in (("base_link", "Link1"), ("base_link", "Link2"), ("Link3", "Link5"), ("Link4", "Link6")):
        ET.SubElement(contact, "exclude", body1=a, body2=b)
    return mujoco


def main() -> None:
    OBJ_DIR.mkdir(parents=True, exist_ok=True)
    MJCF_PATH.parent.mkdir(parents=True, exist_ok=True)
    parts = {link: convert_link(link) for link in ARM_LINKS}
    tree = ET.ElementTree(build_mjcf(parts))
    ET.indent(tree, space="  ")
    tree.write(MJCF_PATH, encoding="unicode")
    print(f"Wrote {sum(len(p) for p in parts.values())} visual OBJs to {OBJ_DIR}")
    print(f"Wrote {MJCF_PATH}")


if __name__ == "__main__":
    main()
