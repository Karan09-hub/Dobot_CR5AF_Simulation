"""Build the cuRobo robot config for the CR5 + Robotiq 2F-85.

Fits collision spheres to every link (gripper included), computes the self-collision
ignore matrix, then adds what the builder doesn't:
  - an `attached_object` link at the TCP with spare sphere slots, so a grasped
    object can be attached and collision-checked while it's carried,
  - the home pose as the default joint position.

Output: curobo_cfg/cr5.yml. Requires a CUDA GPU (cuRobo's URDF parser runs on CUDA).

Run:
    .venv/bin/python tools/build_curobo_robot.py
    .venv/bin/python tools/build_curobo_robot.py --visualize   # inspect spheres at :8080
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as onp
import torch
import tyro
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from cr5sim.kinematics import EE_LINK, HOME_CFG  # noqa: E402
from cr5sim.paths import CUROBO_ROBOT_CFG, URDF  # noqa: E402

ATTACHED_LINK = "attached_object"
ATTACHED_SPHERES = 8


def add_attachment_and_home(cfg_path: Path) -> None:
    data = yaml.safe_load(cfg_path.read_text())
    kin = data["robot_cfg"]["kinematics"] if "robot_cfg" in data else data["kinematics"]
    kin["extra_links"] = {
        ATTACHED_LINK: {
            "parent_link_name": EE_LINK,
            "link_name": ATTACHED_LINK,
            "joint_name": "attach_joint",
            "joint_type": "FIXED",
            "fixed_transform": [0, 0, 0, 1, 0, 0, 0],  # xyz, qw qx qy qz
        }
    }
    kin["extra_collision_spheres"] = {ATTACHED_LINK: ATTACHED_SPHERES}
    if ATTACHED_LINK not in kin["collision_link_names"]:
        kin["collision_link_names"].append(ATTACHED_LINK)
    kin.setdefault("self_collision_buffer", {})[ATTACHED_LINK] = 0.0
    # A held object touches the fingers by definition.
    kin.setdefault("self_collision_ignore", {})[ATTACHED_LINK] = ["gripper", "Link6", "Link5"]
    kin["cspace"]["default_joint_position"] = [float(q) for q in HOME_CFG]
    cfg_path.write_text(yaml.safe_dump(data, sort_keys=False))


def main(sphere_density: float = 1.0, visualize: bool = False, seed: int = 42) -> None:
    """Fit collision spheres to the CR5 + gripper meshes and save a cuRobo robot config.

    Args:
        sphere_density: Sphere budget multiplier (higher = tighter fit, slower planning).
        visualize: Serve a Viser view of the fitted spheres afterwards.
        seed: RNG seed for reproducible fits.
    """
    if not torch.cuda.is_available():
        raise SystemExit("cuRobo needs a CUDA GPU (torch.cuda.is_available() is False). Check nvidia-smi.")
    from curobo.robot_builder import RobotBuilder

    onp.random.seed(seed)
    torch.manual_seed(seed)
    print(f"GPU: {torch.cuda.get_device_name(0)}")

    builder = RobotBuilder(
        urdf_path=str(URDF),
        # Mesh paths in the URDF ("../Assets/...", "../meshes/...") are relative to urdf/.
        asset_path=str(URDF.parent),
        tool_frames=[EE_LINK],
    )
    t0 = time.time()
    builder.fit_collision_spheres(
        sphere_density=sphere_density,
        compute_metrics=True,
        # The base is bolted to the floor: keep its spheres above z=0.
        clip_links={"base_link": ("z", 0.0)},
    )
    print(f"Fitted {builder.num_spheres} spheres in {time.time() - t0:.1f}s")
    for link, m in builder.link_metrics.items():
        print(
            f"  {link:<10s} n={m.num_spheres:3d}  cover={m.coverage * 100:5.1f}%"
            f"  protrude={m.protrusion_dist_mean * 1000:5.2f}mm"
        )
    builder.compute_collision_matrix(prune_collisions=True, num_samples=1000)

    config = builder.build()
    CUROBO_ROBOT_CFG.parent.mkdir(parents=True, exist_ok=True)
    builder.save(config, str(CUROBO_ROBOT_CFG))
    add_attachment_and_home(CUROBO_ROBOT_CFG)
    print(f"Saved {CUROBO_ROBOT_CFG}")

    if visualize:
        builder.visualize(config, port=8080)
        print("Spheres at http://localhost:8080 (Ctrl+C to stop)")
        while True:
            time.sleep(1.0)


if __name__ == "__main__":
    tyro.cli(main)
