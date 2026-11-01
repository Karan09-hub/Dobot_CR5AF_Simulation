"""Tests for Dobot CR5 + Robotiq 2F-85 Simulation (cr5sim).

Tests:
    - URDF validity & link hierarchy
    - MuJoCo physics model loading & world state
    - Forward and Inverse Kinematics (PyRoKi)
    - CPU planner & Cartesian path generation

Run:
    pytest tests -v
"""

from __future__ import annotations

import jaxlie
import mujoco
import numpy as onp
import pytest
import yourdfpy

from cr5sim import paths
from cr5sim.kinematics import EE_LINK, HOME_CFG, CR5Kinematics
from cr5sim.physics import ARM_JOINTS, CR5World
from cr5sim.planner import cartesian_path, make_planner


def test_urdf_exists_and_loads():
    """Verify that cr5_robot.urdf exists and parses correctly."""
    assert paths.URDF.is_file(), f"URDF not found at {paths.URDF}"
    model = yourdfpy.URDF.load(str(paths.URDF))
    assert EE_LINK in model.link_map, f"Expected {EE_LINK} link in URDF"
    assert "gripper" in model.link_map, "Expected gripper link in URDF"
    assert len(model.actuated_joints) == 6, f"Expected 6 arm joints, got {len(model.actuated_joints)}"


def test_mujoco_scene_loads():
    """Verify that MuJoCo scene XML parses and compiles."""
    assert paths.SCENE_MJCF.is_file(), f"Scene MJCF not found at {paths.SCENE_MJCF}"
    model = mujoco.MjModel.from_xml_path(str(paths.SCENE_MJCF))
    assert model.nq >= 6
    assert model.nu >= 7  # 6 arm joints + 1 gripper actuator


def test_cr5_world_init():
    """Verify CR5World physics wrapper initializes and sets home pose."""
    world = CR5World()
    assert world.model is not None
    assert world.data is not None

    # Check TCP pose is finite
    pos, wxyz = world.tcp_pose()
    assert len(pos) == 3 and onp.all(onp.isfinite(pos))
    assert len(wxyz) == 4 and onp.all(onp.isfinite(wxyz))

    # Check obstacles detected
    assert len(world.obstacles) > 0, "Expected static obstacles in the scene"
    obs_names = [o.name for o in world.obstacles]
    assert any("wall" in name or "table" in name or "obs" in name for name in obs_names)


def test_kinematics_fk_and_ik():
    """Verify PyRoKi kinematics forward and inverse solutions."""
    kin = CR5Kinematics()

    # Test forward kinematics on home pose
    poses = kin.forward_kinematics(HOME_CFG)
    assert poses.shape[0] == len(kin.robot.links.names)

    # Get home TCP pose
    tcp_idx = kin.robot.links.names.index(EE_LINK)
    home_pose = jaxlie.SE3(poses[tcp_idx])
    home_pos = onp.asarray(home_pose.translation())
    home_wxyz = onp.asarray(home_pose.rotation().wxyz)

    # Test inverse kinematics on home pose
    q_sol = kin.solve_ik(home_pos, home_wxyz, HOME_CFG)
    assert q_sol.shape == (6,)
    assert onp.all(q_sol >= kin.lower - 1e-3)
    assert onp.all(q_sol <= kin.upper + 1e-3)

    # Check solution error
    sol_poses = kin.forward_kinematics(q_sol)
    sol_tcp = jaxlie.SE3(sol_poses[tcp_idx])
    pos_err = onp.linalg.norm(sol_tcp.translation() - home_pos)
    assert pos_err < 1e-3, f"IK position error too high: {pos_err * 1000:.2f} mm"


def test_cartesian_path_generation():
    """Verify linear Cartesian path interpolation."""
    p_start = onp.array([0.3, 0.0, 0.4])
    p_goal = onp.array([0.4, 0.1, 0.5])
    wxyz = onp.array([1.0, 0.0, 0.0, 0.0])

    path = cartesian_path(p_start, wxyz, p_goal, wxyz, speed=0.1, dt=0.02)
    assert len(path) > 2
    assert onp.allclose(path[0][0], p_start)
    assert onp.allclose(path[-1][0], p_goal)


def test_cpu_planner():
    """Verify CPU fallback planner generates valid trajectory."""
    world = CR5World()
    kin = CR5Kinematics()
    planner = make_planner(kin, world, kind="cpu")

    p_start, wxyz = world.tcp_pose()
    p_goal = p_start + onp.array([0.05, 0.05, 0.0])

    plan = planner.plan(HOME_CFG, p_goal, wxyz)
    assert plan is not None
    assert plan.q.shape[1] == 6
    assert plan.planner == "cpu"
    assert len(plan.q) > 0
