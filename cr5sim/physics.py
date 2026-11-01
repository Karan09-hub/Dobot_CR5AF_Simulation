"""MuJoCo world: CR5 + Robotiq 2F-85 on position actuators, a table, a wall and a cube."""

from __future__ import annotations

from dataclasses import dataclass

import mujoco
import numpy as onp

from cr5sim.kinematics import HOME_CFG
from cr5sim.paths import SCENE_MJCF

ARM_JOINTS = ("joint1", "joint2", "joint3", "joint4", "joint5", "joint6")
GRIPPER_ACTUATOR = "gripper/fingers_actuator"
GRIPPER_CTRL_CLOSED = 255.0  # Robotiq command range is 0 (open) .. 255 (closed)
TCP_SITE = "gripper/pinch"


@dataclass(frozen=True)
class Box:
    """A static box obstacle (obs_* geom), in world coordinates."""

    name: str
    size: onp.ndarray  # full extents (x, y, z)
    position: onp.ndarray
    wxyz: onp.ndarray


class CR5World:
    def __init__(self) -> None:
        spec = mujoco.MjSpec.from_file(str(SCENE_MJCF))
        # The arm bodies have gravcomp (like the CR5 controller's gravity feed-forward);
        # give the gripper the same, as if its payload were configured on the robot.
        for body in spec.bodies:
            if body.name.startswith("gripper/"):
                body.gravcomp = 1.0
        self.model = spec.compile()
        self.data = mujoco.MjData(self.model)
        self._check_data = mujoco.MjData(self.model)  # scratch data for kinematic checks

        m = self.model
        self.arm_qpos_adr = onp.array([m.joint(n).qposadr[0] for n in ARM_JOINTS])
        self.arm_ctrl = onp.array([m.actuator(n).id for n in ARM_JOINTS])
        self.gripper_ctrl = m.actuator(GRIPPER_ACTUATOR).id
        base = m.body("base_link").id
        self.robot_bodies = {b for b in range(m.nbody) if m.body_rootid[b] == base}
        self.cube_qpos_adr = m.joint("cube").qposadr[0]
        self.cube_qpos0 = m.qpos0[self.cube_qpos_adr : self.cube_qpos_adr + 7].copy()
        self.reset()

    # -------------------------------------------------------------- state

    def reset(self, cube_xyz: onp.ndarray | None = None) -> None:
        mujoco.mj_resetData(self.model, self.data)
        self.data.qpos[self.arm_qpos_adr] = HOME_CFG
        self.data.ctrl[self.arm_ctrl] = HOME_CFG
        if cube_xyz is not None:
            self.data.qpos[self.cube_qpos_adr : self.cube_qpos_adr + 3] = cube_xyz
        mujoco.mj_forward(self.model, self.data)

    @property
    def time(self) -> float:
        return float(self.data.time)

    @property
    def q(self) -> onp.ndarray:
        return self.data.qpos[self.arm_qpos_adr].copy()

    @property
    def q_target(self) -> onp.ndarray:
        return self.data.ctrl[self.arm_ctrl].copy()

    def set_arm_target(self, q: onp.ndarray) -> None:
        self.data.ctrl[self.arm_ctrl] = q

    def set_gripper(self, closed_fraction: float) -> None:
        """0 = fully open, 1 = fully closed (stops on contact, force-limited)."""
        self.data.ctrl[self.gripper_ctrl] = float(onp.clip(closed_fraction, 0, 1)) * GRIPPER_CTRL_CLOSED

    @property
    def gripper_command(self) -> float:
        return float(self.data.ctrl[self.gripper_ctrl] / GRIPPER_CTRL_CLOSED)

    def tcp_pose(self) -> tuple[onp.ndarray, onp.ndarray]:
        site = self.data.site(TCP_SITE)
        wxyz = onp.zeros(4)
        mujoco.mju_mat2Quat(wxyz, site.xmat)
        return site.xpos.copy(), wxyz

    def cube_position(self) -> onp.ndarray:
        return self.data.body("cube").xpos.copy()

    def step(self, n: int = 1) -> None:
        mujoco.mj_step(self.model, self.data, nstep=n)

    # ---------------------------------------------------------- obstacles

    def obstacles(self) -> list[Box]:
        m, d = self.model, self.data
        boxes = []
        for g in range(m.ngeom):
            name = m.geom(g).name
            if name.startswith("obs_") and m.geom_type[g] == mujoco.mjtGeom.mjGEOM_BOX:
                wxyz = onp.zeros(4)
                mujoco.mju_mat2Quat(wxyz, d.geom_xmat[g])
                boxes.append(Box(name, 2.0 * m.geom_size[g], d.geom_xpos[g].copy(), wxyz))
        return boxes

    def _static_contacts(self, data: mujoco.MjData) -> list[str]:
        """Names of static geoms (table, wall, floor) touching the robot."""
        m = self.model
        hits = []
        for c in data.contact[: data.ncon]:
            b1, b2 = m.geom_bodyid[c.geom1], m.geom_bodyid[c.geom2]
            for robot_b, other_g, other_b in ((b1, c.geom2, b2), (b2, c.geom1, b1)):
                if robot_b in self.robot_bodies and other_b == 0:
                    hits.append(m.geom(other_g).name)
        return hits

    def robot_static_contacts(self) -> list[str]:
        return self._static_contacts(self.data)

    def count_trajectory_collisions(self, q_traj: onp.ndarray) -> int:
        """Kinematic check: number of samples where the robot touches a static obstacle.

        The gripper is checked in its current state; the cube is ignored.
        """
        d = self._check_data
        d.qpos[:] = self.data.qpos
        hits = 0
        for q in q_traj:
            d.qpos[self.arm_qpos_adr] = q
            mujoco.mj_forward(self.model, d)
            hits += bool(self._static_contacts(d))
        return hits
