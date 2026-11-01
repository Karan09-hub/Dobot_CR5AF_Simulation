"""Motion planning: cuRobo (GPU, collision-free trajectory optimization) or a CPU fallback.

Both planners return a dense joint trajectory (`Plan`) that `cr5sim.physics` tracks.
`make_planner()` picks cuRobo when a CUDA GPU and curobo_cfg/cr5.yml are available.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import jaxlie
import numpy as onp
from loguru import logger

from cr5sim.kinematics import CR5Kinematics
from cr5sim.paths import CUROBO_ROBOT_CFG
from cr5sim.physics import Box, CR5World

DT = 0.02  # trajectory sample period (s)
CUBE_OBSTACLE = "cube"


@dataclass
class Plan:
    q: onp.ndarray  # [T, 6]
    dt: float
    planner: str
    plan_ms: float


def min_jerk(s: onp.ndarray) -> onp.ndarray:
    s = onp.clip(s, 0.0, 1.0)
    return 10 * s**3 - 15 * s**4 + 6 * s**5


def cartesian_path(
    kin: CR5Kinematics,
    q_start: onp.ndarray,
    position: onp.ndarray,
    wxyz: onp.ndarray,
    speed: float = 0.15,
    tolerance: float = 0.003,
) -> Plan | None:
    """Straight-line TCP motion (min-jerk timed), solved with PyRoKi IK at every sample.

    Returns None if any sample is unreachable within `tolerance` (m).
    """
    t0 = time.perf_counter()
    start = kin.fk_ee(q_start)
    goal = jaxlie.SE3.from_rotation_and_translation(jaxlie.SO3(onp.asarray(wxyz)), onp.asarray(position))
    p0, p1 = onp.asarray(start.translation()), onp.asarray(goal.translation())
    rot_delta = (start.rotation().inverse() @ goal.rotation()).log()
    angle = float(onp.linalg.norm(rot_delta))
    duration = max(onp.linalg.norm(p1 - p0) / speed, angle / 1.0, 0.3)
    q = q_start
    traj = []
    for s in min_jerk(onp.linspace(0.0, 1.0, int(onp.ceil(duration / DT)) + 1)):
        pos = p0 + (p1 - p0) * s
        rot = start.rotation() @ jaxlie.SO3.exp(rot_delta * s)
        q = kin.solve_ik(pos, onp.asarray(rot.wxyz), q)
        if onp.linalg.norm(onp.asarray(kin.fk_ee(q).translation()) - pos) > tolerance:
            return None
        traj.append(q)
    return Plan(onp.asarray(traj), DT, "cartesian", (time.perf_counter() - t0) * 1000.0)


def concat(plans: list[Plan], name: str) -> Plan:
    return Plan(onp.concatenate([p.q for p in plans]), DT, name, sum(p.plan_ms for p in plans))


class WaypointPlanner:
    """CPU fallback: lift above every obstacle, travel, descend (Cartesian segments).

    Collision-free for box obstacles on a table, but not optimal, and blind to anything
    the robot passes over (it only knows obstacle heights).
    """

    name = "pyroki-waypoints (CPU)"

    def __init__(self, kin: CR5Kinematics, obstacles: list[Box], clearance: float = 0.08) -> None:
        self.kin = kin
        self.safe_z = max(b.position[2] + b.size[2] / 2 for b in obstacles) + clearance

    def plan(self, q_start: onp.ndarray, position: onp.ndarray, wxyz: onp.ndarray,
             world: CR5World, carrying: bool = False) -> Plan | None:
        safe_z = self.safe_z + (0.05 if carrying else 0.0)
        p0 = onp.asarray(self.kin.fk_ee(q_start).translation())
        start_wxyz = onp.asarray(self.kin.fk_ee(q_start).rotation().wxyz)
        waypoints = [
            (onp.array([p0[0], p0[1], max(p0[2], safe_z)]), start_wxyz),
            (onp.array([position[0], position[1], max(position[2], safe_z)]), wxyz),
            (onp.asarray(position), wxyz),
        ]
        segments, q = [], q_start
        for pos, quat in waypoints:
            seg = cartesian_path(self.kin, q, pos, quat, speed=0.3)
            if seg is None:
                return None
            segments.append(seg)
            q = seg.q[-1]
        return concat(segments, self.name)


class CuroboPlanner:
    """cuRobo MotionPlanner over the scene's obs_* boxes plus the (movable) cube."""

    name = "cuRobo (GPU)"

    def __init__(self, kin: CR5Kinematics, world: CR5World) -> None:
        import torch
        from curobo.motion_planner import MotionPlanner, MotionPlannerCfg

        self.torch = torch
        self.kin = kin
        self._attached = False
        t0 = time.time()
        self.planner = MotionPlanner(
            MotionPlannerCfg.create(robot=str(CUROBO_ROBOT_CFG), scene_model=self._scene_dict(world))
        )
        self.planner.warmup(enable_graph=True, num_warmup_iterations=3)
        self.order = [kin.joint_names.index(n) for n in self.planner.joint_names]
        logger.info(f"cuRobo ready in {time.time() - t0:.1f}s")

    @staticmethod
    def _scene_dict(world: CR5World) -> dict:
        # Floor top 2 cm below z=0 so base spheres resting on it aren't in collision.
        cuboids = {"floor": {"dims": [3.0, 3.0, 0.1], "pose": [0, 0, -0.07, 1, 0, 0, 0]}}
        for box in world.obstacles():
            cuboids[box.name] = {"dims": box.size.tolist(), "pose": [*box.position, *box.wxyz]}
        cube = world.model.geom(CUBE_OBSTACLE)
        cuboids[CUBE_OBSTACLE] = {
            "dims": (2 * cube.size).tolist(),
            "pose": [*world.data.body(CUBE_OBSTACLE).xpos, *world.data.body(CUBE_OBSTACLE).xquat],
        }
        return {"cuboid": cuboids}

    def _joint_state(self, q: onp.ndarray):
        from curobo.types import JointState

        t = self.torch.tensor(q[self.order], device="cuda", dtype=self.torch.float32).unsqueeze(0)
        return JointState.from_position(t, joint_names=self.planner.joint_names)

    def plan(self, q_start: onp.ndarray, position: onp.ndarray, wxyz: onp.ndarray,
             world: CR5World, carrying: bool = False) -> Plan | None:
        from curobo.scene import Scene
        from curobo.types import GoalToolPose, Pose

        torch = self.torch
        # Sync obstacles (the cube moves). Detach first: attachment disables the cube obstacle.
        if self._attached:
            self.planner.attachment_manager.detach()
            self._attached = False
        self.planner.update_world(Scene.create(self._scene_dict(world)))
        start = self._joint_state(q_start)
        if carrying:
            # Carry the cube as part of the robot so it's kept clear of the wall too.
            self.planner.attachment_manager.attach_from_scene(start, [CUBE_OBSTACLE])
            self._attached = True

        cuda = dict(device="cuda", dtype=torch.float32)
        goal = GoalToolPose.from_poses(
            {self.kin.robot.links.names[self.kin.ee_index]: Pose(
                position=torch.tensor([position], **cuda), quaternion=torch.tensor([wxyz], **cuda))},
            ordered_tool_frames=self.planner.tool_frames,
        )
        t0 = time.perf_counter()
        result = self.planner.plan_pose(goal, start)
        plan_ms = (time.perf_counter() - t0) * 1000.0
        if result is None or not bool(result.success.any()):
            logger.warning(f"cuRobo failed after {plan_ms:.0f} ms")
            return None
        traj = result.get_interpolated_plan()
        dt = float(self.planner.trajopt_solver.config.interpolation_dt)
        q = traj.position.reshape(-1, len(self.order)).cpu().numpy()[:, onp.argsort(self.order)]
        # Resample to the common DT.
        t_src = onp.arange(len(q)) * dt
        t_dst = onp.arange(0.0, t_src[-1] + 1e-9, DT)
        q = onp.stack([onp.interp(t_dst, t_src, q[:, j]) for j in range(q.shape[1])], axis=1)
        return Plan(q, DT, self.name, plan_ms)


def make_planner(kin: CR5Kinematics, world: CR5World, prefer: str = "auto"):
    """'auto' uses cuRobo when possible; 'curobo' requires it; 'cpu' forces the fallback."""
    if prefer != "cpu":
        reason = None
        try:
            import torch

            if not torch.cuda.is_available():
                reason = "no CUDA device (check nvidia-smi)"
            elif not CUROBO_ROBOT_CFG.exists():
                reason = f"{CUROBO_ROBOT_CFG.name} missing (run tools/build_curobo_robot.py)"
        except ImportError:
            reason = "cuRobo/torch not installed"
        if reason is None:
            return CuroboPlanner(kin, world)
        if prefer == "curobo":
            raise SystemExit(f"cuRobo unavailable: {reason}")
        logger.warning(f"cuRobo unavailable: {reason}. Using the CPU waypoint planner.")
    return WaypointPlanner(kin, world.obstacles())
