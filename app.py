"""Dobot CR5 + Robotiq 2F-85 simulator: one app for IK, physics, planning and pick & place.

    PyRoKi  - IK on the gripper TCP (interactive target, Cartesian moves)
    MuJoCo  - physics: position-controlled arm, articulated gripper, contacts
    cuRobo  - collision-free motion planning on the GPU (CPU fallback without one)
    Viser   - browser UI showing the live MuJoCo state

Run:
    .venv/bin/python app.py                  # http://localhost:8080
    .venv/bin/python app.py --planner cpu    # force the CPU planner
    .venv/bin/python app.py --mujoco-viewer  # also open MuJoCo's native viewer
"""

from __future__ import annotations

import time
from collections.abc import Iterator
from typing import Literal

from cr5sim.env_check import require_venv

require_venv()

import mujoco
import mujoco.viewer
import numpy as onp
import tyro
import viser
from loguru import logger

from cr5sim.kinematics import HOME_CFG, CR5Kinematics
from cr5sim.physics import CR5World
from cr5sim.planner import Plan, cartesian_path, make_planner
from cr5sim.viser_scene import MujocoViserScene

CONTROL_DT = 0.02  # control / UI tick (s)
APPROACH_HEIGHT = 0.12  # pre-grasp / pre-place height above the object (m)
GRASP_OFFSET = 0.008  # TCP above the cube center at grasp, keeps fingertips off the table
CUBE_HALF = 0.02

Task = Iterator[None]


class TaskFailed(Exception):
    pass


class App:
    def __init__(self, port: int, planner: str, mujoco_viewer: bool) -> None:
        self.world = CR5World()
        self.kin = CR5Kinematics()
        self.substeps = round(CONTROL_DT / self.world.model.opt.timestep)
        self.home_tcp, self.down_wxyz = self.world.tcp_pose()
        logger.info("Compiling IK...")
        self.kin.solve_ik(self.home_tcp, self.down_wxyz, HOME_CFG)
        self.planner = make_planner(self.kin, self.world, planner)

        self.server = viser.ViserServer(port=port)
        self.server.scene.set_up_direction("+z")
        self.server.initial_camera.position = (0.45, 0.95, 0.8)
        self.server.initial_camera.look_at = (-0.35, 0.0, 0.25)
        self.scene = MujocoViserScene(self.server, self.world.model, self.world.data)
        self.target = self.server.scene.add_transform_controls(
            "/target", scale=0.15, position=self.home_tcp, wxyz=self.down_wxyz
        )
        self.ik_cfg = HOME_CFG.copy()
        self.task: Task | None = None
        self.pending: str | None = None  # action requested from a GUI callback thread
        self.viewer = mujoco.viewer.launch_passive(self.world.model, self.world.data) if mujoco_viewer else None
        self._build_gui()

    # ------------------------------------------------------------------ GUI

    def _build_gui(self) -> None:
        gui = self.server.gui
        with gui.add_folder("Status"):
            gui.add_text("Planner", self.planner.name, disabled=True)
            self.status = gui.add_text("Task", "idle", disabled=True)
            self.contacts = gui.add_text("Obstacle contact", "none", disabled=True)
            self.tcp_text = gui.add_text("TCP (mm)", "", disabled=True)
            self.rtf = gui.add_number("Real-time factor", 1.0, disabled=True)
        with gui.add_folder("Control"):
            self.mode = gui.add_dropdown("Mode", ("IK target", "Joint sliders"), initial_value="IK target")
            for label, action in (
                ("Plan & move to target", "plan"),
                ("Pick & place cube", "pick_place"),
                ("Home", "home"),
                ("Reset scene", "reset"),
            ):
                button = gui.add_button(label)
                button.on_click(lambda _, a=action: setattr(self, "pending", a))
            self.show_path = gui.add_checkbox("Show planned path", True)
        with gui.add_folder("Gripper (Robotiq 2F-85)"):
            self.gripper = gui.add_slider("Close", min=0.0, max=1.0, step=0.01, initial_value=0.0)
        with gui.add_folder("Joints (deg)", expand_by_default=False):
            self.sliders = [
                gui.add_slider(n, min=float(onp.rad2deg(lo)), max=float(onp.rad2deg(hi)), step=0.1,
                               initial_value=float(onp.rad2deg(q)))
                for n, lo, hi, q in zip(self.kin.joint_names, self.kin.lower, self.kin.upper, HOME_CFG)
            ]

    def _set_status(self, text: str) -> None:
        self.status.value = text
        logger.info(text)

    def _sync_controls_to_robot(self) -> None:
        """After a task, make the interactive controls hold the current pose."""
        self.ik_cfg = self.world.q_target
        pos, wxyz = self.kin_tcp(self.ik_cfg)
        self.target.position, self.target.wxyz = pos, wxyz
        for slider, q in zip(self.sliders, self.ik_cfg):
            slider.value = float(onp.rad2deg(q))
        self.gripper.value = round(self.world.gripper_command, 2)

    def kin_tcp(self, q: onp.ndarray) -> tuple[onp.ndarray, onp.ndarray]:
        pose = self.kin.fk_ee(q)
        return onp.asarray(pose.translation()), onp.asarray(pose.rotation().wxyz)

    # ---------------------------------------------------------------- tasks

    def _wait(self, seconds: float) -> Task:
        for _ in range(int(seconds / CONTROL_DT)):
            yield

    def _follow(self, plan: Plan, label: str) -> Task:
        hits = self.world.count_trajectory_collisions(plan.q)
        self._set_status(
            f"{label}: {plan.planner}, {plan.plan_ms:.0f} ms, {len(plan.q) * plan.dt:.1f}s"
            + (f", {hits} colliding samples!" if hits else "")
        )
        if self.show_path.value:
            pts = onp.array([self.kin_tcp(q)[0] for q in plan.q[::2]])
            if len(pts) > 1:
                self.server.scene.add_line_segments(
                    "/path", onp.stack([pts[:-1], pts[1:]], axis=1), colors=(255, 140, 0), thickness=3.0, thickness_units="screen"
                )
        stride = max(1, round(CONTROL_DT / plan.dt))
        for q in plan.q[::stride]:
            self.world.set_arm_target(q)
            yield
        self.world.set_arm_target(plan.q[-1])
        yield from self._wait(0.2)

    def _planned_move(self, pos: onp.ndarray, wxyz: onp.ndarray, label: str, carrying: bool = False) -> Task:
        self._set_status(f"{label}: planning...")
        yield
        plan = self.planner.plan(self.world.q_target, pos, wxyz, self.world, carrying=carrying)
        if plan is None:
            raise TaskFailed(f"{label}: no collision-free plan found")
        yield from self._follow(plan, label)

    def _linear_move(self, pos: onp.ndarray, label: str) -> Task:
        plan = cartesian_path(self.kin, self.world.q_target, pos, self.down_wxyz, speed=0.08)
        if plan is None:
            raise TaskFailed(f"{label}: target unreachable")
        yield from self._follow(plan, label)

    def _gripper(self, closed: float, label: str) -> Task:
        self._set_status(label)
        self.world.set_gripper(closed)
        yield from self._wait(0.8)

    def task_plan_to_target(self) -> Task:
        yield from self._planned_move(onp.asarray(self.target.position), onp.asarray(self.target.wxyz), "target")

    def task_home(self) -> Task:
        yield from self._planned_move(self.home_tcp, self.down_wxyz, "home")

    def task_pick_and_place(self) -> Task:
        cube = self.world.cube_position()
        place = self.world.data.site("place_target").xpos + [0.0, 0.0, CUBE_HALF + 0.003]
        grasp = cube + [0.0, 0.0, GRASP_OFFSET]
        above = onp.array([0.0, 0.0, APPROACH_HEIGHT])

        yield from self._gripper(0.0, "open gripper")
        yield from self._planned_move(grasp + above, self.down_wxyz, "pre-grasp")
        yield from self._linear_move(grasp, "descend")
        yield from self._gripper(1.0, "grasp")
        yield from self._linear_move(grasp + above, "lift")
        if onp.linalg.norm(self.world.cube_position() - (grasp + above - [0, 0, GRASP_OFFSET])) > 0.02:
            raise TaskFailed("lift: the cube slipped out of the gripper")
        yield from self._planned_move(place + [0, 0, GRASP_OFFSET] + above, self.down_wxyz, "transfer",
                                      carrying=True)
        yield from self._linear_move(place + [0, 0, GRASP_OFFSET], "lower")
        yield from self._gripper(0.0, "release")
        yield from self._linear_move(place + [0, 0, GRASP_OFFSET] + above, "retreat")
        yield from self._planned_move(self.home_tcp, self.down_wxyz, "home")
        error = onp.linalg.norm(self.world.cube_position()[:2] - place[:2]) * 1000
        self._set_status(f"pick & place done: cube {error:.1f} mm from the target")

    # ------------------------------------------------------------ main loop

    def _start_pending(self) -> None:
        action, self.pending = self.pending, None
        if action == "reset":
            self.task = None
            self.world.reset()
            self.server.scene.remove_by_name("/path")
            self._sync_controls_to_robot()
            self._set_status("scene reset")
        elif self.task is not None:
            self._set_status(f"busy: ignoring '{action}'")
        elif action is not None:
            self.task = {
                "plan": self.task_plan_to_target,
                "pick_place": self.task_pick_and_place,
                "home": self.task_home,
            }[action]()

    def tick(self) -> None:
        if self.pending:
            self._start_pending()
        if self.task is not None:
            try:
                next(self.task)
            except StopIteration:
                self.task = None
                self._sync_controls_to_robot()
            except TaskFailed as e:
                self.task = None
                self._set_status(f"FAILED - {e}")
                self._sync_controls_to_robot()
        elif self.mode.value == "IK target":
            self.ik_cfg = self.kin.solve_ik(onp.asarray(self.target.position), onp.asarray(self.target.wxyz),
                                            self.ik_cfg)
            self.world.set_arm_target(self.ik_cfg)
            self.world.set_gripper(self.gripper.value)
        else:
            self.ik_cfg = onp.deg2rad([s.value for s in self.sliders])
            self.world.set_arm_target(self.ik_cfg)
            self.world.set_gripper(self.gripper.value)
        self.world.step(self.substeps)

    def render(self) -> None:
        with self.server.atomic():
            self.scene.update()
        hits = self.world.robot_static_contacts()
        self.contacts.value = ", ".join(sorted(set(hits))) if hits else "none"
        self.tcp_text.value = "  ".join(f"{v:7.1f}" for v in self.world.tcp_pose()[0] * 1000)
        if self.viewer is not None:
            self.viewer.sync()

    def run(self) -> None:
        logger.info("Ready: open the Viser URL above.")
        while self.viewer is None or self.viewer.is_running():
            start = time.perf_counter()
            self.tick()
            self.render()
            elapsed = time.perf_counter() - start
            self.rtf.value = round(0.95 * self.rtf.value + 0.05 * min(CONTROL_DT / max(elapsed, 1e-6), 1.0), 2)
            time.sleep(max(0.0, CONTROL_DT - elapsed))


def main(
    port: int = 8080,
    planner: Literal["auto", "curobo", "cpu"] = "auto",
    mujoco_viewer: bool = False,
) -> None:
    """Dobot CR5 + Robotiq 2F-85 simulator.

    Args:
        port: Viser web UI port.
        planner: auto (cuRobo if a GPU is available), curobo (require it) or cpu.
        mujoco_viewer: Also open MuJoCo's native viewer window.
    """
    App(port, planner, mujoco_viewer).run()


if __name__ == "__main__":
    tyro.cli(main)
