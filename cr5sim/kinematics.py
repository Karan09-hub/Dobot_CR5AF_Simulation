"""PyRoKi kinematics for the CR5 + Robotiq 2F-85: model loading, FK and IK on the TCP."""

from __future__ import annotations

from functools import partial

import jax
import jax.numpy as jnp
import jax_dataclasses as jdc
import jaxlie
import jaxls
import numpy as onp
import pyroki as pk
import yourdfpy

from cr5sim.paths import URDF as URDF_PATH

EE_LINK = "tcp"  # Robotiq 2F-85 fingertip pinch point
# Elbow-up "ready" pose (radians): J1..J6.
HOME_CFG = onp.deg2rad([0.0, 0.0, 90.0, 0.0, -90.0, 0.0])
# Link pairs whose capsule approximations overlap in every configuration
# (empty frame links, links sharing a joint origin, the compact wrist and gripper mount).
IGNORED_COLLISION_PAIRS = (
    ("world", "base_link"),
    ("world", "Link1"),
    ("dummy_link", "base_link"),
    ("dummy_link", "Link1"),
    ("base_link", "Link2"),
    ("Link3", "Link5"),
    ("Link4", "Link6"),
    ("Link5", "gripper"),
)

# PyRoKi ships self-collision only as a soft cost; as a hard constraint it can't be
# traded off against the pose cost, so the arm stops instead of passing through itself.
self_collision_constraint = jaxls.Cost.factory(kind="constraint_leq_zero")(
    pk._residuals.self_collision_residual
)


@jdc.jit
def _solve_ik_jax(
    robot: pk.Robot,
    robot_coll: pk.collision.RobotCollision,
    target_link_index: jax.Array,
    target_wxyz: jax.Array,
    target_position: jax.Array,
    prev_cfg: jax.Array,
    ori_weight: jax.Array,
    use_self_collision: jdc.Static[bool],
) -> jax.Array:
    joint_var = robot.joint_var_cls(0)
    costs = [
        pk.costs.pose_cost_analytic_jac(
            robot,
            joint_var,
            jaxlie.SE3.from_rotation_and_translation(
                jaxlie.SO3(target_wxyz), target_position
            ),
            target_link_index,
            pos_weight=50.0,
            ori_weight=ori_weight,
        ),
        pk.costs.limit_constraint(robot, joint_var),
        # Stay close to the previous solution so the arm moves continuously
        # instead of jumping between equivalent IK branches.
        pk.costs.rest_cost(joint_var, prev_cfg, weight=0.05),
    ]
    if use_self_collision:
        costs.append(
            self_collision_constraint(robot, robot_coll, joint_var, margin=0.01)
        )
    sol = (
        jaxls.LeastSquaresProblem(costs=costs, variables=[joint_var])
        .analyze()
        .solve(
            initial_vals=jaxls.VarValues.make([joint_var.with_value(prev_cfg)]),
            verbose=False,
            linear_solver="dense_cholesky",
            trust_region=jaxls.TrustRegionConfig(lambda_initial=1.0),
        )
    )
    return sol[joint_var]


class CR5Kinematics:
    """CR5 URDF loaded into PyRoKi, with FK/IK helpers that take and return numpy arrays."""

    def __init__(self) -> None:
        self.urdf = yourdfpy.URDF.load(
            URDF_PATH,
            load_collision_meshes=True,
            # Mesh paths in the URDF are relative to the URDF file.
            filename_handler=partial(yourdfpy.filename_handler_relative, dir=URDF_PATH.parent),
        )
        self.robot = pk.Robot.from_urdf(self.urdf, default_joint_cfg=HOME_CFG)
        self.robot_coll = pk.collision.RobotCollision.from_urdf(
            self.urdf, user_ignore_pairs=IGNORED_COLLISION_PAIRS
        )
        self.ee_index = self.robot.links.names.index(EE_LINK)
        self.joint_names = list(self.urdf.actuated_joint_names)
        self.lower = onp.asarray(self.robot.joints.lower_limits)
        self.upper = onp.asarray(self.robot.joints.upper_limits)

    def fk_ee(self, cfg: onp.ndarray) -> jaxlie.SE3:
        poses = self.robot.forward_kinematics(jnp.asarray(cfg))
        return jaxlie.SE3(onp.asarray(poses[self.ee_index]))

    def solve_ik(
        self,
        position: onp.ndarray,
        wxyz: onp.ndarray,
        prev_cfg: onp.ndarray,
        position_only: bool = False,
        self_collision: bool = True,
    ) -> onp.ndarray:
        cfg = _solve_ik_jax(
            self.robot,
            self.robot_coll,
            jnp.array(self.ee_index),
            jnp.asarray(wxyz),
            jnp.asarray(position),
            jnp.asarray(prev_cfg),
            jnp.array(0.0 if position_only else 10.0),
            self_collision,
        )
        return onp.asarray(cfg)
