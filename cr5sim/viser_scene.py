"""Mirror a MuJoCo model into a Viser scene: meshes are sent once, body poses every frame."""

from __future__ import annotations

import mujoco
import numpy as onp
import trimesh
import viser

# Geom groups to draw: 0-2 are visual by convention; 3 holds collision hulls, 5 debug sites.
VISIBLE_GROUPS = (0, 1, 2)


def _geom_mesh(model: mujoco.MjModel, g: int) -> trimesh.Trimesh | None:
    gtype, size = model.geom_type[g], model.geom_size[g]
    if gtype == mujoco.mjtGeom.mjGEOM_MESH:
        mesh_id = model.geom_dataid[g]
        v0, nv = model.mesh_vertadr[mesh_id], model.mesh_vertnum[mesh_id]
        f0, nf = model.mesh_faceadr[mesh_id], model.mesh_facenum[mesh_id]
        return trimesh.Trimesh(model.mesh_vert[v0 : v0 + nv], model.mesh_face[f0 : f0 + nf], process=False)
    if gtype == mujoco.mjtGeom.mjGEOM_BOX:
        return trimesh.creation.box(extents=2 * size)
    if gtype == mujoco.mjtGeom.mjGEOM_SPHERE:
        return trimesh.creation.icosphere(subdivisions=2, radius=size[0])
    if gtype == mujoco.mjtGeom.mjGEOM_CYLINDER:
        return trimesh.creation.cylinder(radius=size[0], height=2 * size[1])
    if gtype == mujoco.mjtGeom.mjGEOM_CAPSULE:
        return trimesh.creation.capsule(radius=size[0], height=2 * size[1])
    return None  # planes are drawn as a grid


def _rgba(model: mujoco.MjModel, g: int) -> onp.ndarray:
    mat = model.geom_matid[g]
    return model.mat_rgba[mat] if mat >= 0 else model.geom_rgba[g]


class MujocoViserScene:
    def __init__(self, server: viser.ViserServer, model: mujoco.MjModel, data: mujoco.MjData,
                 root: str = "/sim") -> None:
        self.model, self.data = model, data
        server.scene.add_grid(f"{root}/floor", width=2.0, height=2.0, cell_size=0.1)
        self.frames: dict[int, viser.FrameHandle] = {}
        for g in range(model.ngeom):
            if model.geom_group[g] not in VISIBLE_GROUPS:
                continue
            mesh = _geom_mesh(model, g)
            if mesh is None:
                continue
            body = model.geom_bodyid[g]
            if body == 0:  # static: place once in world coordinates
                parent, pos = root, data.geom_xpos[g]
                wxyz = onp.zeros(4)
                mujoco.mju_mat2Quat(wxyz, data.geom_xmat[g])
            else:
                if body not in self.frames:
                    self.frames[body] = server.scene.add_frame(f"{root}/body{body}", show_axes=False)
                parent, pos, wxyz = f"{root}/body{body}", model.geom_pos[g], model.geom_quat[g]
            rgba = _rgba(model, g)
            server.scene.add_mesh_simple(
                f"{parent}/geom{g}",
                vertices=mesh.vertices.astype(onp.float32),
                faces=mesh.faces.astype(onp.uint32),
                color=tuple(int(c * 255) for c in rgba[:3]),
                opacity=float(rgba[3]) if rgba[3] < 1.0 else None,
                position=pos,
                wxyz=wxyz,
            )
        # Box sites (e.g. the place target marker).
        for s in range(model.nsite):
            if model.site_type[s] == mujoco.mjtGeom.mjGEOM_BOX and model.site_group[s] in VISIBLE_GROUPS:
                wxyz = onp.zeros(4)
                mujoco.mju_mat2Quat(wxyz, data.site_xmat[s])
                rgba = model.site_rgba[s]
                server.scene.add_box(
                    f"{root}/site{s}", dimensions=2 * model.site_size[s],
                    color=tuple(int(c * 255) for c in rgba[:3]),
                    opacity=float(rgba[3]), position=data.site_xpos[s], wxyz=wxyz,
                )
        self.update()

    def update(self) -> None:
        for body, frame in self.frames.items():
            frame.position = self.data.xpos[body]
            frame.wxyz = self.data.xquat[body]
