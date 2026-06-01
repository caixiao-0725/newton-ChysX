# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
#
# Minimal test: cloth drops onto a STATIC Franka arm (no robot motion).
# Verifies MeshContact stability with the BVH point-query pipeline.

from __future__ import annotations

from collections import defaultdict

import numpy as np
import warp as wp

import newton
import newton.examples
import newton.utils
from newton import ModelBuilder, eval_fk


class Example:
    def __init__(self, viewer, args):
        self.sim_substeps = 1
        self.fps = 100
        self.frame_dt = 1.0 / self.fps
        self.sim_dt = self.frame_dt / self.sim_substeps
        self.sim_time = 0.0
        self.viewer = viewer

        builder = ModelBuilder(up_axis=newton.Axis.Z, gravity=-9.81)

        # ---- Franka (static, just for collision geometry) --------------------
        asset_path = newton.utils.download_asset("franka_emika_panda")
        robot_builder = ModelBuilder(gravity=-9.81)
        robot_builder.add_urdf(
            str(asset_path / "urdf" / "fr3_franka_hand.urdf"),
            xform=wp.transform((0.0, 0.0, 0.0), wp.quat_identity()),
            floating=False,
            scale=1.0,
            enable_self_collisions=False,
            collapse_fixed_joints=True,
            force_show_colliders=False,
        )
        # Default home-like pose (same as sdf_franka example)
        robot_builder.joint_q[:6] = [0.0, 0.0, 0.0, -1.59695, 0.0, 2.5307]

        # Extract collision meshes
        robot_shape_meshes = []
        for geo in robot_builder.shape_source:
            if geo is not None and hasattr(geo, "vertices") and hasattr(geo, "indices"):
                verts = np.asarray(geo.vertices, dtype=np.float32).reshape(-1, 3)
                indices = np.asarray(geo.indices, dtype=np.int32).ravel()
                robot_shape_meshes.append((verts, indices))
            else:
                robot_shape_meshes.append(None)

        robot_shape_body = list(robot_builder.shape_body)
        robot_shape_transform = [
            np.array(t, dtype=np.float32) for t in robot_builder.shape_transform
        ]
        robot_shape_scale = [
            np.array(s, dtype=np.float32) for s in robot_builder.shape_scale
        ]

        builder.add_world(robot_builder)

        # ---- ground plane ----------------------------------------------------
        builder.add_ground_plane()

        # ---- cloth patch (0.40 x 0.30 m grid) above the forearm ----------------
        # Forearm centre at x≈0.4, y≈0, z≈0.72. Drop from z=0.95
        res = 30
        cloth_w = 0.40
        cloth_h = 0.30
        cell_x = cloth_w / res
        cell_y = cloth_h / res
        cloth_z = 0.95  # above the forearm top at z≈0.80
        cx, cy = 0.35, 0.0  # over the forearm
        verts = []
        for j in range(res + 1):
            for i in range(res + 1):
                verts.append(wp.vec3(
                    cx - cloth_w / 2 + i * cell_x,
                    cy - cloth_h / 2 + j * cell_y,
                    cloth_z,
                ))
        indices = []
        for j in range(res):
            for i in range(res):
                v0 = j * (res + 1) + i
                v1 = v0 + 1
                v2 = v0 + (res + 1)
                v3 = v2 + 1
                indices.extend([v0, v1, v2])
                indices.extend([v1, v3, v2])

        edge_med = max(cell_x, cell_y)

        builder.add_cloth_mesh(
            pos=wp.vec3(0.0, 0.0, 0.0),
            rot=wp.quat_identity(),
            scale=1.0,
            vel=wp.vec3(0.0, 0.0, 0.0),
            vertices=verts,
            indices=indices,
            density=0.0,
            tri_ke=0.0, tri_ka=0.0, tri_kd=0.0,
            edge_ke=0.0, edge_kd=0.0,
            particle_radius=0.4 * edge_med,
        )

        self.model = builder.finalize()

        # ---- ChysX solver ----------------------------------------------------
        self_thickness = 1.0e-3
        static_thickness = 0.4 * edge_med

        self.solver = newton.solvers.SolverChysX(
            self.model,
            damping=0.9,
            fem_stretch_stiffness=5.0e2,
            fem_shear_stiffness=5.0e2,
            bending_stiffness=4.0e-5,
            pcg_iterations=50,
            surface_density=0.1,
            self_collision_enabled=True,
            self_collision_thickness=self_thickness,
            self_collision_stiffness=1.0e2,
            self_collision_max_contacts_factor=32,
            self_collision_max_ef_candidates_factor=256,
            static_contact_enabled=True,
            static_contact_thickness=static_thickness,
            static_contact_stiffness=1.0e4,
            static_contact_friction=0.3,
            untangle_enabled=True,
            untangle_thickness=self_thickness,
            untangle_stiffness=3.0e2,
            untangle_max_contacts_factor=32,
        )

        # ---- register mesh bodies (static — pose set once) -------------------
        mesh_thickness = 0.002
        mesh_stiffness = 5.0e4

        # Evaluate FK to get body transforms
        self.state_0 = self.model.state()
        self.state_1 = self.model.state()
        self.control = self.model.control()
        self.contacts = self.model.contacts()

        eval_fk(self.model, self.model.joint_q, self.model.joint_qd, self.state_0)
        wp.synchronize()

        body_q_np = self.state_0.body_q.numpy()

        body_shapes = defaultdict(list)
        for si, mesh_data in enumerate(robot_shape_meshes):
            if mesh_data is None:
                continue
            body_idx = robot_shape_body[si]
            if body_idx < 0:
                continue
            verts_arr, idx_arr = mesh_data
            if len(idx_arr) < 3 or len(verts_arr) < 3:
                continue
            body_shapes[body_idx].append((si, len(idx_arr) // 3))

        for body_idx, candidates in body_shapes.items():
            # Pick the shape with the most triangles
            si, _ = max(candidates, key=lambda x: x[1])
            verts_arr, idx_arr = robot_shape_meshes[si]
            scale = robot_shape_scale[si]
            scaled_verts = verts_arr * scale

            mc_idx = self.solver.add_mesh_body(
                vertices=scaled_verts,
                indices=idx_arr,
                thickness=mesh_thickness,
                stiffness=mesh_stiffness,
                friction=0.0,
                friction_epsilon=0.01,
                contact_kd=0.02,
                ipc_friction=True,
            )

            tf = body_q_np[body_idx]
            body_pos = tf[:3]
            body_quat = tf[3:]  # (x, y, z, w)
            qw, qx, qy, qz = body_quat[3], body_quat[0], body_quat[1], body_quat[2]

            shape_tf = robot_shape_transform[si]
            sp = shape_tf[:3]
            sq = shape_tf[3:]
            sqw, sqx, sqy, sqz = sq[3], sq[0], sq[1], sq[2]

            rw = qw*sqw - qx*sqx - qy*sqy - qz*sqz
            rx = qw*sqx + qx*sqw + qy*sqz - qz*sqy
            ry = qw*sqy - qx*sqz + qy*sqw + qz*sqx
            rz = qw*sqz + qx*sqy - qy*sqx + qz*sqw

            t = 2.0 * np.array([qy*sp[2]-qz*sp[1], qz*sp[0]-qx*sp[2], qx*sp[1]-qy*sp[0]], dtype=np.float32)
            rotated_sp = sp + qw*t + np.array([qy*t[2]-qz*t[1], qz*t[0]-qx*t[2], qx*t[1]-qy*t[0]], dtype=np.float32)
            world_pos = body_pos + rotated_sp

            rot = np.array([
                [1-2*(ry*ry+rz*rz), 2*(rx*ry-rz*rw), 2*(rx*rz+ry*rw)],
                [2*(rx*ry+rz*rw), 1-2*(rx*rx+rz*rz), 2*(ry*rz-rx*rw)],
                [2*(rx*rz-ry*rw), 2*(ry*rz+rx*rw), 1-2*(rx*rx+ry*ry)],
            ], dtype=np.float32)

            self.solver.set_mesh_body_pose(mc_idx, world_pos, rot)

        # ---- viewer ----------------------------------------------------------
        self.viewer.set_model(self.model)
        # Camera looking at the forearm area (x≈0.3, y≈0, z≈0.7)
        self.viewer.set_camera(wp.vec3(0.3, 1.2, 0.8), -90.0, -20.0)

    def step(self):
        for _ in range(self.sim_substeps):
            self.state_0.clear_forces()
            self.solver.step(
                self.state_0, self.state_0,
                self.control, self.contacts, self.sim_dt)
        self.sim_time += self.frame_dt

    def render(self):
        self.viewer.begin_frame(self.sim_time)
        self.viewer.log_state(self.state_0)
        self.viewer.end_frame()

    def test_final(self):
        q = self.state_0.particle_q.numpy().reshape(-1, 3)
        qd = self.state_0.particle_qd.numpy().reshape(-1, 3)
        max_speed = float(np.linalg.norm(qd, axis=1).max())
        z_min = float(q[:, 2].min())
        print(f"[mesh_drop_test] z_min={z_min:.4f} max|v|={max_speed:.3f}")
        if max_speed > 20.0:
            raise ValueError(f"speed exploded: {max_speed:.3f}")


if __name__ == "__main__":
    parser = newton.examples.create_parser()
    parser.set_defaults(num_frames=500)
    viewer, args = newton.examples.init(parser)
    newton.examples.run(Example(viewer, args), args)
