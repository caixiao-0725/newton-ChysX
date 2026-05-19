# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

###########################################################################
# Example ChysX SDF Franka
#
# Coupled robot-cloth simulation using the ChysX CUDA solver for cloth
# and Featherstone for the Franka arm.  Robot-cloth contact is handled
# through per-link MeshContact bodies: each link's collision mesh is
# uploaded into ChysX's BVH-accelerated closest-point system at setup.
# Poses are updated every substep from FK output.
#
# The simulation runs in **metre** scale.
#
# Command: ``python -m newton.examples chysx_sdf_franka``
#
###########################################################################

from __future__ import annotations

import numpy as np
import warp as wp
from pxr import Usd

import newton
import newton.examples
import newton.usd
import newton.utils
from newton import ModelBuilder, eval_fk
from newton.examples.chysx._camera import frame_z_up_camera_viewer
from newton.solvers import SolverFeatherstone


# --------------------------------------------------------------------- #
# T-shirt loader (metres, bbox-centred) — identical to tshirt_drop
# --------------------------------------------------------------------- #

def _load_tshirt_m() -> tuple[np.ndarray, np.ndarray, float]:
    """Load t-shirt mesh, convert cm->m, no centering (matches cloth_franka)."""
    stage = Usd.Stage.Open(newton.examples.get_asset("unisex_shirt.usd"))
    prim = stage.GetPrimAtPath("/root/shirt")
    m = newton.usd.get_mesh(prim)

    v_cm = np.asarray(m.vertices, dtype=np.float32)
    idx = np.asarray(m.indices, dtype=np.int32).reshape(-1, 3)

    v_m = v_cm * 0.01

    e = np.concatenate([
        np.linalg.norm(v_m[idx[:, a]] - v_m[idx[:, b]], axis=1)
        for (a, b) in [(0, 1), (1, 2), (2, 0)]
    ])
    edge_med = float(np.median(e))
    return v_m, idx, edge_med


# --------------------------------------------------------------------- #
# Warp kernel for velocity-driven FK control
# --------------------------------------------------------------------- #

@wp.kernel
def compute_ee_delta(
    body_q: wp.array[wp.transform],
    offset: wp.transform,
    body_id: int,
    bodies_per_world: int,
    target: wp.transform,
    ee_delta: wp.array[wp.spatial_vector],
):
    world_id = wp.tid()
    tf = body_q[bodies_per_world * world_id + body_id] * offset
    pos = wp.transform_get_translation(tf)
    pos_des = wp.transform_get_translation(target)
    pos_diff = pos_des - pos
    rot = wp.transform_get_rotation(tf)
    rot_des = wp.transform_get_rotation(target)
    ang_diff = rot_des * wp.quat_inverse(rot)
    ee_delta[world_id] = wp.spatial_vector(
        pos_diff[0], pos_diff[1], pos_diff[2],
        ang_diff[0], ang_diff[1], ang_diff[2])


# --------------------------------------------------------------------- #
# Example
# --------------------------------------------------------------------- #

class Example:
    def __init__(self, viewer, args):
        # ---- timing (metres) ------------------------------------------------
        self.sim_substeps = 1
        self.fps = 100
        self.frame_dt = 1.0 / self.fps
        self.sim_dt = self.frame_dt / self.sim_substeps
        self.sim_time = 0.0

        self.viewer = viewer
        self.args = args

        # ---- world geometry (metres) ----------------------------------------
        builder = ModelBuilder(up_axis=newton.Axis.Z, gravity=-9.81)

        # Franka robot
        asset_path = newton.utils.download_asset("franka_emika_panda")
        robot_builder = ModelBuilder(gravity=-9.81)
        robot_builder.add_urdf(
            str(asset_path / "urdf" / "fr3_franka_hand.urdf"),
            xform=wp.transform((-0.50, -0.50, 0.0), wp.quat_identity()),
            floating=False,
            scale=1.0,
            enable_self_collisions=False,
            collapse_fixed_joints=True,
            force_show_colliders=False,
        )
        robot_builder.joint_q[:6] = [0.0, 0.0, 0.0, -1.59695, 0.0, 2.5307]

        # Extract collision meshes BEFORE finalizing
        self._robot_shape_meshes = []
        for geo in robot_builder.shape_source:
            if geo is not None and hasattr(geo, "vertices") and hasattr(geo, "indices"):
                verts = np.asarray(geo.vertices, dtype=np.float32).reshape(-1, 3)
                indices = np.asarray(geo.indices, dtype=np.int32).ravel()
                self._robot_shape_meshes.append((verts, indices))
            else:
                self._robot_shape_meshes.append(None)

        self._robot_shape_body = list(robot_builder.shape_body)
        self._robot_shape_transform = [
            np.array(t, dtype=np.float32) for t in robot_builder.shape_transform
        ]
        self._robot_shape_scale = [
            np.array(s, dtype=np.float32) for s in robot_builder.shape_scale
        ]

        builder.add_world(robot_builder)
        self._bodies_per_world = robot_builder.body_count
        self._dof_q_per_world = robot_builder.joint_coord_count
        self._dof_qd_per_world = robot_builder.joint_dof_count

        # Table — matches cloth_franka cm-scale (0, -50, 10) with half (40, 40, 10)
        self._table_half_ext = (0.40, 0.40, 0.10)
        self._table_top_z = 0.20  # top = centre_z + hz = 0.10 + 0.10
        table_centre = (0.0, -0.50, 0.10)
        builder.add_shape_box(
            body=-1,
            xform=wp.transform(table_centre, wp.quat_identity()),
            hx=self._table_half_ext[0],
            hy=self._table_half_ext[1],
            hz=self._table_half_ext[2],
        )
        builder.add_ground_plane()

        # T-shirt — position matches cloth_franka (cm->m conversion)
        # Original: pos=(0, 70, 30) cm, rot=180° about Z
        verts_m, tris, edge_med = _load_tshirt_m()
        self._edge_med = edge_med

        builder.add_cloth_mesh(
            pos=wp.vec3(0.0, 0.70, 0.40),
            rot=wp.quat_from_axis_angle(wp.vec3(0.0, 0.0, 1.0), np.pi),
            scale=1.0,
            vel=wp.vec3(0.0, 0.0, 0.0),
            vertices=[wp.vec3(float(v[0]), float(v[1]), float(v[2]))
                      for v in verts_m],
            indices=tris.flatten().tolist(),
            density=0.0,
            tri_ke=0.0, tri_ka=0.0, tri_kd=0.0,
            edge_ke=0.0, edge_kd=0.0,
            particle_radius=0.4 * edge_med,
        )

        self.model = builder.finalize()

        # ---- solver (same params as tshirt_drop) ----------------------------
        self_thickness = 1.2e-3
        static_thickness = 0.5 * edge_med
        sc_narrow_factor = 32
        sc_broad_factor = 128

        self.solver = newton.solvers.SolverChysX(
            self.model,
            damping=0.5,
            fem_stretch_stiffness=5.0e2,
            fem_shear_stiffness=5.0e2,
            bending_stiffness=4.0e-5,
            pcg_iterations=50,
            surface_density=0.1,
            self_collision_enabled=True,
            self_collision_thickness=self_thickness,
            self_collision_stiffness=1.0e3,
            self_collision_max_contacts_factor=sc_narrow_factor,
            self_collision_max_ef_candidates_factor=sc_broad_factor,
            static_contact_enabled=True,
            static_contact_thickness=static_thickness,
            static_contact_stiffness=1.0e4,
            static_contact_friction=0.03,
            untangle_enabled=True,
            untangle_thickness=self_thickness,
            untangle_stiffness=3.0e3,
            untangle_max_contacts_factor=sc_narrow_factor,
        )

        # ---- register robot mesh bodies in ChysX ----------------------------
        mesh_thickness = 0.004
        mesh_stiffness = 5.0e4
        mesh_friction = 1.0
        self._mesh_body_indices = []

        for shape_idx, mesh_data in enumerate(self._robot_shape_meshes):
            if mesh_data is None:
                self._mesh_body_indices.append(-1)
                continue
            verts, indices = mesh_data
            scale = self._robot_shape_scale[shape_idx]
            scaled_verts = verts * scale
            if len(indices) < 3 or len(scaled_verts) < 3:
                self._mesh_body_indices.append(-1)
                continue
            idx = self.solver.add_mesh_body(
                vertices=scaled_verts,
                indices=indices,
                thickness=mesh_thickness,
                stiffness=mesh_stiffness,
                friction=mesh_friction,
                friction_epsilon=0.01,
                contact_kd=0.01,
                ipc_friction=True,
            )
            self._mesh_body_indices.append(idx)

        # ---- states ----------------------------------------------------------
        self.state_0 = self.model.state()
        self.state_1 = self.model.state()
        self.control = self.model.control()
        self.contacts = self.model.contacts()

        self.target_joint_qd = wp.empty_like(self.state_0.joint_qd)

        # ---- robot solver ----------------------------------------------------
        self.robot_solver = SolverFeatherstone(
            self.model, update_mass_matrix_interval=self.sim_substeps)
        self._setup_robot_control(robot_builder)

        # evaluate FK for initial state
        eval_fk(self.model, self.model.joint_q, self.model.joint_qd, self.state_0)

        # ---- camera ----------------------------------------------------------
        self.viewer.set_model(self.model)
        self.viewer.set_camera(wp.vec3(-0.6, 0.6, 1.24), -42.0, -58.0)

        self._frame_idx = 0

    # ---- robot control -------------------------------------------------------

    def _setup_robot_control(self, robot_builder):
        self.endeffector_id = robot_builder.body_count - 3
        self.endeffector_offset = wp.transform((0.0, 0.0, 0.22), wp.quat_identity())

        clamp_close = 0.001
        clamp_open = 0.008

        self.robot_key_poses = np.array([
            # [duration_s, x, y, z, qw, qx, qy, qz, gripper_width]
            [4.0, 0.31, -0.60, 0.40, 0.8536, -0.3536, 0.3536, -0.1464, clamp_open],
            [2.0, 0.31, -0.60, 0.20, 0.8536, -0.3536, 0.3536, -0.1464, clamp_open],
            [2.0, 0.31, -0.60, 0.20, 0.8536, -0.3536, 0.3536, -0.1464, clamp_close],
            [2.0, 0.26, -0.60, 0.26, 0.8536, -0.3536, 0.3536, -0.1464, clamp_close],
            [2.0, 0.12, -0.60, 0.31, 0.8536, -0.3536, 0.3536, -0.1464, clamp_close],
            [3.0, -0.06, -0.60, 0.31, 0.8536, -0.3536, 0.3536, -0.1464, clamp_close],
            [1.0, -0.06, -0.60, 0.31, 0.8536, -0.3536, 0.3536, -0.1464, clamp_open],
            [2.0, 0.15, -0.33, 0.31, 0.8536, -0.3536, 0.3536, -0.1464, clamp_open],
            [3.0, 0.15, -0.33, 0.21, 0.8536, -0.3536, 0.3536, -0.1464, clamp_open],
            [3.0, 0.15, -0.33, 0.21, 0.8536, -0.3536, 0.3536, -0.1464, clamp_close],
            [2.0, 0.15, -0.33, 0.28, 0.8536, -0.3536, 0.3536, -0.1464, clamp_close],
            [3.0, -0.02, -0.33, 0.28, 0.8536, -0.3536, 0.3536, -0.1464, clamp_close],
            [1.0, -0.02, -0.33, 0.28, 0.8536, -0.3536, 0.3536, -0.1464, clamp_open],
            [2.0, -0.28, -0.60, 0.28, 0.9239, -0.3827, 0.0, 0.0, clamp_open],
            [2.0, -0.28, -0.60, 0.20, 0.9239, -0.3827, 0.0, 0.0, clamp_open],
            [2.0, -0.28, -0.60, 0.20, 0.9239, -0.3827, 0.0, 0.0, clamp_close],
        ], dtype=np.float32)
        self.targets = self.robot_key_poses[:, 1:]
        self.transition_duration = self.robot_key_poses[:, 0]
        self.target = self.targets[0]
        self.robot_key_poses_time = np.cumsum(self.robot_key_poses[:, 0])

        in_dim = self.model.joint_dof_count
        out_dim = 6

        def onehot(i, od):
            return wp.array([1.0 if j == i else 0.0 for j in range(od)], dtype=float)

        self.Jacobian_one_hots = [onehot(i, out_dim) for i in range(out_dim)]

        @wp.kernel
        def compute_body_out(
            body_q: wp.array[wp.transform],
            body_qd: wp.array[wp.spatial_vector],
            body_com: wp.array[wp.vec3],
            body_out: wp.array[float],
        ):
            ee_id = wp.static(self.endeffector_id)
            ee_offset = wp.static(wp.vec3(0.0, 0.0, 0.22))
            X_wb = body_q[ee_id]
            r_world = wp.transform_vector(X_wb, ee_offset - body_com[ee_id])
            qd = body_qd[ee_id]
            omega = wp.spatial_bottom(qd)
            v_com = wp.spatial_top(qd)
            v_tip = v_com + wp.cross(omega, r_world)
            body_out[0] = v_tip[0]
            body_out[1] = v_tip[1]
            body_out[2] = v_tip[2]
            body_out[3] = omega[0]
            body_out[4] = omega[1]
            body_out[5] = omega[2]

        self.compute_body_out_kernel = compute_body_out
        self.temp_state_for_jacobian = self.model.state(requires_grad=True)
        self.body_out = wp.empty(out_dim, dtype=float, requires_grad=True)
        self.J_flat = wp.empty(out_dim * in_dim, dtype=float)
        self.ee_delta = wp.empty(1, dtype=wp.spatial_vector)
        self.initial_pose = self.model.joint_q.numpy()

    def _compute_body_jacobian(self, joint_q, joint_qd):
        joint_q.requires_grad = True
        joint_qd.requires_grad = True
        in_dim = self.model.joint_dof_count
        out_dim = 6

        tape = wp.Tape()
        with tape:
            eval_fk(self.model, joint_q, joint_qd, self.temp_state_for_jacobian)
            wp.launch(
                self.compute_body_out_kernel, 1,
                inputs=[
                    self.temp_state_for_jacobian.body_q,
                    self.temp_state_for_jacobian.body_qd,
                    self.model.body_com,
                ],
                outputs=[self.body_out],
            )
        for i in range(out_dim):
            tape.backward(grads={self.body_out: self.Jacobian_one_hots[i]})
            wp.copy(self.J_flat[i * in_dim: (i + 1) * in_dim], joint_qd.grad)
            tape.zero()

    def _generate_control(self, state_in):
        if self.sim_time >= self.robot_key_poses_time[-1]:
            self.target_joint_qd.zero_()
            return

        current_interval = int(np.searchsorted(self.robot_key_poses_time, self.sim_time))
        self.target = self.targets[min(current_interval, len(self.targets) - 1)]

        wp.launch(
            compute_ee_delta, dim=1,
            inputs=[
                state_in.body_q,
                self.endeffector_offset,
                self.endeffector_id,
                self._bodies_per_world,
                wp.transform(*self.target[:7]),
            ],
            outputs=[self.ee_delta],
        )
        self._compute_body_jacobian(state_in.joint_q, state_in.joint_qd)
        J = self.J_flat.numpy().reshape(-1, self.model.joint_dof_count)
        delta_target = self.ee_delta.numpy()[0]

        if not np.isfinite(J).all():
            self.target_joint_qd.zero_()
            return

        try:
            J_inv = np.linalg.pinv(J)
        except np.linalg.LinAlgError:
            self.target_joint_qd.zero_()
            return

        I = np.eye(J.shape[1], dtype=np.float32)
        N = I - J_inv @ J

        q = state_in.joint_q.numpy()
        q_des = q.copy()
        q_des[1:] = self.initial_pose[1:]
        K_null = 1.0
        delta_q_null = K_null * (q_des - q)
        delta_q = J_inv @ delta_target + N @ delta_q_null

        # Gripper finger control (metre scale)
        delta_q[-2] = self.target[-1] - q[-2]
        delta_q[-1] = self.target[-1] - q[-1]

        self.target_joint_qd.assign(delta_q)

    # ---- mesh body pose update ------------------------------------------------

    def _update_mesh_body_poses(self):
        """Read body_q from state, update each mesh body's pose + velocity."""
        body_q_np = self.state_0.body_q.numpy()
        body_qd_np = self.state_0.body_qd.numpy()

        for shape_idx, mc_idx in enumerate(self._mesh_body_indices):
            if mc_idx < 0:
                continue
            body_idx = self._robot_shape_body[shape_idx]
            if body_idx < 0:
                continue

            # body world transform
            tf = body_q_np[body_idx]
            body_pos = tf[:3]
            body_quat = tf[3:]  # (x, y, z, w)
            # Newton stores quaternion as (px, py, pz, qx, qy, qz, qw) in transform
            # body_q is stored as wp.transform = (p, q) where p is 3 floats and q is 4 floats (x,y,z,w)
            qw, qx, qy, qz = body_quat[3], body_quat[0], body_quat[1], body_quat[2]

            # Shape local transform
            shape_tf = self._robot_shape_transform[shape_idx]
            sp = shape_tf[:3]
            sq = shape_tf[3:]  # (x, y, z, w)
            sqw, sqx, sqy, sqz = sq[3], sq[0], sq[1], sq[2]

            # Compose: world = body_tf * shape_tf
            # Quaternion multiply (Hamilton convention)
            rw = qw * sqw - qx * sqx - qy * sqy - qz * sqz
            rx = qw * sqx + qx * sqw + qy * sqz - qz * sqy
            ry = qw * sqy - qx * sqz + qy * sqw + qz * sqx
            rz = qw * sqz + qx * sqy - qy * sqx + qz * sqw

            # Rotate shape local offset by body rotation
            # v' = q * v * q^-1 (using expanded formula)
            t = 2.0 * np.array([
                qy * sp[2] - qz * sp[1],
                qz * sp[0] - qx * sp[2],
                qx * sp[1] - qy * sp[0],
            ], dtype=np.float32)
            rotated_sp = sp + qw * t + np.array([
                qy * t[2] - qz * t[1],
                qz * t[0] - qx * t[2],
                qx * t[1] - qy * t[0],
            ], dtype=np.float32)

            world_pos = body_pos + rotated_sp

            # Rotation matrix from quaternion (rw, rx, ry, rz)
            rot = np.array([
                [1 - 2*(ry*ry + rz*rz), 2*(rx*ry - rz*rw), 2*(rx*rz + ry*rw)],
                [2*(rx*ry + rz*rw), 1 - 2*(rx*rx + rz*rz), 2*(ry*rz - rx*rw)],
                [2*(rx*rz - ry*rw), 2*(ry*rz + rx*rw), 1 - 2*(rx*rx + ry*ry)],
            ], dtype=np.float32)

            self.solver.set_mesh_body_pose(mc_idx, world_pos, rot)

            # Linear velocity (from spatial velocity of body)
            sv = body_qd_np[body_idx]
            v_lin = sv[:3]  # spatial_top = linear velocity at COM
            self.solver.set_mesh_body_velocity(mc_idx, v_lin.astype(np.float32))

    # ---- physics ---------------------------------------------------------

    def step(self):
        self._generate_control(self.state_0)

        for _ in range(self.sim_substeps):
            self.state_0.clear_forces()
            self.state_1.clear_forces()

            # Robot FK update
            particle_count = self.model.particle_count
            self.model.particle_count = 0
            self.state_0.joint_qd.assign(self.target_joint_qd)
            self.robot_solver.step(
                self.state_0, self.state_1, self.control, None, self.sim_dt)
            self.model.particle_count = particle_count

            # Copy robot state back to state_0 for mesh body update
            wp.copy(self.state_0.body_q, self.state_1.body_q)
            wp.copy(self.state_0.body_qd, self.state_1.body_qd)
            wp.copy(self.state_0.joint_q, self.state_1.joint_q)
            wp.copy(self.state_0.joint_qd, self.state_1.joint_qd)

            wp.synchronize()

            # Update mesh body poses from new FK
            self._update_mesh_body_poses()

            # Cloth sim
            self.solver.step(
                self.state_0, self.state_0,
                self.control, self.contacts, self.sim_dt)

        self.sim_time += self.frame_dt
        self._frame_idx += 1

    def render(self):
        self.viewer.begin_frame(self.sim_time)
        self.viewer.log_state(self.state_0)
        self.viewer.end_frame()

    # ---- regression check ------------------------------------------------

    def test_final(self):
        q = self.state_0.particle_q.numpy().reshape(-1, 3)
        qd = self.state_0.particle_qd.numpy().reshape(-1, 3)
        if not (np.isfinite(q).all() and np.isfinite(qd).all()):
            raise ValueError("non-finite values in particle state")
        bound = 5.0
        if (np.abs(q) > bound).any():
            raise ValueError(
                f"cloth escaped {bound:.1f} m bbox; "
                f"max |q| = {float(np.abs(q).max()):.3f}")
        z_min = float(q[:, 2].min())
        slack = -2.0 * float(self.solver._sim.static_contact_thickness())
        if z_min < slack:
            raise ValueError(
                f"cloth fell through ground: min z = {z_min:.4f} m "
                f"(allowed slack {slack:.4f} m)")
        max_speed = float(np.linalg.norm(qd, axis=1).max())
        if max_speed > 50.0:
            raise ValueError(
                f"particle speed exploded: max |v| = {max_speed:.3f} m/s")
        print(
            f"[chysx_sdf_franka] final state: max |v| = {max_speed:.3f} m/s, "
            f"z range = [{float(q[:, 2].min()):.3f}, {float(q[:, 2].max()):.3f}] m")


if __name__ == "__main__":
    parser = newton.examples.create_parser()
    parser.set_defaults(num_frames=600)

    viewer, args = newton.examples.init(parser)
    newton.examples.run(Example(viewer, args), args)
