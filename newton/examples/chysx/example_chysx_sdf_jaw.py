# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

###########################################################################
# Example ChysX SDF Jaw (Rounded-Box)
#
# Two rounded-box SDF bodies simulate a parallel-jaw gripper clamping a
# vertical cloth sheet.  Unlike the sphere variant, the jaws here are
# flat pads with smooth edges — closer to a real finger pad geometry.
# The rounded-box SDF provides continuous normals around edges/corners,
# preventing the force-direction jumps that a sharp box causes on
# adjacent cloth vertices.
#
# Animation schedule (same as sdf_gripper):
#
#   * t in [0.0, 0.5 s]   — jaws open, cloth floats free (no gravity).
#   * t in [0.5, 2.0 s]   — jaws close horizontally onto the cloth.
#   * t in [2.0, 2.5 s]   — hold closed (let transients decay).
#   * t in [2.5, 5.0 s]   — jaws translate upward + forward (y).
#
# Command: ``python -m newton.examples chysx_sdf_jaw``
#
###########################################################################

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import warp as wp

import newton
import newton.examples
from newton.examples.chysx._camera import frame_z_up_camera_viewer


class Example:
    def __init__(self, viewer, args):
        self.fps = 100
        self.frame_dt = 1.0 / self.fps
        self.sim_substeps = 1
        self.sim_dt = self.frame_dt / self.sim_substeps
        self.sim_time = 0.0

        self.viewer = viewer
        self.args = args

        # ---- jaw geometry (rounded box) ----------------------------------
        self._jaw_hx = 0.012   # 24 mm thick (closing axis)
        self._jaw_hy = 0.1    # 50 cm wide — covers cloth y-extent + lift travel
        self._jaw_hz = 0.1    # 40 cm tall — fully covers ±150 mm cloth z-extent

        self._voxel_size = 2.0e-3

        # ---- motion schedule ---------------------------------------------
        self._t_close_start = 0.5
        self._t_close_stop = 2.0
        self._t_hold_stop = 2.5
        self._t_lift_stop = 20.0

        self._half_gap_open = 0.150
        self._lift_v = 0.00
        self._lift_y_travel = 0.20

        # ---- world geometry ----------------------------------------------
        builder = newton.ModelBuilder(up_axis=newton.Axis.Z, gravity=0.0)

        self._scene_z_offset = 0.25

        self._cloth_dim = 101
        self._cloth_size = 0.3
        cell = self._cloth_size / (self._cloth_dim - 1)
        particle_radius = 0.4 * cell

        contact_thickness = max(0.5 * cell, 2.0 * self._voxel_size)
        self._half_gap_closed = self._jaw_hx + 0.5 * contact_thickness

        self._jaw_z0 = self._scene_z_offset + 0.5 * self._cloth_size

        cloth_rot = wp.quat_from_axis_angle(wp.vec3(0.0, 1.0, 0.0), math.pi * 0.5)
        cloth_origin = wp.vec3(0.0, -0.5 * self._cloth_size, self._scene_z_offset + self._cloth_size)
        builder.add_cloth_grid(
            pos=cloth_origin,
            rot=cloth_rot,
            vel=wp.vec3(0.0, 0.0, 0.0),
            dim_x=self._cloth_dim - 1,
            dim_y=self._cloth_dim - 1,
            cell_x=cell,
            cell_y=cell,
            mass=0.0,
            tri_ke=0.0, tri_ka=0.0, tri_kd=0.0,
            edge_ke=0.0, edge_kd=0.0,
            particle_radius=particle_radius,
        )

        builder.add_ground_plane()

        # ---- visualisation proxies (rounded box shown as box) ------------
        self._jaw_initial_xforms = [
            wp.transform((-self._half_gap_open, 0.0, self._jaw_z0), wp.quat_identity()),
            wp.transform((+self._half_gap_open, 0.0, self._jaw_z0), wp.quat_identity()),
        ]
        self._jaw_body_ids = []
        for xform in self._jaw_initial_xforms:
            body_id = builder.add_body(
                xform=xform,
                is_kinematic=True,
                label=f"sdf_jaw_{len(self._jaw_body_ids)}",
            )
            builder.add_shape_box(
                body=body_id,
                hx=self._jaw_hx,
                hy=self._jaw_hy,
                hz=self._jaw_hz,
            )
            self._jaw_body_ids.append(body_id)

        self.model = builder.finalize()

        # ---- solver ------------------------------------------------------
        self.solver = newton.solvers.SolverChysX(
            self.model,
            damping=0.05,
            fem_stretch_stiffness=5.0e2,
            fem_shear_stiffness=5.0e2,
            bending_stiffness=1.0e-4,
            pcg_iterations=50,
            surface_density=0.3,
            self_collision_enabled=False,
            static_contact_enabled=False,
        )

        # ---- bake rounded-box SDF per jaw --------------------------------
        # corner_radius smooths the edges; must be < min(hx, hy, hz).
        # Using jaw_hx * 0.8 = ~9.6 mm gives heavily rounded thin edges
        # while keeping the broad faces flat.
        corner_radius = 0.8 * self._jaw_hx
        self._jaw_volume_ids = []
        for _ in self._jaw_body_ids:
            vid = self.solver.bake_sdf_box(
                hx=self._jaw_hx,
                hy=self._jaw_hy,
                hz=self._jaw_hz,
                voxel_size=self._voxel_size,
                corner_radius=corner_radius,
                thickness=contact_thickness,
                stiffness=1.0e4,
                friction=1.0,
                friction_epsilon=2.0e-3,
            )
            self._jaw_volume_ids.append(vid)

        self._jaw_pos = [
            np.array([-self._half_gap_open, 0.0, self._jaw_z0], dtype=np.float32),
            np.array([+self._half_gap_open, 0.0, self._jaw_z0], dtype=np.float32),
        ]
        self._jaw_vel = [
            np.zeros(3, dtype=np.float32),
            np.zeros(3, dtype=np.float32),
        ]
        for i, vid in enumerate(self._jaw_volume_ids):
            self.solver.set_sdf_pose(vid, self._jaw_pos[i])
            self.solver.set_sdf_body_velocity(vid, self._jaw_vel[i])

        self.state_0 = self.model.state()
        self.state_1 = self.model.state()
        self.control = self.model.control()
        self.contacts = self.model.contacts()
        self._initial_q = self.state_0.particle_q.numpy().reshape(-1, 3).copy()

        self.viewer.set_model(self.model)
        bmin = np.array([-0.20, -0.15, 0.10], dtype=np.float64)
        bmax = np.array([+0.20, +0.35, 0.70], dtype=np.float64)
        frame_z_up_camera_viewer(self.viewer, bmin, bmax)

        self._obj_dir: Path | None = None
        self._frame_idx = 0

        self._cuda_graph = None
        self._capture_graph()

    # ---- per-frame motion ------------------------------------------------

    def _update_jaw_poses(self) -> None:
        t = self.sim_time

        close_duration = self._t_close_stop - self._t_close_start
        if t < self._t_close_start:
            x_l = -self._half_gap_open
            vx = 0.0
        elif t < self._t_close_stop:
            close_speed = (self._half_gap_open - self._half_gap_closed) / close_duration
            x_l = -self._half_gap_open + close_speed * (t - self._t_close_start)
            vx = +close_speed
        else:
            x_l = -self._half_gap_closed
            vx = 0.0

        lift_duration = self._t_lift_stop - self._t_hold_stop
        if t < self._t_hold_stop:
            y_jaw = 0.0
            vy = 0.0
            z_jaw = self._jaw_z0
            vz = 0.0
        elif t < self._t_lift_stop:
            tau = (t - self._t_hold_stop) / lift_duration
            y_jaw = self._lift_y_travel * tau
            vy = self._lift_y_travel / lift_duration
            z_jaw = self._jaw_z0 + self._lift_v * (t - self._t_hold_stop)
            vz = self._lift_v
        else:
            y_jaw = self._lift_y_travel
            vy = 0.0
            z_jaw = self._jaw_z0 + self._lift_v * (self._t_lift_stop - self._t_hold_stop)
            vz = 0.0

        self._jaw_pos[0][:] = (x_l, y_jaw, z_jaw)
        self._jaw_pos[1][:] = (-x_l, y_jaw, z_jaw)
        self._jaw_vel[0][:] = (+vx, vy, vz)
        self._jaw_vel[1][:] = (-vx, vy, vz)

        for i, vid in enumerate(self._jaw_volume_ids):
            self.solver.set_sdf_pose(vid, self._jaw_pos[i])
            self.solver.set_sdf_body_velocity(vid, self._jaw_vel[i])

        body_q = self.state_0.body_q.numpy()
        for i, body_id in enumerate(self._jaw_body_ids):
            body_q[body_id, :3] = self._jaw_pos[i]
            body_q[body_id, 3:] = (0.0, 0.0, 0.0, 1.0)
        self.state_0.body_q.assign(body_q)

    # ---- per-frame physics -----------------------------------------------

    def _simulate_substeps(self) -> None:
        for _ in range(self.sim_substeps):
            self.state_0.clear_forces()
            self.solver.step(
                self.state_0,
                self.state_0,
                self.control,
                self.contacts,
                self.sim_dt,
            )

    def _capture_graph(self) -> None:
        self._cuda_graph = None
        device = wp.get_device()
        if not device.is_cuda:
            return
        self._simulate_substeps()
        wp.synchronize_device(device)
        with wp.ScopedCapture() as capture:
            self._simulate_substeps()
        self._cuda_graph = capture.graph

    def step(self):
        self._update_jaw_poses()
        if self._cuda_graph is not None:
            wp.capture_launch(self._cuda_graph)
        else:
            self._simulate_substeps()
        self.sim_time += self.sim_substeps * self.sim_dt
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
        bound = 1.0
        if (np.abs(q) > bound).any():
            raise ValueError(
                f"cloth particles escaped the {bound:.1f} m bounding box; "
                f"max |q| = {float(np.abs(q).max()):.3f}"
            )

        max_speed = float(np.linalg.norm(qd, axis=1).max())
        if max_speed > 5.0:
            raise ValueError(
                f"particle speed exploded: max |v| = {max_speed:.3f} m/s"
            )

        z_mean = float(q[:, 2].mean())
        z_jaw = float(self._jaw_pos[0][2])
        print(
            f"[chysx_sdf_jaw] final state:  jaw z = {z_jaw*1000:.1f} mm, "
            f"cloth mean z = {z_mean*1000:.1f} mm (lag "
            f"{(z_jaw-z_mean)*1000:.1f} mm), max |v| = {max_speed:.3f} m/s"
        )


if __name__ == "__main__":
    parser = newton.examples.create_parser()
    parser.set_defaults(num_frames=int(100 * 5.5))

    viewer, args = newton.examples.init(parser)
    newton.examples.run(Example(viewer, args), args)
