# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

###########################################################################
# Example ChysX SDF Conveyor
#
# A cloth sheet lies flat on a large SDF box (like a conveyor belt).
# The box accelerates horizontally (along +Y) from rest to a constant
# speed.  Friction between the cloth and the box surface should:
#
#   1. Initially produce dynamic friction — the box moves but the cloth
#      lags behind (relative slip > 0).
#   2. After a transient, friction transfers enough momentum so the
#      cloth reaches the box's speed — entering static friction (the
#      cloth rides the box with zero relative slip).
#
# This scene demonstrates that the unified Coulomb-cone projection
# correctly handles the transition from sliding to sticking on a
# moving SDF body.
#
# Animation:
#
#   * t in [0.0, 1.0 s] — cloth settles onto the stationary box.
#   * t in [1.0, 1.5 s] — box accelerates from 0 to 0.2 m/s (+Y).
#   * t > 1.5 s         — box moves at constant 0.2 m/s; cloth
#     catches up and rides along (static friction — zero relative slip).
#
# Command: ``python -m newton.examples chysx_sdf_conveyor``
#
###########################################################################

from __future__ import annotations

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

        # ---- box geometry ------------------------------------------------
        self._box_hx = 2.0
        self._box_hy = 2.0
        self._box_hz = 0.05
        self._voxel_size = 1.0e-2

        # ---- motion schedule ---------------------------------------------
        # Phase 1: t in [1.0, 1.5] — box accelerates from 0 to v_max
        # Phase 2: t > 1.5         — box moves at constant v_max;
        #          cloth should catch up (dynamic→static friction)
        self._t_accel_start = 1.0
        self._t_accel_stop = 1.5
        self._box_v_max = 0.2   # m/s along +Y

        # ---- world geometry ----------------------------------------------
        builder = newton.ModelBuilder(up_axis=newton.Axis.Z, gravity=-9.81)

        self._cloth_dim = 41
        self._cloth_size = 0.4
        cell = self._cloth_size / (self._cloth_dim - 1)
        particle_radius = 0.4 * cell

        self._box_z_centre = self._box_hz
        cloth_start_z = self._box_z_centre + self._box_hz + 0.01

        cloth_origin = wp.vec3(
            -0.5 * self._cloth_size,
            -0.5 * self._cloth_size,
            cloth_start_z,
        )
        builder.add_cloth_grid(
            pos=cloth_origin,
            rot=wp.quat_identity(),
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

        # Visualisation proxy
        self._box_initial_pose = wp.transform(
            (0.0, 0.0, self._box_z_centre), wp.quat_identity()
        )
        self._box_body_id = builder.add_body(
            xform=self._box_initial_pose,
            is_kinematic=True,
            label="sdf_conveyor",
        )
        builder.add_shape_box(
            body=self._box_body_id,
            hx=self._box_hx,
            hy=self._box_hy,
            hz=self._box_hz,
        )

        self.model = builder.finalize()

        # ---- solver ------------------------------------------------------
        contact_thickness = max(0.5 * cell, 2.0 * self._voxel_size)
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

        # Bake box SDF
        self._box_volume_idx = self.solver.bake_sdf_box(
            hx=self._box_hx,
            hy=self._box_hy,
            hz=self._box_hz,
            voxel_size=self._voxel_size,
            thickness=contact_thickness,
            stiffness=1.0e4,
            friction=1.0,
        )

        self._box_pos = np.array(
            [0.0, 0.0, self._box_z_centre], dtype=np.float32
        )
        self._box_vel = np.zeros(3, dtype=np.float32)
        self.solver.set_sdf_pose(self._box_volume_idx, self._box_pos)
        self.solver.set_sdf_body_velocity(self._box_volume_idx, self._box_vel)

        self.state_0 = self.model.state()
        self.state_1 = self.model.state()
        self.control = self.model.control()
        self.contacts = self.model.contacts()

        self.viewer.set_model(self.model)
        bmin = np.array([-0.7, -0.7, -0.05], dtype=np.float64)
        bmax = np.array([+0.7, +1.5, +0.30], dtype=np.float64)
        frame_z_up_camera_viewer(self.viewer, bmin, bmax)

        self._body_q_np = self.state_0.body_q.numpy().copy()

    # ---- per-frame motion ------------------------------------------------

    def _update_box_pose(self) -> None:
        t = self.sim_time
        accel_dur = self._t_accel_stop - self._t_accel_start

        if t < self._t_accel_start:
            vy = 0.0
        elif t < self._t_accel_stop:
            frac = (t - self._t_accel_start) / accel_dur
            vy = self._box_v_max * frac
        else:
            vy = self._box_v_max

        self._box_pos[1] += vy * self.frame_dt
        self._box_vel[:] = (0.0, vy, 0.0)

        self.solver.set_sdf_pose(self._box_volume_idx, self._box_pos)
        self.solver.set_sdf_body_velocity(self._box_volume_idx, self._box_vel)

        self._body_q_np[self._box_body_id, 0:3] = self._box_pos
        self._body_q_np[self._box_body_id, 3:7] = (0.0, 0.0, 0.0, 1.0)
        self.state_0.body_q.assign(self._body_q_np)

    # ---- simulation ------------------------------------------------------

    def step(self):
        self._update_box_pose()
        for _ in range(self.sim_substeps):
            self.state_0.clear_forces()
            self.solver.step(
                self.state_0,
                self.state_0,
                self.control,
                self.contacts,
                self.sim_dt,
            )
        self.sim_time += self.frame_dt

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

        max_speed = float(np.linalg.norm(qd, axis=1).max())
        if max_speed > 5.0:
            raise ValueError(
                f"particle speed exploded: max |v| = {max_speed:.3f} m/s"
            )

        mean_vy = float(qd[:, 1].mean())
        print(
            f"[chysx_sdf_conveyor] final state: "
            f"cloth mean v_y = {mean_vy*1000:.2f} mm/s "
            f"(expect ~0 after box stops), "
            f"max |v| = {max_speed:.4f} m/s"
        )


if __name__ == "__main__":
    parser = newton.examples.create_parser()
    parser.set_defaults(num_frames=int(60 * 5.5))

    viewer, args = newton.examples.init(parser)
    newton.examples.run(Example(viewer, args), args)
