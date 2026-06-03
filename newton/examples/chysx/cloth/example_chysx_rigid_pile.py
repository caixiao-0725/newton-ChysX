# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

###########################################################################
# Example ChysX Rigid Pile
#
# Demonstrates the AVBD rigid-body solver running entirely in ChysX's
# native CUDA backend.  Creates a pile of capsule bodies connected by
# BALL joints (emulating cable segments), similar to the Newton
# cable_pile example, then drops them onto a ground plane.
#
# Command: python -m newton.examples chysx_rigid_pile
#
###########################################################################

from __future__ import annotations

import math

import numpy as np
import warp as wp

import newton
import newton.examples


def _capsule_inertia(mass: float, radius: float, half_height: float) -> np.ndarray:
    """Inertia tensor for a solid capsule (Y-axis aligned in body frame)."""
    r2 = radius * radius
    h = 2.0 * half_height
    m_cyl = mass * (h / (h + (4.0 / 3.0) * radius))
    m_cap = mass - m_cyl
    Iyy_cyl = 0.5 * m_cyl * r2
    Ixx_cyl = m_cyl * (3.0 * r2 + h * h) / 12.0
    Iyy_cap = 2.0 * m_cap * r2 * (2.0 / 5.0)
    Ixx_cap = 2.0 * m_cap * (2.0 * r2 / 5.0 + half_height * half_height)
    Ixx = Ixx_cyl + Ixx_cap
    Iyy = Iyy_cyl + Iyy_cap
    return np.diag([Ixx, Iyy, Ixx]).astype(np.float32)


class Example:
    def __init__(
        self,
        viewer,
        args=None,
        layers: int = 40,
        lanes_per_layer: int = 10,
    ):
        self.viewer = viewer
        self.args = args

        # Simulation cadence — matches cable_pile exactly
        self.fps = 60
        self.frame_dt = 1.0 / self.fps
        self.sim_time = 0.0
        self.sim_substeps = 10
        self.sim_iterations = 5
        self.sim_dt = self.frame_dt / self.sim_substeps

        # Cable pile parameters — matches cable_pile exactly
        num_elements = 40
        segment_length = 0.05
        cable_radius = 0.012
        cable_length = num_elements * segment_length

        lane_spacing = max(8.0 * cable_radius, 0.15)
        layer_gap = cable_radius * 6.0

        # Derived
        hh = segment_length * 0.5
        cable_density = 1000.0
        cable_volume = math.pi * cable_radius**2 * segment_length
        segment_mass = cable_density * cable_volume

        # Import chysx native module
        import chysx._chysx_native as _native  # type: ignore[import]

        sim = _native.RigidSimulator()
        sim.set_iterations(self.sim_iterations)
        # Z-up like cable_pile
        sim.set_gravity(np.array([0.0, 0.0, -9.81], dtype=np.float32))
        sim.set_contact_hard(True)
        sim.set_contact_history(True)
        sim.set_avbd_alpha(0.95)
        sim.set_avbd_gamma(0.999)
        sim.set_avbd_beta(0.0)

        total_bodies_est = layers * lanes_per_layer * num_elements
        sim.set_contact_buffer_size(max(16384, total_bodies_est * 8))
        sim.set_per_body_contact_capacity(128)
        sim.set_max_broadphase_pairs(max(131072, total_bodies_est * 16))

        # Ground plane — Z-up, ke/kd/mu match cable_pile's ground
        # cable_pile uses mu=1e9 for the ground; we use a large friction
        sim.add_ground_plane(ke=1e5, kd=0.0, mu=1e9)

        I_seg = _capsule_inertia(segment_mass, cable_radius, hh)
        com = np.array([0.0, 0.0, 0.0], dtype=np.float32)
        identity_q = np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32)

        # Build cables layer by layer — same geometry as cable_pile
        all_body_count = 0

        for layer in range(layers):
            orient_x = (layer % 2 == 0)
            z0 = 0.3 + layer * layer_gap

            for lane in range(lanes_per_layer):
                offset = (lane - (lanes_per_layer - 1) * 0.5) * lane_spacing
                prev_body = -1

                for seg in range(num_elements):
                    t = (seg + 0.5) / num_elements
                    along = -0.5 * cable_length + (seg + 0.5) * segment_length

                    # Sinusoidal waviness (same as cable_pile: cycles=2, scale=0.05, wav=0.5)
                    phase = 2.0 * math.pi * 2.0 * t
                    wav_offset = 0.5 * cable_length * 0.05 * math.sin(phase)

                    if orient_x:
                        x = along
                        y = offset + wav_offset
                        z = z0
                        # Capsule Y-axis along world X: rotate 90° around Z
                        q = np.array([0.0, 0.0, 0.7071068, 0.7071068], dtype=np.float32)
                    else:
                        x = offset + wav_offset
                        y = along
                        z = z0
                        # Capsule Y-axis along world Y: identity
                        q = identity_q.copy()

                    pos = np.array([x, y, z], dtype=np.float32)
                    body_id = sim.add_body(segment_mass, I_seg, com, pos, q)
                    sim.add_shape_capsule(body_id, cable_radius, hh,
                                          ke=1e5, kd=0.0, mu=1.0, gap=0.005)
                    all_body_count += 1

                    if prev_body >= 0:
                        anchor_p = np.array([0.0, hh, 0.0], dtype=np.float32)
                        anchor_c = np.array([0.0, -hh, 0.0], dtype=np.float32)
                        sim.add_joint(
                            _native.JointType.BALL,
                            prev_body, body_id,
                            anchor_p, identity_q,
                            anchor_c, identity_q,
                        )
                    prev_body = body_id

        sim.finalize()
        self.sim = sim
        self.total_bodies = all_body_count

        # --- Newton model for viewer rendering ---
        builder = newton.ModelBuilder()
        for i in range(all_body_count):
            builder.add_body(xform=wp.transform_identity())
            builder.add_shape_capsule(
                body=i,
                radius=cable_radius,
                half_height=hh,
            )
        builder.add_ground_plane()

        self.model = builder.finalize()
        self.model.ground = True
        self.state = self.model.state()

        if self.viewer is not None:
            self.viewer.set_model(self.model)
            self.viewer.set_camera(wp.vec3(0.0, -2.0, 1.5), -20.0, 90.0)

    def step(self):
        for _ in range(self.sim_substeps):
            self.sim.step(self.sim_dt)
        self.sim_time += self.frame_dt

    def render(self):
        if self.viewer is None:
            return

        pos_arr, quat_arr = self.sim.get_body_poses()

        body_q_np = self.state.body_q.numpy()
        body_q_np[:self.total_bodies, 0:3] = pos_arr
        body_q_np[:self.total_bodies, 3:7] = quat_arr
        self.state.body_q.assign(body_q_np)

        self.viewer.begin_frame(self.sim_time)
        self.viewer.log_state(self.state)
        self.viewer.end_frame()

    def capture(self):
        pass

    def test_final(self):
        """Verify bodies have settled — mirrors cable_pile test_final logic."""
        cable_radius = 0.012
        cable_diameter = 2.0 * cable_radius
        tolerance = 0.5

        pos_arr, _ = self.sim.get_body_poses()

        assert np.isfinite(pos_arr).all(), "Non-finite positions"

        z_positions = pos_arr[:, 2]
        min_z = float(np.min(z_positions))
        max_z = float(np.max(z_positions))

        assert min_z > -tolerance, (
            f"Cables penetrated ground: min_z={min_z:.3f} < {-tolerance:.3f}"
        )

        max_z_settled = 40 * cable_diameter + tolerance
        assert max_z < max_z_settled, (
            f"Pile too high: max_z={max_z:.3f} > expected {max_z_settled:.3f}"
        )


if __name__ == "__main__":
    viewer, args = newton.examples.init()
    newton.examples.run(Example(viewer, args), args)
