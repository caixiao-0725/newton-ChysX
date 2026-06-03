# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

###########################################################################
# Example ChysX Softbody Hanging — VBD solver
#
# Same scenario as example_chysx_softbody_hanging but using ChysX's VBD
# (Vertex Block Descent) solver instead of PCG.  Four tetrahedral grids
# with different damping values hanging from fixed left-face particles.
#
# The VBD solver uses graph coloring and per-vertex Gauss-Seidel
# iterations, ported 1:1 from Newton's SolverVBD.
#
# Command: uv run -m newton.examples chysx.softbody.example_chysx_softbody_hanging_vbd
#
###########################################################################

import numpy as np
import warp as wp

import newton
import newton.examples


class Example:
    def __init__(self, viewer, args):
        self.viewer = viewer
        self.sim_time = 0.0
        self.fps = 60
        self.frame_dt = 1.0 / self.fps
        self.sim_substeps = 10
        self.sim_dt = self.frame_dt / self.sim_substeps

        builder = newton.ModelBuilder()
        builder.add_ground_plane()

        dim_x = 12
        dim_y = 4
        dim_z = 4
        cell_size = 0.1

        damping_values = [1e-1, 1e-2, 1e-3, 1e-4]
        spacing = 0.6

        for i, k_damp in enumerate(damping_values):
            y_offset = i * spacing
            builder.add_soft_grid(
                pos=wp.vec3(0.0, 1.0 + y_offset, 1.0),
                rot=wp.quat_identity(),
                vel=wp.vec3(0.0, 0.0, 0.0),
                dim_x=dim_x,
                dim_y=dim_y,
                dim_z=dim_z,
                cell_x=cell_size,
                cell_y=cell_size,
                cell_z=cell_size,
                density=1.0e3,
                k_mu=1.0e5,
                k_lambda=1.0e5,
                k_damp=k_damp,
                fix_left=True,
            )

        self.model = builder.finalize()
        self.model.soft_contact_ke = 1.0e2
        self.model.soft_contact_kd = 0
        self.model.soft_contact_mu = 1.0

        inv_mass_np = self.model.particle_inv_mass.numpy()
        pin_mask = inv_mass_np == 0.0
        pin_indices = np.where(pin_mask)[0].tolist()

        self.solver = newton.solvers.SolverChysX(
            model=self.model,
            tet_fem_enabled=True,
            solver_type="vbd",
            vbd_iterations=5,
            pin_indices=pin_indices,
            pin_stiffness=1.0e12,
            static_contact_enabled=True,
            static_contact_thickness=5.0e-3,
            static_contact_stiffness=1.0e3,
        )

        self.state_0 = self.model.state()
        self.state_1 = self.model.state()
        self.control = self.model.control()
        self.contacts = self.model.contacts()

        self.viewer.set_model(self.model)

    def step(self):
        for _ in range(self.sim_substeps):
            self.state_0.clear_forces()
            self.solver.step(
                self.state_0, self.state_0, self.control, self.contacts, self.sim_dt
            )
        self.sim_time += self.frame_dt

    def render(self):
        self.viewer.begin_frame(self.sim_time)
        self.viewer.log_state(self.state_0)
        self.viewer.end_frame()

    def test_final(self):
        q = self.state_0.particle_q.numpy().reshape(-1, 3)
        qd = self.state_0.particle_qd.numpy().reshape(-1, 3)

        if not (np.isfinite(q).all() and np.isfinite(qd).all()):
            raise ValueError("non-finite values in particle state")

        p_lower = np.array([-1.0, -0.5, 0.0])
        p_upper = np.array([3.0, 4.0, 3.0])
        if (q < p_lower).any() or (q > p_upper).any():
            raise ValueError(
                f"particles escaped the bounding box; "
                f"min={q.min(axis=0)}, max={q.max(axis=0)}"
            )


if __name__ == "__main__":
    parser = newton.examples.create_parser()
    viewer, args = newton.examples.init(parser)
    newton.examples.run(Example(viewer, args), args)
