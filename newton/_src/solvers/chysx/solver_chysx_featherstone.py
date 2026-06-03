# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""ChysX Featherstone rigid-body solver.

Drop-in replacement for ``SolverFeatherstone`` that routes all rigid-body
dynamics (FK, RNEA, CRBA, Cholesky solve, integration) through ChysX's
native C++/CUDA backend.

The solver accepts the same Newton ``Model`` / ``State`` / ``Control``
objects and writes results back to Warp arrays so the rest of the Newton
pipeline (rendering, IK, etc.) continues to work.
"""

from __future__ import annotations

import numpy as np
import warp as wp

from ...sim import Control, Model, State


class SolverChysXFeatherstone:
    """ChysX-native Featherstone rigid body solver.

    Args:
        model: Newton ``Model`` with articulated rigid bodies.
        update_mass_matrix_interval: Ignored (ChysX rebuilds every step).
    """

    def __init__(self, model: Model, update_mass_matrix_interval: int = 1):
        try:
            import chysx  # noqa: PLC0415
        except ImportError as e:
            raise ImportError(
                "SolverChysXFeatherstone requires the `chysx` package."
            ) from e

        self.model = model
        self._device = wp.get_device(str(model.device))

        self._chysx_model = chysx.ArticulationModel()
        self._chysx_solver = chysx.FeatherstoneSolver()

        self._populate_model(model)
        self._chysx_solver.set_model(self._chysx_model)

        self._state_in = chysx.ArticulationState()
        self._state_out = chysx.ArticulationState()
        self._state_in.allocate(model.joint_coord_count, model.joint_dof_count)
        self._state_out.allocate(model.joint_coord_count, model.joint_dof_count)

        self._body_f_ext = wp.zeros(model.body_count, dtype=wp.spatial_vector, device=self._device)
        self._fixed_gravity = True

    def _populate_model(self, model: Model) -> None:
        """Transfer all model data from Newton to ChysX ArticulationModel."""
        m = self._chysx_model
        m.body_count = model.body_count
        m.joint_count = model.joint_count
        m.articulation_count = model.articulation_count
        m.joint_coord_count = model.joint_coord_count
        m.joint_dof_count = model.joint_dof_count

        m.upload_joint_type(model.joint_type.numpy().astype(np.int32))
        m.upload_joint_parent(model.joint_parent.numpy().astype(np.int32))
        m.upload_joint_child(model.joint_child.numpy().astype(np.int32))
        m.upload_joint_q_start(model.joint_q_start.numpy().astype(np.int32))
        m.upload_joint_qd_start(model.joint_qd_start.numpy().astype(np.int32))
        m.upload_joint_ancestor(model.joint_ancestor.numpy().astype(np.int32))

        axis_np = model.joint_axis.numpy().reshape(-1, 3).astype(np.float32)
        m.upload_joint_axis(np.ascontiguousarray(axis_np))

        dof_dim = model.joint_dof_dim.numpy().astype(np.int32).reshape(-1)
        m.upload_joint_dof_dim(np.ascontiguousarray(dof_dim))

        xp = model.joint_X_p.numpy().reshape(-1, 7).astype(np.float32)
        m.upload_joint_X_p(np.ascontiguousarray(xp))
        xc = model.joint_X_c.numpy().reshape(-1, 7).astype(np.float32)
        m.upload_joint_X_c(np.ascontiguousarray(xc))

        com = model.body_com.numpy().reshape(-1, 3).astype(np.float32)
        m.upload_body_com(np.ascontiguousarray(com))

        inertia = model.body_inertia.numpy().reshape(-1, 3, 3).astype(np.float32)
        m.upload_body_inertia(np.ascontiguousarray(inertia))

        m.upload_body_mass(model.body_mass.numpy().astype(np.float32))
        m.upload_body_flags(model.body_flags.numpy().astype(np.int32))
        m.upload_body_world(model.body_world.numpy().astype(np.int32))

        m.upload_joint_target_ke(model.joint_target_ke.numpy().astype(np.float32))
        m.upload_joint_target_kd(model.joint_target_kd.numpy().astype(np.float32))
        m.upload_joint_limit_lower(model.joint_limit_lower.numpy().astype(np.float32))
        m.upload_joint_limit_upper(model.joint_limit_upper.numpy().astype(np.float32))
        m.upload_joint_limit_ke(model.joint_limit_ke.numpy().astype(np.float32))
        m.upload_joint_limit_kd(model.joint_limit_kd.numpy().astype(np.float32))

        armature = model.joint_armature.numpy().astype(np.float32).copy()
        body_flags = model.body_flags.numpy()
        joint_child = model.joint_child.numpy()
        joint_qd_start = model.joint_qd_start.numpy()
        for j in range(model.joint_count):
            child = joint_child[j]
            if body_flags[child] & 2:  # KINEMATIC
                ds = joint_qd_start[j]
                de = joint_qd_start[j + 1]
                armature[ds:de] = 1.0e10
        m.upload_joint_armature(armature)

        art_start = model.articulation_start.numpy().astype(np.int32)
        m.upload_articulation_start(np.ascontiguousarray(art_start))

        gravity = model.gravity.numpy().reshape(-1, 3).astype(np.float32)
        m.upload_gravity(np.ascontiguousarray(gravity))

        self._setup_descendant_info(model)

    def _setup_descendant_info(self, model: Model) -> None:
        """Identify descendant FREE/DISTANCE joints for body pose correction."""
        joint_type = model.joint_type.numpy()
        joint_parent = model.joint_parent.numpy()
        art_start = model.articulation_start.numpy()

        desc_indices = []
        desc_art_ids = []
        desc_joint_starts = []

        for j in range(model.joint_count):
            jt = int(joint_type[j])
            if jt not in (4, 5):  # FREE=4, DISTANCE=5
                continue
            if int(joint_parent[j]) < 0:
                continue
            for a in range(model.articulation_count):
                if art_start[a] <= j < art_start[a + 1]:
                    desc_indices.append(j)
                    desc_art_ids.append(a)
                    desc_joint_starts.append(j)
                    break

        m = self._chysx_model
        if desc_indices:
            m.upload_descendant_info(
                np.array(desc_indices, dtype=np.int32),
                np.array(desc_art_ids, dtype=np.int32),
                np.array(desc_joint_starts, dtype=np.int32),
            )
        else:
            m.n_descendant_free_distance = 0

    def step(
        self,
        state_in: State,
        state_out: State,
        control: Control | None,
        contacts,
        dt: float,
    ) -> None:
        model = self.model
        if model.joint_count == 0:
            return

        if control is None:
            control = model.control(clone_variables=False)

        stream = wp.get_stream(self._device).cuda_stream

        if not self._fixed_gravity:
            n_worlds = model.gravity.shape[0] if len(model.gravity.shape) > 0 else 1
            self._chysx_model.sync_gravity_from_ptr(model.gravity.ptr, n_worlds)

        # Copy joint_q and joint_qd from Newton state to ChysX state (device→device)
        self._state_in.upload_joint_q(state_in.joint_q.ptr, model.joint_coord_count)
        self._state_in.upload_joint_qd(state_in.joint_qd.ptr, model.joint_dof_count)
        self._state_out.upload_joint_q(state_in.joint_q.ptr, model.joint_coord_count)
        self._state_out.upload_joint_qd(state_in.joint_qd.ptr, model.joint_dof_count)

        # Prepare external body forces
        wp.copy(self._body_f_ext, state_in.body_f)

        self._chysx_solver.step(
            state_in=self._state_in,
            state_out=self._state_out,
            target_pos_ptr=control.joint_target_pos.ptr,
            target_vel_ptr=control.joint_target_vel.ptr,
            joint_f_ptr=control.joint_f.ptr,
            body_f_ext_ptr=self._body_f_ext.ptr,
            dt=float(dt),
            cuda_stream=stream,
        )

        # Copy results back to Newton state (device→device via pybind11 helpers)
        self._state_out.copy_joint_q_to(state_out.joint_q.ptr, model.joint_coord_count)
        self._state_out.copy_joint_qd_to(state_out.joint_qd.ptr, model.joint_dof_count)

        if model.body_count > 0:
            self._chysx_solver.copy_body_q_to(state_out.body_q.ptr)
            self._chysx_solver.copy_body_qd_to(state_out.body_qd.ptr)
