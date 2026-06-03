# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""ChysX VBD solver for coupled rigid-soft scenes.

Drop-in replacement for ``SolverVBD(integrate_with_external_rigid_solver=True)``
that routes the particle VBD solve through ChysX's CUDA C++ backend while
using Newton for rigid bodies (Featherstone / IK), collision detection
(``CollisionPipeline``), and model data.

Architecture per substep:
    1. (Caller) Featherstone → state.body_q, state.body_qd
    2. (Caller) CollisionPipeline → contacts
    3. (This)   CoupledSimulator VBD → particle_q, particle_qd
"""

from __future__ import annotations

import math

import numpy as np
import warp as wp

from ...core.types import override
from ...sim import Contacts, Control, Model, State
from ..solver import SolverBase


class SolverChysXCoupled(SolverBase):
    """ChysX VBD particle solver with body-particle contact coupling.

    Rigid body integration is performed externally (e.g. Featherstone);
    this solver only handles particle VBD + body-particle contact forces.

    Args:
        model: Newton ``Model`` with particles and (optionally) rigid bodies.
        iterations: VBD Gauss-Seidel iterations per substep.
        friction_epsilon: IPC friction regularisation velocity [m/s].
        use_native_collision: If True, register shapes from the model
            and use ChysX's internal collision pipeline instead of
            Newton's ``CollisionPipeline``.  Pass ``contacts=None``
            in :meth:`step` to activate.
        soft_contact_margin: Collision margin [m] for native pipeline.
    """

    def __init__(
        self,
        model: Model,
        iterations: int = 5,
        friction_epsilon: float = 1e-2,
        use_native_collision: bool = False,
        soft_contact_margin: float = 0.01,
    ):
        super().__init__(model=model)

        try:
            import chysx  # noqa: PLC0415
        except ImportError as e:
            raise ImportError(
                "SolverChysXCoupled requires the `chysx` package. "
                "Install from ChysX/: `uv pip install ./ChysX --no-build-isolation`."
            ) from e

        self._device = wp.get_device(str(model.device))
        self._iterations = iterations
        self._friction_epsilon = friction_epsilon
        self._use_native_collision = use_native_collision
        self._soft_contact_margin = soft_contact_margin

        g_np = model.gravity.numpy().reshape(-1, 3)[0]
        self._gravity = (float(g_np[0]), float(g_np[1]), float(g_np[2]))

        self._sim = chysx.CoupledSimulator()

        if (
            getattr(model, "tet_indices", None) is not None
            and model.tet_count > 0
            and model.particle_count > 0
        ):
            tet_idx = model.tet_indices.numpy().reshape(-1, 4).astype(np.int32)
            tet_mat_raw = model.tet_materials.numpy().reshape(-1, 3).astype(np.float32)

            self._tets_np = np.ascontiguousarray(tet_idx)
            self._mats_np = np.ascontiguousarray(tet_mat_raw)
            self._n_tets = int(self._tets_np.shape[0])

            if hasattr(model, "particle_colors") and model.particle_colors is not None:
                newton_colors = model.particle_colors.numpy().astype(np.int32)
                self._sim.set_coloring(
                    np.ascontiguousarray(newton_colors),
                    self._tets_np,
                    model.particle_count,
                )
            else:
                self._sim.build_coloring(self._tets_np, model.particle_count)

            self._tet_indices_wp = wp.array(self._tets_np, dtype=wp.int32, device=self._device)
            self._build_tet_poses(model)
        else:
            self._n_tets = 0
            self._tet_indices_wp = None
            self._tet_poses_wp = None
            self._tet_materials_wp = None

        if hasattr(model, "particle_colors") and model.particle_colors is not None:
            self._particle_colors_wp = wp.array(
                model.particle_colors.numpy().astype(np.int32),
                dtype=wp.int32,
                device=self._device,
            )
        elif self._sim.num_colors() > 0:
            self._build_particle_colors(model)

        # Per-contact AVBD penalty/material arrays (allocated on first contact)
        self._contact_penalty_k: wp.array | None = None
        self._contact_material_kd: wp.array | None = None
        self._contact_material_mu: wp.array | None = None
        self._last_contact_max: int = 0

        if use_native_collision:
            self._register_collision_shapes(model)

    def _build_tet_poses(self, model: Model) -> None:
        """Compute Dm_inv for each tet from rest positions."""
        pos_np = model.particle_q.numpy().reshape(-1, 3)
        tets = self._tets_np

        n_tets = tets.shape[0]
        Dm_inv = np.zeros((n_tets, 9), dtype=np.float32)

        for t in range(n_tets):
            v0 = pos_np[tets[t, 0]]
            v1 = pos_np[tets[t, 1]]
            v2 = pos_np[tets[t, 2]]
            v3 = pos_np[tets[t, 3]]

            # Ds = [v1-v0, v2-v0, v3-v0] as column vectors
            Ds = np.column_stack([v1 - v0, v2 - v0, v3 - v0])
            try:
                Bm = np.linalg.inv(Ds)
            except np.linalg.LinAlgError:
                Bm = np.eye(3, dtype=np.float32)

            # Row-major 3x3
            Dm_inv[t] = Bm.flatten()

        self._tet_poses_wp = wp.array(
            Dm_inv.reshape(-1, 3, 3), dtype=wp.mat33, device=self._device
        )
        self._tet_materials_wp = wp.array(
            self._mats_np, dtype=wp.vec3, device=self._device
        )

    def _build_particle_colors(self, model: Model) -> None:
        """Build particle color assignment from the ChysX coloring."""
        # ChysX's coloring built during build_coloring — but we need
        # the per-particle color array on the GPU. The easiest approach
        # is to reconstruct from the same topology.
        n = model.particle_count
        tets = self._tets_np

        # Rebuild the same greedy coloring on the Python side to get
        # per-particle colors. (This mirrors ChysX C++ build_coloring.)
        from collections import defaultdict
        import heapq

        edges = set()
        for t in range(tets.shape[0]):
            verts = tets[t]
            for a in range(4):
                for b in range(a + 1, 4):
                    u, v = int(verts[a]), int(verts[b])
                    if u > v:
                        u, v = v, u
                    edges.add((u, v))

        adj = defaultdict(list)
        for u, v in edges:
            adj[u].append(v)
            adj[v].append(u)

        # Largest-degree-first ordering
        order = sorted(range(n), key=lambda x: len(adj[x]), reverse=True)
        colors = [-1] * n
        for idx in order:
            neighbor_colors = {colors[nb] for nb in adj[idx] if colors[nb] >= 0}
            c = 0
            while c in neighbor_colors:
                c += 1
            colors[idx] = c

        self._particle_colors_wp = wp.array(
            np.array(colors, dtype=np.int32), dtype=wp.int32, device=self._device
        )

    def _register_collision_shapes(self, model: Model) -> None:
        """Register all shapes from the model into ChysX's collision pipeline."""
        shape_body_np = model.shape_body.numpy() if model.shape_body is not None else np.zeros(0, dtype=np.int32)
        shape_type_np = model.shape_type.numpy() if model.shape_type is not None else np.zeros(0, dtype=np.int32)
        shape_scale_np = model.shape_scale.numpy().reshape(-1, 3) if model.shape_scale is not None else np.zeros((0, 3), dtype=np.float32)
        shape_transform_np = model.shape_transform.numpy().reshape(-1, 7) if model.shape_transform is not None else np.zeros((0, 7), dtype=np.float32)
        shape_flags_np = model.shape_flags.numpy() if hasattr(model, "shape_flags") and model.shape_flags is not None else np.zeros(0, dtype=np.int32)

        has_source_ptr = hasattr(model, "shape_source_ptr") and model.shape_source_ptr is not None
        source_ptr_np = model.shape_source_ptr.numpy() if has_source_ptr else np.zeros(0, dtype=np.uint64)

        mat_ke_np = model.shape_material_ke.numpy() if hasattr(model, "shape_material_ke") and model.shape_material_ke is not None else np.zeros(0, dtype=np.float32)
        mat_kd_np = model.shape_material_kd.numpy() if hasattr(model, "shape_material_kd") and model.shape_material_kd is not None else np.zeros(0, dtype=np.float32)
        mat_mu_np = model.shape_material_mu.numpy() if hasattr(model, "shape_material_mu") and model.shape_material_mu is not None else np.zeros(0, dtype=np.float32)

        for i in range(model.shape_count):
            body = int(shape_body_np[i]) if i < len(shape_body_np) else -1
            geo_type = int(shape_type_np[i]) if i < len(shape_type_np) else 0
            scale = shape_scale_np[i].astype(np.float32) if i < len(shape_scale_np) else np.zeros(3, dtype=np.float32)
            tf = np.ascontiguousarray(shape_transform_np[i].astype(np.float32)) if i < len(shape_transform_np) else np.zeros(7, dtype=np.float32)
            flags = int(shape_flags_np[i]) if i < len(shape_flags_np) else 0
            mesh_id = int(source_ptr_np[i]) if has_source_ptr and i < len(source_ptr_np) else 0

            m_ke = float(mat_ke_np[i]) if i < len(mat_ke_np) else 0.0
            m_kd = float(mat_kd_np[i]) if i < len(mat_kd_np) else 0.0
            m_mu = float(mat_mu_np[i]) if i < len(mat_mu_np) else 0.0

            self._sim.add_collision_shape(
                body=body,
                geo_type=geo_type,
                sx=float(scale[0]),
                sy=float(scale[1]),
                sz=float(scale[2]),
                local_tf=tf,
                flags=flags,
                mesh_id=mesh_id,
                mat_ke=m_ke,
                mat_kd=m_kd,
                mat_mu=m_mu,
            )

        max_soft_contacts = model.particle_count * 4 if model.particle_count > 0 else 64 * 1024
        self._sim.finalize_collision(max_soft_contacts)

    def _ensure_contact_buffers(self, max_contacts: int) -> None:
        """Allocate/resize per-contact AVBD arrays."""
        if max_contacts <= self._last_contact_max:
            return
        self._contact_penalty_k = wp.zeros(max_contacts, dtype=float, device=self._device)
        self._contact_material_kd = wp.zeros(max_contacts, dtype=float, device=self._device)
        self._contact_material_mu = wp.zeros(max_contacts, dtype=float, device=self._device)
        self._last_contact_max = max_contacts

    def rebuild_bvh(self, state: State) -> None:
        """No-op: CollisionPipeline handles BVH externally."""

    @override
    def step(
        self,
        state_in: State,
        state_out: State,
        control: Control | None,
        contacts: Contacts | None,
        dt: float,
    ) -> None:
        del control

        n = self.model.particle_count
        if n == 0:
            return

        if not self._device.is_cuda:
            raise RuntimeError("SolverChysXCoupled requires a CUDA device.")

        # Copy particle state (double-buffer convention)
        if state_out is not state_in:
            wp.copy(state_out.particle_q, state_in.particle_q)
            wp.copy(state_out.particle_qd, state_in.particle_qd)

        stream = wp.get_stream(self._device).cuda_stream

        model = self.model
        gx, gy, gz = self._gravity

        # Native collision path: contacts=None + use_native_collision
        if contacts is None and self._use_native_collision:
            self._step_native_collision(state_in, state_out, dt, stream)
            return

        # External contacts path (Newton CollisionPipeline)
        self._step_external_contacts(state_in, state_out, contacts, dt, stream)

    def _step_native_collision(
        self, state_in: State, state_out: State, dt: float, stream: int,
    ) -> None:
        """VBD step with ChysX-internal collision detection."""
        model = self.model
        n = model.particle_count
        gx, gy, gz = self._gravity

        body_q_ptr = 0
        body_q_prev_ptr = 0
        n_bodies = 0
        if model.body_count > 0:
            body_q_ptr = state_out.body_q.ptr
            body_q_prev_ptr = state_in.body_q.ptr
            n_bodies = model.body_count

        particle_radius_ptr = 0
        if model.particle_radius is not None:
            particle_radius_ptr = model.particle_radius.ptr

        particle_flags_ptr = 0
        if hasattr(model, "particle_flags") and model.particle_flags is not None:
            particle_flags_ptr = model.particle_flags.ptr

        self._sim.step_with_collision(
            pos_ptr=state_out.particle_q.ptr,
            vel_ptr=state_out.particle_qd.ptr,
            inv_mass_ptr=model.particle_inv_mass.ptr,
            n_particles=n,
            tet_indices_ptr=self._tet_indices_wp.ptr if self._tet_indices_wp is not None else 0,
            tet_poses_ptr=self._tet_poses_wp.ptr if self._tet_poses_wp is not None else 0,
            tet_materials_ptr=self._tet_materials_wp.ptr if self._tet_materials_wp is not None else 0,
            n_tets=self._n_tets,
            gx=gx, gy=gy, gz=gz,
            dt=float(dt),
            iterations=self._iterations,
            body_q_ptr=body_q_ptr,
            body_q_prev_ptr=body_q_prev_ptr,
            n_bodies=n_bodies,
            particle_radius_ptr=particle_radius_ptr,
            particle_flags_ptr=particle_flags_ptr,
            margin=self._soft_contact_margin,
            friction_epsilon=self._friction_epsilon,
            soft_contact_ke=float(model.soft_contact_ke),
            soft_contact_kd=float(model.soft_contact_kd),
            soft_contact_mu=float(model.soft_contact_mu),
            cuda_stream=stream,
        )

    def _step_external_contacts(
        self,
        state_in: State,
        state_out: State,
        contacts: Contacts | None,
        dt: float,
        stream: int,
    ) -> None:
        """VBD step with externally-provided contacts (Newton CollisionPipeline)."""
        model = self.model
        n = model.particle_count
        gx, gy, gz = self._gravity

        contact_particle_ptr = 0
        contact_count_ptr = 0
        contact_max = 0
        contact_ke_ptr = 0
        contact_kd_ptr = 0
        contact_mu_ptr = 0
        contact_shape_ptr = 0
        contact_body_pos_ptr = 0
        contact_body_vel_ptr = 0
        contact_normal_ptr = 0

        body_q_ptr = 0
        body_q_prev_ptr = 0
        body_qd_ptr = 0
        body_com_ptr = 0
        shape_body_ptr = 0
        particle_radius_ptr = 0
        particle_colors_ptr = 0
        n_bodies = 0
        n_shapes = 0

        if contacts is not None and hasattr(contacts, "soft_contact_count"):
            scm = contacts.soft_contact_max
            if scm > 0:
                self._ensure_contact_buffers(scm)
                self._init_contact_materials(contacts, model)

                contact_particle_ptr = contacts.soft_contact_particle.ptr
                contact_count_ptr = contacts.soft_contact_count.ptr
                contact_max = scm
                contact_ke_ptr = self._contact_penalty_k.ptr
                contact_kd_ptr = self._contact_material_kd.ptr
                contact_mu_ptr = self._contact_material_mu.ptr
                contact_shape_ptr = contacts.soft_contact_shape.ptr
                contact_body_pos_ptr = contacts.soft_contact_body_pos.ptr
                contact_body_vel_ptr = contacts.soft_contact_body_vel.ptr
                contact_normal_ptr = contacts.soft_contact_normal.ptr

        if model.body_count > 0:
            body_q_ptr = state_out.body_q.ptr
            body_q_prev_ptr = state_in.body_q.ptr
            body_qd_ptr = state_out.body_qd.ptr
            body_com_ptr = model.body_com.ptr
            n_bodies = model.body_count

        if model.shape_count > 0 and model.shape_body is not None:
            shape_body_ptr = model.shape_body.ptr
            n_shapes = model.shape_count

        if model.particle_radius is not None:
            particle_radius_ptr = model.particle_radius.ptr

        if hasattr(self, "_particle_colors_wp") and self._particle_colors_wp is not None:
            particle_colors_ptr = self._particle_colors_wp.ptr

        self._sim.step(
            pos_ptr=state_out.particle_q.ptr,
            vel_ptr=state_out.particle_qd.ptr,
            inv_mass_ptr=model.particle_inv_mass.ptr,
            n_particles=n,
            tet_indices_ptr=self._tet_indices_wp.ptr if self._tet_indices_wp is not None else 0,
            tet_poses_ptr=self._tet_poses_wp.ptr if self._tet_poses_wp is not None else 0,
            tet_materials_ptr=self._tet_materials_wp.ptr if self._tet_materials_wp is not None else 0,
            n_tets=self._n_tets,
            gx=gx, gy=gy, gz=gz,
            dt=float(dt),
            iterations=self._iterations,
            friction_epsilon=self._friction_epsilon,
            contact_particle_ptr=contact_particle_ptr,
            contact_count_ptr=contact_count_ptr,
            contact_max=contact_max,
            contact_ke_ptr=contact_ke_ptr,
            contact_kd_ptr=contact_kd_ptr,
            contact_mu_ptr=contact_mu_ptr,
            contact_shape_ptr=contact_shape_ptr,
            contact_body_pos_ptr=contact_body_pos_ptr,
            contact_body_vel_ptr=contact_body_vel_ptr,
            contact_normal_ptr=contact_normal_ptr,
            body_q_ptr=body_q_ptr,
            body_q_prev_ptr=body_q_prev_ptr,
            body_qd_ptr=body_qd_ptr,
            body_com_ptr=body_com_ptr,
            shape_body_ptr=shape_body_ptr,
            particle_radius_ptr=particle_radius_ptr,
            particle_colors_ptr=particle_colors_ptr,
            n_bodies=n_bodies,
            n_shapes=n_shapes,
            cuda_stream=stream,
        )

    def _init_contact_materials(self, contacts: Contacts, model: Model) -> None:
        """Fill per-contact penalty_k, kd, mu from model materials.

        Mirrors Newton's ``init_body_particle_contacts`` kernel:
        arithmetic mean for ke/kd, geometric mean for mu.
        """
        wp.launch(
            kernel=_init_body_particle_contact_materials_kernel,
            dim=contacts.soft_contact_max,
            inputs=[
                contacts.soft_contact_count,
                contacts.soft_contact_shape,
                model.soft_contact_ke,
                model.soft_contact_kd,
                model.soft_contact_mu,
                model.shape_material_ke,
                model.shape_material_kd,
                model.shape_material_mu,
            ],
            outputs=[
                self._contact_penalty_k,
                self._contact_material_kd,
                self._contact_material_mu,
            ],
            device=self._device,
        )


@wp.kernel
def _init_body_particle_contact_materials_kernel(
    contact_count: wp.array[wp.int32],
    contact_shape: wp.array[wp.int32],
    soft_contact_ke: float,
    soft_contact_kd: float,
    soft_contact_mu: float,
    shape_material_ke: wp.array[float],
    shape_material_kd: wp.array[float],
    shape_material_mu: wp.array[float],
    out_ke: wp.array[float],
    out_kd: wp.array[float],
    out_mu: wp.array[float],
):
    tid = wp.tid()
    if tid >= contact_count[0]:
        return

    shape_idx = contact_shape[tid]
    if shape_idx < 0:
        return

    ke0 = soft_contact_ke
    kd0 = soft_contact_kd
    mu0 = soft_contact_mu
    ke1 = shape_material_ke[shape_idx]
    kd1 = shape_material_kd[shape_idx]
    mu1 = shape_material_mu[shape_idx]

    out_ke[tid] = 0.5 * (ke0 + ke1)
    out_kd[tid] = 0.5 * (kd0 + kd1)
    out_mu[tid] = wp.sqrt(mu0 * mu1)
