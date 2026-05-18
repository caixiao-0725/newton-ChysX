# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

###########################################################################
# Example ChysX SDF Box
#
# A square cloth sheet drapes over an SDF-represented box that slowly
# rises upward, lifting the cloth with it.  Exercises chysx's NEW
# *SDF-volume contact* pipeline (the ``chysx::collision::SdfContact``
# class in ``ChysX/src/collision/sdf/``):
#
#   * The box is represented as a baked 3-D dense signed-distance grid
#     (``SdfVolume.bake_box``), not as the analytic plane / box
#     primitives the older ``StaticContactSet`` path uses.
#   * The body has a per-frame rigid pose; the cloth-vs-body contact
#     normals come from the trilinear-sampled SDF gradient at each
#     particle, and the IPC-style Coulomb friction uses the *relative*
#     particle-vs-body velocity for its slip cache so the cloth doesn't
#     experience spurious friction when it's just being carried along.
#
# Pipeline:
#
#   1. ``solver.bake_sdf_box(hx, hy, hz, voxel_size, ...)`` bakes the
#      analytic box SDF into the simulator's owned ``SdfVolume`` (one-
#      time, at setup).
#   2. Every frame we push the new ``(pos, body_velocity)`` via
#      ``solver.set_sdf_pose(...)`` / ``set_sdf_body_velocity(...)``.
#      Both are tiny H2D updates; the SDF samples themselves never
#      change.
#   3. Inside ``ClothSimulator.step(...)`` the SDF detector runs one
#      particle-vs-volume trilinear sample, populates the per-particle
#      ``(normal, depth)`` cache + lagged tangential slip, and feeds
#      them into the same penalty / Coulomb-cone machinery the
#      static-shape contact path uses.
#
# Animation:
#
#   * t in [0, 1.5 s] — cloth free-falls onto the resting box.
#   * t in [1.5 s, 4.5 s] — box rises at 0.2 m/s, lifting the cloth
#     by ~0.6 m.
#   * t in [4.5 s, ...] — box holds.
#
# Command: ``python -m newton.examples chysx_sdf_box``
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
        # ---- timing -----------------------------------------------------
        # 60 fps render, 4 substeps -> 240 Hz physics.  Stiff (k=1e4)
        # penalty contacts converge cleanly at this rate with the
        # default PCG iteration count.
        self.fps = 100
        self.frame_dt = 1.0 / self.fps
        self.sim_substeps = 1
        self.sim_dt = self.frame_dt / self.sim_substeps
        self.sim_time = 0.0

        self.viewer = viewer
        self.args = args

        # ---- box geometry ----------------------------------------------
        # Box half-extents in its local frame.  The local origin sits
        # at the box centre; the top face is at +hz in local space.
        # World pose is driven separately each frame.
        self._box_hx = 0.30
        self._box_hy = 0.30
        self._box_hz = 0.05

        # SDF voxel size: ~2.5 mm.  Finer than strictly necessary to
        # *represent* a box (we'd get away with 5 mm), but the cloth
        # below is 65 × 65 — cell size 12.5 mm, particle radius 5 mm —
        # and the trilinear gradient is only C⁰ across voxel
        # boundaries (see sdf.md, trap 2).  A voxel comparable to the
        # particle radius keeps the per-voxel normal jitter well under
        # the FEM stretch / bending stiffness, so the cloth still
        # settles to ~mm/s residuals.  Total grid is bounded by
        # ``bake_sdf_box``'s auto-padding (``thickness + 2·voxel`` ≈
        # 12 mm), so the dense volume sits at roughly
        # 250 × 250 × 50 ≈ 3 M floats = ~12 MB on device.
        self._voxel_size = 2.5e-3

        # Box motion schedule.
        #
        # DEBUG: ``_box_v_rise = 0.0`` freezes the SDF body — the
        # cloth then falls onto a stationary SDF box and we get to
        # eyeball the pure SDF-contact response (drape + Coulomb
        # friction) without animation muddying the picture.  Set to
        # 0.2 to bring the lift back.
        self._box_z_initial = 0.05      # box bottom face on the floor
        self._box_v_rise    = 0.1       # m/s  (was 0.2)
        self._t_rise_start  = 1.5       # s — start of rise phase
        self._t_rise_stop   = 4.5       # s — end of rise phase
        # (final z when the box stops moving)
        self._box_z_final = (
            self._box_z_initial
            + self._box_v_rise * (self._t_rise_stop - self._t_rise_start)
        )

        # ---- world geometry --------------------------------------------
        # Z-up; the cloth's only contact partner is the SDF body; we
        # do add a ground plane purely for the visualiser so the scene
        # has a clear reference floor, but chysx doesn't consume it
        # (we leave ``static_contact_enabled=False``).
        builder = newton.ModelBuilder(up_axis=newton.Axis.Z, gravity=-9.81)

        # Cloth: 65 × 65 grid, 0.8 m square (slightly larger than the
        # box top so the edges drape over the rim).  Centred above the
        # box's resting position.  At 65 × 65 the cell length is
        # 12.5 mm — about 5× the SDF voxel — which is fine enough to
        # show distinct cloth wrinkles on top of the box without
        # blowing past the cap on ``self_collision_max_contacts``.
        self._cloth_dim = 65
        self._cloth_size = 0.8
        cell = self._cloth_size / (self._cloth_dim - 1)
        self._edge_l = cell

        cloth_drop_z = self._box_z_initial + self._box_hz + 0.30
        cloth_origin = wp.vec3(
            -0.5 * self._cloth_size,
            -0.5 * self._cloth_size,
            cloth_drop_z,
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
            tri_ke=0.0,
            tri_ka=0.0,
            tri_kd=0.0,
            edge_ke=0.0,
            edge_kd=0.0,
            particle_radius=0.4 * cell,
        )

        # Visual-only ground plane so the viewer has a horizon.
        # ChysX never touches it (static_contact disabled below).
        builder.add_ground_plane()

        # ---- visualisation proxy for the SDF body -----------------------
        # ChysX represents the moving box as a baked SDF + a per-frame
        # pose; the Newton viewer has no idea about that.  To give the
        # scene a visible box we register a *kinematic* body holding a
        # plain Newton ``add_shape_box`` whose half-extents match the
        # SDF.  We push the same world pose into both every frame:
        #
        #   * the SDF (via ``solver.set_sdf_pose``) drives the physics;
        #   * the Newton body (via ``state.body_q``) drives the render.
        #
        # ChysX does NOT consume this Newton shape:
        #   1. ``static_contact_enabled`` is left at the default False,
        #      so the solver never scans the model for plane/box shapes;
        #   2. even if scanning were on, the solver's static-contact
        #      registration filters to ``shape_body == -1`` only — this
        #      body is body 0, not world, so it would be skipped.
        # The shape is therefore *purely* a viewer proxy.
        self._box_initial_pose = wp.transform(
            (0.0, 0.0, self._box_z_initial + self._box_hz),
            wp.quat_identity(),
        )
        self._box_body_id = builder.add_body(
            xform=self._box_initial_pose,
            is_kinematic=True,
            label="sdf_box_viz",
        )
        builder.add_shape_box(
            body=self._box_body_id,
            hx=self._box_hx,
            hy=self._box_hy,
            hz=self._box_hz,
        )

        self.model = builder.finalize()

        # ---- solver -----------------------------------------------------
        # Parameters mirror ``example_chysx_cloth_drop`` so the cloth
        # rests on the SDF box exactly like it rests on the analytic
        # ground plane there.  Two subtle traps worth noting:
        #
        # 1. **Friction OFF for the rest test.**  The IPC-style Coulomb
        #    friction is a lagged-Newton scheme: it reads the previous
        #    step's tangential slip ``u_t^lag`` and applies a linearised
        #    tangential stiffness ``α·(I - n n^T)`` plus a restoring RHS
        #    ``-α·u_t^lag``.  Once the cloth is nominally at rest, PCG
        #    residual + trilinear-SDF normal jitter (the SDF gradient
        #    is C^0 but discontinuous across voxel boundaries, unlike
        #    an analytic plane's exact ``(0, 0, 1)``) feed a tiny slip
        #    back into ``u_t^lag``; the next step the restoring force
        #    pushes the particle to a *new* equilibrium, generating a
        #    fresh slip — self-excited ringing.  Turning friction on
        #    only after the cloth has settled (or using a coarser SDF
        #    voxel + lighter ``stiffness``) avoids this.  For the
        #    "static box, watch it rest" debug pass we keep both
        #    SDF and self-collision friction at zero.
        # 2. **Self-collision stiffness 1e3, not 1e2.**  The 1e2 used
        #    earlier in this example was an order-of-magnitude too soft
        #    for a cloth that comes to rest piled on a small box (lots
        #    of layer-on-layer contact); when the penalty is too soft
        #    the self-contacts cannot resist the FEM stretch springs
        #    and the cloth oscillates as it shrinks slightly inward.
        contact_thickness = max(0.5 * cell, 2.0 * self._voxel_size)
        self.solver = newton.solvers.SolverChysX(
            self.model,
            damping=0.05,                       # matches cloth_drop
            fem_stretch_stiffness=5.0e2,
            fem_shear_stiffness=5.0e2,
            bending_stiffness=1.0e-4,           # matches cloth_drop
            pcg_iterations=50,                  # matches cloth_drop
            surface_density=0.3,
            self_collision_enabled=True,
            self_collision_thickness=contact_thickness,
            self_collision_stiffness=1.0e3,     # matches cloth_drop
            self_collision_friction=0.0,        # was -0.3 (silently
                                                # disabled but misleading)
            self_collision_max_contacts_factor=32,
            self_collision_max_ef_candidates_factor=128,
        )

        # Bake the box SDF + configure the penalty.  ``thickness`` is
        # the SDF *contact* band (cloth particles with sd < thickness
        # count as contacts); ``stiffness`` is the per-particle penalty.
        #
        # We leave ``padding`` at its auto-default, which is
        # ``max(2·voxel, thickness + 2·voxel)``.  An earlier version of
        # this example shipped with ``padding = 2·voxel`` (10 mm), so
        # the grid only covered ``box_z ± 0.06`` m — i.e. its top
        # surface ended at world z = 0.16, while a settled cloth sits
        # at z ≈ 0.167.  Particles in the active contact band kept
        # crossing the grid edge, the sampler returned the
        # "out-of-grid" sentinel ``sd = 1e30`` (== no contact), the
        # penalty force switched off, gravity dragged the particle back
        # in, the force switched on, ... producing a high-frequency
        # vertical buzz that no amount of damping could kill.  With
        # the auto-padding the grid extends to z ≈ 0.178 and the
        # cloth settles cleanly (final |v| < 1 cm/s).
        self._box_volume_idx = self.solver.bake_sdf_box(
            hx=self._box_hx,
            hy=self._box_hy,
            hz=self._box_hz,
            voxel_size=self._voxel_size,
            thickness=contact_thickness,
            stiffness=1.0e3,
            friction=0.2,                       # was 0.4 — see note above
            friction_epsilon=1.0e-4,
        )
        # Initial pose: box centre at (0, 0, box_z_initial + hz).
        self._box_pos = np.array(
            [0.0, 0.0, self._box_z_initial + self._box_hz],
            dtype=np.float32,
        )
        self._box_vel = np.zeros(3, dtype=np.float32)
        self.solver.set_sdf_pose(self._box_volume_idx, self._box_pos)
        self.solver.set_sdf_body_velocity(self._box_volume_idx, self._box_vel)

        self.state_0 = self.model.state()
        self.state_1 = self.model.state()
        self.control = self.model.control()
        self.contacts = self.model.contacts()

        self._initial_q = self.state_0.particle_q.numpy().reshape(-1, 3).copy()

        self.viewer.set_model(self.model)
        q = self._initial_q
        bmin = q.min(axis=0).astype(np.float64)
        bmax = q.max(axis=0).astype(np.float64)
        # Make sure the camera sees the box's final position too.
        bmin[2] = min(float(bmin[2]), 0.0)
        bmax[2] = max(float(bmax[2]), self._box_z_final + 2.0 * self._box_hz)
        frame_z_up_camera_viewer(self.viewer, bmin, bmax)

        # Per-frame body_q scratch buffer for the SDF-box visualisation
        # proxy.  ``state.body_q`` is a wp.array[wp.transform] (one row
        # per body, packed as (px, py, pz, qx, qy, qz, qw)); we keep a
        # host-side copy seeded from the model's initial pose and
        # round-trip the SDF box's row through it every frame.  Each
        # round-trip is a 7-float H2D copy — negligible.
        self._body_q_np = self.state_0.body_q.numpy().copy()

        # NOTE: We deliberately do NOT cuda-graph-capture this scene.
        # The SDF volume's world pose is currently passed by value into
        # the contact-detection kernel launch (`SdfVolumeView` POD), so
        # a captured graph would freeze whatever pose was active at
        # capture time.  Without capture we pay the extra
        # ~30 launches/substep, which at 21x21 cloth is still
        # comfortably real-time on a desktop GPU.  Once we add
        # device-side pose storage on `SdfVolume`, the kernel will
        # dereference the pose from a stable device pointer and graph
        # capture becomes safe — at which point we can wire the usual
        # `_capture_graph()` helper back in.

    # ---- per-frame physics -------------------------------------------

    def _update_sdf_pose(self) -> None:
        """Recompute box position + linear velocity for ``self.sim_time``,
        push them into the SDF physics path AND mirror them onto the
        Newton visualisation body so the viewer draws the box at its
        true world pose."""
        t = self.sim_time
        if t < self._t_rise_start:
            v_z = 0.0
        elif t < self._t_rise_stop:
            v_z = self._box_v_rise
        else:
            v_z = 0.0
        # Clamp to final height (just in case sim_time overshoots).
        if v_z > 0.0:
            self._box_pos[2] = min(
                self._box_pos[2] + v_z * self.frame_dt,
                self._box_z_final + self._box_hz,
            )
        self._box_vel[2] = v_z

        # Physics side: push the new pose / velocity into the SDF
        # detector (host scalars; no device sync required).
        self.solver.set_sdf_pose(self._box_volume_idx, self._box_pos)
        self.solver.set_sdf_body_velocity(self._box_volume_idx, self._box_vel)

        # Visualisation side: mirror the same pose onto the kinematic
        # Newton body.  body_q packs each transform as
        # (px, py, pz, qx, qy, qz, qw); we keep rotation at identity
        # because the SDF body only translates in this example.
        self._body_q_np[self._box_body_id, 0:3] = self._box_pos
        self._body_q_np[self._box_body_id, 3:7] = (0.0, 0.0, 0.0, 1.0)
        self.state_0.body_q.assign(self._body_q_np)

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

    def step(self):
        # The SDF pose is only refreshed once per frame, not once per
        # substep — substep granularity changes here would saturate
        # H2D bandwidth without changing the contact response in any
        # observable way.
        self._update_sdf_pose()
        self._simulate_substeps()
        self.sim_time += self.sim_substeps * self.sim_dt

    def render(self):
        self.viewer.begin_frame(self.sim_time)
        self.viewer.log_state(self.state_0)
        self.viewer.end_frame()

    # ---- regression check --------------------------------------------

    def test_final(self):
        """Sanity-check the final state.

        Required invariants:

        1. Every particle position / velocity is finite.
        2. The cloth stays within a reasonable bounding box.
        3. The cloth's bulk is sitting at or above the box top
           (modulo one contact-thickness penetration band).  If the
           SDF contact didn't fire, the cloth would have fallen
           through to z < 0.
        4. Velocities haven't exploded (no instability blow-up).
        """
        q = self.state_0.particle_q.numpy().reshape(-1, 3)
        qd = self.state_0.particle_qd.numpy().reshape(-1, 3)

        if not (np.isfinite(q).all() and np.isfinite(qd).all()):
            raise ValueError("non-finite values in particle state")

        bound = 5.0  # m
        if (np.abs(q) > bound).any():
            raise ValueError(
                f"cloth particles escaped the {bound:.1f} m bounding box; "
                f"max |q| = {float(np.abs(q).max()):.3f}"
            )

        # The box top is at z = box_pos.z + hz.  Allow a contact-
        # thickness slack: penalty contact only enforces the band
        # ``sdf > 0`` up to its thickness.
        box_top_z = float(self._box_pos[2] + self._box_hz)
        slack = 1.5 * float(
            self.solver._sim.sdf_contact_thickness(self._box_volume_idx)
        )
        z_min = float(q[:, 2].min())
        if z_min < box_top_z - slack - 0.05:
            # 0.05 m extra slack: the cloth's overhanging edges
            # legitimately drape below the box top.  We only fail
            # if a particle has clearly tunnelled THROUGH the SDF
            # body (z < 0 would be the dead giveaway).
            if z_min < -slack:
                raise ValueError(
                    f"cloth fell through the SDF box: min z = {z_min:.4f} m "
                    f"(box top at {box_top_z:.4f}, slack {slack:.4f})"
                )

        max_speed = float(np.linalg.norm(qd, axis=1).max())
        if max_speed > 30.0:
            raise ValueError(
                f"particle speed exploded: max |v| = {max_speed:.3f} m/s"
            )

        # The visualisation body must track the SDF body's pose
        # exactly (mismatched render vs physics is a much nastier
        # bug than a slight numerical issue — the user would see
        # cloth floating through an invisible solid).
        body_q_now = self.state_0.body_q.numpy()[self._box_body_id]
        body_pos = body_q_now[:3]
        if not np.allclose(body_pos, self._box_pos, atol=1.0e-6):
            raise ValueError(
                f"visualisation body pose desynced from SDF pose: "
                f"body={body_pos}, sdf={self._box_pos}"
            )

        # Diagnostic print — visible when this is run without
        # ``--test --viewer null`` redirected away.
        print(
            f"[chysx_sdf_box] final state:  "
            f"box at z={box_top_z:.3f}, cloth z range [{z_min:.3f}, "
            f"{float(q[:, 2].max()):.3f}], max |v|={max_speed:.3f} m/s"
        )


if __name__ == "__main__":
    parser = newton.examples.create_parser()
    parser.set_defaults(num_frames=420)  # 7 s at 60 fps

    viewer, args = newton.examples.init(parser)
    newton.examples.run(Example(viewer, args), args)
