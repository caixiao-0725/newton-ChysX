# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

###########################################################################
# Example ChysX SDF Gripper
#
# Two thin SDF-represented boxes play the role of a parallel-jaw
# gripper, like the fingers of a robot end effector.  The cloth is
# held VERTICALLY in the yz-plane, the jaws sit one on each side
# along x, close horizontally onto the cloth, then translate upward.
# Gravity is OFF, so the only forces that move the cloth are the
# SDF contacts — penalty normals during the closing phase and
# Coulomb friction during the lift.
#
# Why vertical cloth?  When the cloth is flat (lying in the xy plane)
# it sits at a saddle of the zero-gravity potential: any in-plane
# squeeze from the jaws buckles the whole sheet out of plane, the
# cloth pops free, and you can't tell whether friction is doing its
# job.  Standing it up turns that into a well-conditioned problem:
# the cloth occupies a known 10×10 cm patch between the jaw faces,
# every contact normal is pure ±x, and a lift in +z is purely
# tangential — exactly what we want to measure.
#
# What this exercises:
#
#   * ChysX's multi-volume SDF API.  `solver.add_sdf_volume()`
#     allocates one independent (SdfVolume, SdfContact) pair per jaw;
#     `bake_sdf_box(volume_index=...)`, `set_sdf_pose(volume_index,
#     ...)` and `set_sdf_body_velocity(volume_index, ...)` route to
#     the matching slot.  Each jaw has its own contact normals,
#     friction coefficient and slip cache, so the gripper picks the
#     cloth up only because the *sum* of the two jaws' friction
#     forces beats whatever the cloth pulls back with — exactly the
#     way a real jaw-and-pad pair behaves.
#   * IPC-style Coulomb friction with a HIGH μ (= 0.8): once the
#     gripper closes, the lift translates almost entirely into cloth
#     motion (any slipping is logged by `test_final`).
#
# Animation schedule (5.5 s of physics, 60 fps render, 4-substep sim):
#
#   * t in [0.0, 0.5 s]   — both jaws stay open (gap = 0.30 m); cloth
#                           floats free in zero gravity.
#   * t in [0.5, 2.0 s]   — jaws close at 5 cm/s until the inner
#                           faces sit a thickness-band's worth of
#                           clearance apart, just pinching the cloth.
#   * t in [2.0, 2.5 s]   — jaws hold (let the closing transient
#                           bleed through friction damping).
#   * t in [2.5, 5.0 s]   — both jaws translate upward at 5 cm/s,
#                           lifting the cloth by ~12.5 cm.
#
# Command: ``python -m newton.examples chysx_sdf_gripper``
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
        # ---- timing -----------------------------------------------------
        # 60 fps render, 4 substeps -> 240 Hz physics.  The IPC
        # friction term α has units N/m and lands directly on the
        # tangential Hessian diagonal; at dt ≈ 4 ms it dominates the
        # per-particle inertial term m/dt² by ~5 orders of magnitude,
        # so the linearised solve converges the cloth's tangential
        # displacement to the body's tangential displacement in
        # essentially one step — which is what "stick" means.  At
        # coarser dt (10 ms) the IPC tangential stiffness drops
        # proportionally and the cloth visibly slips out of the
        # gripper instead of riding it up.
        self.fps = 100
        self.frame_dt = 1.0 / self.fps
        self.sim_substeps = 1
        self.sim_dt = self.frame_dt / self.sim_substeps
        self.sim_time = 0.0

        self.viewer = viewer
        self.args = args

        # ---- jaw geometry ----------------------------------------------
        # Thin pads in x (the closing axis), wide in y, modestly tall
        # in z.  Half-extents in the jaw's own local frame; the world
        # pose is pushed per frame.  The pads must be larger than the
        # cloth in y AND z, otherwise the cloth's outer rim ends up
        # squeezed by the closing pads with no in-plane room to give —
        # self-collision then buckles the whole sheet out of plane and
        # the cloth pops free of the gripper.
        self._jaw_hx = 0.012   # 24 mm thick: a credible "finger pad"
        self._jaw_hy = 0.080   # 16 cm wide   (cloth y half-size 5 cm)
        self._jaw_hz = 0.080   # 16 cm tall   (lots of vertical reach)

        # SDF voxel size: 2 mm.  Pad thickness is 24 mm so the thin
        # axis gets 12 voxels — plenty for clean trilinear gradients
        # on a feature only 24 mm thick.  The auto-padding adds
        # ``thickness + 2·voxel`` (≈ 9 mm) on every axis, keeping the
        # baked grid tiny per jaw (~6 MB).
        self._voxel_size = 2.0e-3

        # ---- jaw motion schedule (world frame) -------------------------
        # Phases are picked so the closing and lifting transients
        # don't overlap — once the jaws stop closing we wait half a
        # second for the cloth to settle into the contact band, then
        # start lifting.
        self._t_close_start =  0.5
        self._t_close_stop  =  2.0
        self._t_hold_stop   =  2.5
        self._t_lift_stop   =  5.0

        self._half_gap_open   = 0.150          # 30 cm open
        self._lift_v          = 0.05           # 5 cm/s upward
        # During the lift phase both jaws also translate along +y.
        # Keep this as an explicit travel distance so it's easy to tune.
        self._lift_y_travel   = 0.20           # doubled travel distance
        # Half-closed-gap chosen below after we know the cloth's cell
        # length (==> particle radius).

        # ---- world geometry --------------------------------------------
        # Z-up, gravity = 0:  we want the cloth to do nothing on its
        # own — the only forces that can move it are the SDF contacts
        # (penalty + friction).
        builder = newton.ModelBuilder(up_axis=newton.Axis.Z, gravity=0.0)

        # Raise the whole setup above the visual ground plane so the
        # scene stays visible in default viewer framing.
        self._scene_z_offset = 0.25

        # Cloth: 41 × 41 square, 12 cm on a side, standing VERTICALLY
        # in the yz-plane so that:
        #   * every particle starts at x = 0 — the closing axis is
        #     perpendicular to the sheet, normal load is well-defined
        #     and identical for every contact;
        #   * the cloth occupies y ∈ [-5, +5] cm and
        #     z ∈ [offset, offset + 10] cm,
        #     entirely inside the jaws' y-extent (8 cm half) and
        #     z-extent (8 cm half) when the jaw centres sit at the
        #     cloth's mid-height (z = 5 cm) — the whole sheet is
        #     pinched, no rim squeeze, no buckling.
        self._cloth_dim  = 101
        self._cloth_size = 0.3
        cell             = self._cloth_size / (self._cloth_dim - 1)
        particle_radius  = 0.4 * cell

        # Closed half-gap: tighten the pinch so the cloth gets a
        # stronger normal load from both jaws (better grip, less slip
        # during the lift).  Keep a small margin above jaw_hx to avoid
        # over-compressing particles into the SDF interior.
        contact_thickness = max(0.5 * cell, 2.0 * self._voxel_size)
        self._half_gap_closed = self._jaw_hx + 0.25 * contact_thickness

        # Jaw z-centre — both jaws are aligned with the cloth's mid-
        # height so the whole 10 cm sheet falls inside the 16 cm
        # z-extent of the pads with 3 cm overhead and 3 cm underhead.
        self._jaw_z0 = self._scene_z_offset + 0.5 * self._cloth_size

        # Rotate the default xy-plane cloth into the yz-plane.  A
        # +90° rotation about the y-axis sends the cloth-local x̂
        # to world -ẑ and leaves ŷ alone, so grid index (i, j) ends
        # up at world (0, -size/2 + j·cell, size - i·cell).
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

        # Visual-only ground plane so the viewer has a horizon.  ChysX
        # never touches it (static_contact disabled in the solver).
        builder.add_ground_plane()

        # ---- visualisation proxies for the two SDF jaws ----------------
        # Same kinematic-body-with-Newton-box-shape trick as
        # example_chysx_sdf_box: ChysX is fed the SDF, the viewer is
        # fed a matching Newton box for rendering, and we sync the
        # pose into both every frame.
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

        # ---- solver ----------------------------------------------------
        # FEM + bending parameters mirror example_chysx_cloth_drop.
        # Self-collision is OFF: the cloth starts as a flat sheet in
        # the yz-plane and the jaws only ever push it in ±x, so it
        # cannot fold onto itself in this scene — the self-collision
        # SpMV sidecar would just burn cycles producing no contacts.
        # static_contact is also OFF (the only obstacles are the
        # two SDF volumes).
        self.solver = newton.solvers.SolverChysX(
            self.model,
            damping=0.05,
            fem_stretch_stiffness=5.0e2,
            fem_shear_stiffness=5.0e2,
            bending_stiffness=1.0e-4,
            pcg_iterations=30,
            surface_density=0.3,
            self_collision_enabled=False,
            static_contact_enabled=False,
        )

        # ---- bake one SDF body per jaw ---------------------------------
        # Each jaw owns an independent (SdfVolume, SdfContact) pair.
        # We use HIGH friction (μ = 0.8) on both because the whole
        # point of the scene is "can the jaws drag the cloth purely
        # through tangential coupling?".  With μ = 0 the cloth would
        # just slide out of the gripper as soon as the jaws move up.
        # ε_u = 5e-4 is loose enough to avoid the lagged-Newton
        # self-excitation discussed in sdf.md but tight enough that
        # the slip-to-stick transition still looks crisp.
        # Rounded SDF edges avoid normal-direction jumps at box
        # edges/corners (adjacent cloth vertices seeing conflicting
        # normals), which otherwise stretches and distorts the cloth.
        corner_radius = max(2.0 * self._voxel_size, 0.5 * contact_thickness)
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
                friction=0.8,
                friction_epsilon=5.0e-4,
            )
            self._jaw_volume_ids.append(vid)

        # Per-jaw position / velocity caches in the SAME order as
        # _jaw_volume_ids / _jaw_body_ids:  index 0 = left (-x),
        # index 1 = right (+x).
        self._jaw_pos = [
            np.array([-self._half_gap_open, 0.0, self._jaw_z0], dtype=np.float32),
            np.array([+self._half_gap_open, 0.0, self._jaw_z0], dtype=np.float32),
        ]
        self._jaw_vel = [
            np.zeros(3, dtype=np.float32),
            np.zeros(3, dtype=np.float32),
        ]
        # Initial sync into the solver so the very first step doesn't
        # see whatever stale pose `bake_sdf_box` left behind.
        for i, vid in enumerate(self._jaw_volume_ids):
            self.solver.set_sdf_pose(vid, self._jaw_pos[i])
            self.solver.set_sdf_body_velocity(vid, self._jaw_vel[i])

        self.state_0 = self.model.state()
        self.state_1 = self.model.state()
        self.control = self.model.control()
        self.contacts = self.model.contacts()
        self._initial_q = self.state_0.particle_q.numpy().reshape(-1, 3).copy()

        self.viewer.set_model(self.model)
        # Frame on a box that contains the open jaws + the raised
        # cloth + overhead for the lift.
        bmin = np.array([-0.20, -0.12, 0.10], dtype=np.float64)
        bmax = np.array([+0.20, +0.32, 0.70], dtype=np.float64)
        frame_z_up_camera_viewer(self.viewer, bmin, bmax)

        # Per-frame OBJ export: cloth + both jaw boxes in one file.
        self._obj_dir = Path("frames")
        self._obj_dir.mkdir(parents=True, exist_ok=True)
        self._frame_idx = 0
        print(f"[chysx_sdf_gripper] exporting scene OBJ frames -> {self._obj_dir.resolve()}")

        # ---- CUDA Graph capture ----------------------------------------
        # The per-frame SDF pose update lives OUTSIDE the captured
        # region (set_sdf_pose pokes host-side scalars only — the
        # graph reads them at launch time via the per-volume view
        # POD), so we can safely capture the 4 sim substeps once and
        # replay every frame, even while the jaws move.
        self._cuda_graph = None
        self._capture_graph()

    # ---- per-frame motion --------------------------------------------

    def _update_jaw_poses(self) -> None:
        """Update both jaws' (position, velocity) for the current
        frame.  Pure host-side; the solver picks them up at the start
        of step().

        Phase 1 (t < t_close_start):  open and idle.
        Phase 2 (t_close_start ≤ t < t_close_stop):  close inwards at
            half-speed each so the gap closes at the full close rate.
        Phase 3 (t_close_stop ≤ t < t_hold_stop):  hold closed.
        Phase 4 (t_hold_stop ≤ t < t_lift_stop):  translate both jaws
            upward together at lift_v.
        """
        t = self.sim_time

        # Compute closed-axis (x) position and velocity for the LEFT
        # jaw — the right is mirrored, see below.
        close_duration = self._t_close_stop - self._t_close_start
        if t < self._t_close_start:
            x_l = -self._half_gap_open
            vx  = 0.0
        elif t < self._t_close_stop:
            # Both jaws close symmetrically: each one moves
            # (open - closed) / close_duration metres per second
            # toward the centre.
            close_speed = (self._half_gap_open - self._half_gap_closed) / close_duration
            x_l = -self._half_gap_open + close_speed * (t - self._t_close_start)
            vx  = +close_speed   # left jaw moves in +x while closing
        else:
            x_l = -self._half_gap_closed
            vx  = 0.0

        # Lift phase motion, applied to BOTH jaws once we've held
        # briefly:
        #   * z: move upward at constant lift_v
        #   * y: sweep forward over a fixed travel distance
        # The jaw z-centre starts at the cloth's mid-height.
        lift_duration = self._t_lift_stop - self._t_hold_stop
        if t < self._t_hold_stop:
            y_jaw = 0.0
            vy    = 0.0
            z_jaw = self._jaw_z0
            vz    = 0.0
        elif t < self._t_lift_stop:
            tau   = (t - self._t_hold_stop) / lift_duration
            y_jaw = self._lift_y_travel * tau
            vy    = self._lift_y_travel / lift_duration
            z_jaw = self._jaw_z0 + self._lift_v * (t - self._t_hold_stop)
            vz    = self._lift_v
        else:
            y_jaw = self._lift_y_travel
            vy    = 0.0
            z_jaw = self._jaw_z0 + self._lift_v * (self._t_lift_stop - self._t_hold_stop)
            vz    = 0.0

        # Left jaw (index 0) at -|x_l|, right jaw (index 1) at +|x_l|;
        # velocities mirror x and share y/z motion.
        self._jaw_pos[0][:] = (x_l,         y_jaw, z_jaw)
        self._jaw_pos[1][:] = (-x_l,        y_jaw, z_jaw)
        self._jaw_vel[0][:] = (+vx,         vy,    vz)
        self._jaw_vel[1][:] = (-vx,         vy,    vz)

        # Physics:  push into both SDF detectors.
        for i, vid in enumerate(self._jaw_volume_ids):
            self.solver.set_sdf_pose(vid, self._jaw_pos[i])
            self.solver.set_sdf_body_velocity(vid, self._jaw_vel[i])

        # Visualisation:  sync the kinematic Newton bodies' transforms
        # so the viewer's boxes track exactly where the SDF jaws are
        # in physics.  body_q packs (px, py, pz, qx, qy, qz, qw); the
        # jaws never rotate so the quaternion stays identity.
        body_q = self.state_0.body_q.numpy()
        for i, body_id in enumerate(self._jaw_body_ids):
            body_q[body_id, :3] = self._jaw_pos[i]
            body_q[body_id, 3:] = (0.0, 0.0, 0.0, 1.0)
        self.state_0.body_q.assign(body_q)

    # ---- per-frame physics -------------------------------------------

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

    # ---- scene OBJ export ---------------------------------------------

    def _build_cloth_faces(self) -> np.ndarray:
        """Build triangle connectivity for the regular cloth grid."""
        n = self._cloth_dim
        tris = np.empty((2 * (n - 1) * (n - 1), 3), dtype=np.int32)
        t = 0
        for i in range(n - 1):
            for j in range(n - 1):
                v00 = i * n + j
                v10 = (i + 1) * n + j
                v01 = i * n + (j + 1)
                v11 = (i + 1) * n + (j + 1)
                tris[t] = (v00, v10, v01)
                tris[t + 1] = (v01, v10, v11)
                t += 2
        return tris

    def _build_box_mesh(self, center: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Return 8 box vertices and 12 triangle faces."""
        hx, hy, hz = self._jaw_hx, self._jaw_hy, self._jaw_hz
        cx, cy, cz = float(center[0]), float(center[1]), float(center[2])
        verts = np.array(
            [
                [cx - hx, cy - hy, cz - hz],
                [cx + hx, cy - hy, cz - hz],
                [cx + hx, cy + hy, cz - hz],
                [cx - hx, cy + hy, cz - hz],
                [cx - hx, cy - hy, cz + hz],
                [cx + hx, cy - hy, cz + hz],
                [cx + hx, cy + hy, cz + hz],
                [cx - hx, cy + hy, cz + hz],
            ],
            dtype=np.float64,
        )
        faces = np.array(
            [
                [0, 1, 2], [0, 2, 3],  # -z
                [4, 6, 5], [4, 7, 6],  # +z
                [0, 4, 5], [0, 5, 1],  # -y
                [3, 2, 6], [3, 6, 7],  # +y
                [0, 3, 7], [0, 7, 4],  # -x
                [1, 5, 6], [1, 6, 2],  # +x
            ],
            dtype=np.int32,
        )
        return verts, faces

    def _export_scene_obj(self, path: Path) -> None:
        """Write one OBJ containing cloth + two jaw boxes."""
        path.parent.mkdir(parents=True, exist_ok=True)

        cloth_verts = self.state_0.particle_q.numpy().reshape(-1, 3).astype(np.float64)
        cloth_faces = self._build_cloth_faces()
        jaw0_verts, jaw_faces = self._build_box_mesh(self._jaw_pos[0])
        jaw1_verts, _ = self._build_box_mesh(self._jaw_pos[1])

        with path.open("w", encoding="utf-8") as f:
            f.write("# chysx_sdf_gripper scene snapshot\n")
            f.write("o cloth\n")
            np.savetxt(f, cloth_verts, fmt="v %.6f %.6f %.6f")
            np.savetxt(f, cloth_faces + 1, fmt="f %d %d %d")

            jaw0_offset = cloth_verts.shape[0]
            f.write("o sdf_jaw_0\n")
            np.savetxt(f, jaw0_verts, fmt="v %.6f %.6f %.6f")
            np.savetxt(f, jaw_faces + 1 + jaw0_offset, fmt="f %d %d %d")

            jaw1_offset = jaw0_offset + jaw0_verts.shape[0]
            f.write("o sdf_jaw_1\n")
            np.savetxt(f, jaw1_verts, fmt="v %.6f %.6f %.6f")
            np.savetxt(f, jaw_faces + 1 + jaw1_offset, fmt="f %d %d %d")

        print(f"[chysx_sdf_gripper] scene OBJ written -> {path.resolve()}")

    def _export_scene_obj_frame(self, frame_idx: int) -> None:
        path = self._obj_dir / f"chysx_sdf_gripper_scene_{frame_idx:05d}.obj"
        self._export_scene_obj(path)

    def step(self):
        # Same once-per-frame pose refresh as the SDF box example —
        # the jaws move slowly compared to the substep cadence, so
        # interpolating finer than per-frame would not change the
        # contact response noticeably and would saturate the H2D
        # pose-poke path.
        self._update_jaw_poses()
        if self._cuda_graph is not None:
            wp.capture_launch(self._cuda_graph)
        else:
            self._simulate_substeps()
        self.sim_time += self.sim_substeps * self.sim_dt
        self._export_scene_obj_frame(self._frame_idx)
        self._frame_idx += 1

    def render(self):
        self.viewer.begin_frame(self.sim_time)
        self.viewer.log_state(self.state_0)
        self.viewer.end_frame()

    # ---- regression check --------------------------------------------

    def test_final(self):
        """Validate the cloth's final state.

        The scene's invariants are:

        1. Every particle's position / velocity is finite.
        2. Nothing escapes a generous bounding box (the scene fits in
           a ~50 cm cube).
        3. The cloth's MEAN z is meaningfully above its starting
           height (cloth_size / 2 ≈ 5 cm).  This is the real "did the
           friction drag work?" check — if friction were broken the
           cloth would stay at the height where the jaws closed on
           it and not move upward at all as the jaws rise.
        4. Cloth has not slid completely out of the gripper:  the
           gap between the jaw z and the cloth mean z stays small
           compared to the lift distance.  Perfect tracking is
           unrealistic for an IPC-style friction model with non-zero
           ε_u; tens-of-percent slippage is normal.
        5. Velocities haven't exploded.
        """
        q  = self.state_0.particle_q.numpy().reshape(-1, 3)
        qd = self.state_0.particle_qd.numpy().reshape(-1, 3)

        if not (np.isfinite(q).all() and np.isfinite(qd).all()):
            raise ValueError("non-finite values in particle state")
        bound = 1.0  # m
        if (np.abs(q) > bound).any():
            raise ValueError(
                f"cloth particles escaped the {bound:.1f} m bounding box; "
                f"max |q| = {float(np.abs(q).max()):.3f}"
            )

        z_mean    = float(q[:, 2].mean())
        z_jaw     = float(self._jaw_pos[0][2])  # both jaws share z
        z_start   = self._scene_z_offset + 0.5 * self._cloth_size

        # Cloth must have risen at least 1 cm above its starting
        # height.  Without friction the cloth would sit at the
        # starting plane forever (no gravity, no normal lift in z);
        # ≥ 1 cm above is unambiguous evidence the gripper is
        # dragging the cloth upward through tangential coupling.
        if z_mean < z_start + 0.01:
            raise ValueError(
                f"cloth was not lifted by the gripper: mean z = "
                f"{z_mean*1000:.1f} mm, expected ≥ {(z_start + 0.01)*1000:.1f} "
                f"mm.  Suggests the SDF friction is wired wrong "
                f"(the IPC RHS used to have an inverted sign — see "
                f"sdf_contact.cu)."
            )

        # Loose upper bound: cloth-vs-jaw lag should be a fraction
        # of the lift distance.  In practice we see ~20-30% slip at
        # μ=0.8 with these timings.  A 5 cm lag (≈ 40% of the 12.5
        # cm lift) means the cloth is essentially slipping out.
        max_lag = 0.05
        if (z_jaw - z_mean) > max_lag:
            raise ValueError(
                f"cloth lagged the gripper by more than {max_lag*1000:.0f} mm: "
                f"jaw z = {z_jaw*1000:.1f} mm, cloth mean z = "
                f"{z_mean*1000:.1f} mm."
            )

        max_speed = float(np.linalg.norm(qd, axis=1).max())
        if max_speed > 5.0:
            raise ValueError(
                f"particle speed exploded: max |v| = {max_speed:.3f} m/s"
            )

        print(
            f"[chysx_sdf_gripper] final state:  jaw z = {z_jaw*1000:.1f} mm, "
            f"cloth mean z = {z_mean*1000:.1f} mm (lag "
            f"{(z_jaw-z_mean)*1000:.1f} mm), max |v| = {max_speed:.3f} m/s"
        )


if __name__ == "__main__":
    parser = newton.examples.create_parser()
    parser.set_defaults(num_frames=int(60 * 5.5))  # 5.5 s at 60 fps

    viewer, args = newton.examples.init(parser)
    newton.examples.run(Example(viewer, args), args)
