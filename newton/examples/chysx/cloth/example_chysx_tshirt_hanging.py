# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

###########################################################################
# Example ChysX T-shirt Hanging  (pinned collar + gravity fall)
#
# ``unisex_shirt.usd`` is authored flat with **X = width**, **Y = garment
# height (neck toward +Y)**, **Z = front–back thickness** — same as
# ``example_chysx_tshirt_drop``.  Pins are **high‑Y + near centre‑X** (neck /
# collar opening), **not** ``max(Z)`` (that pins mainly one thickness sheet /
# chest).  The garment is lifted to ``z = hang_height``.  World gravity is
# Newton Z-up ``g_z = -9.81``; ``SolverChysX(gravity=(0, 0, -9.81))`` copies
# the same vector into ChysX so the material matches ``Model.gravity``.
#
# Contacts
# --------
#
# * **DCD self-collision** (VF / EE narrow-phase + QuantBvh broadphase).
# * **EF untangle** (5-vertex ICM penalty, diagonal-only Hessian); shares
#   the same EF broadphase stream as self-collision.
#
# Contact penalty stiffness:
#
# * ``self_collision_stiffness = 1e3`` (VF / EE proximity),
# * ``untangle_stiffness = 2e3`` (= 2× proximity, style3d-style ratio).
#
# Thickness (distance band, **not** stiffness): ``self_collision_thickness`` and
# ``untangle_thickness`` are both ``1e-3`` m (1 mm).
#
# Command: ``python -m newton.examples chysx_tshirt_hanging``
#
# OBJ dump (optional): ``--obj-out DIR [--obj-stride N]`` writes
# ``tshirt_hanging_<frame>.obj`` (deformed verts + rest connectivity).
#
###########################################################################

from __future__ import annotations

import os

import numpy as np
import warp as wp
from pxr import Usd

import newton
import newton.examples
from newton.examples.chysx._camera import frame_z_up_camera_viewer
import newton.usd


def _load_centered_tshirt_m() -> tuple[np.ndarray, np.ndarray, float]:
    """Same loader as ``example_chysx_tshirt_drop``: cm -> m + bbox-centre.

    Returns ``(vertices, indices, median_edge_length)`` ready for
    :py:meth:`ModelBuilder.add_cloth_mesh`.
    """

    stage = Usd.Stage.Open(newton.examples.get_asset("unisex_shirt.usd"))
    prim = stage.GetPrimAtPath("/root/shirt")
    m = newton.usd.get_mesh(prim)

    v_cm = np.asarray(m.vertices, dtype=np.float32)
    idx = np.asarray(m.indices, dtype=np.int32).reshape(-1, 3)

    v_m = v_cm * 0.01
    centre = 0.5 * (v_m.min(axis=0) + v_m.max(axis=0))
    v_m = v_m - centre

    e = np.concatenate(
        [
            np.linalg.norm(v_m[idx[:, a]] - v_m[idx[:, b]], axis=1)
            for (a, b) in [(0, 1), (1, 2), (2, 0)]
        ]
    )
    edge_med = float(np.median(e))
    return v_m, idx, edge_med


def _pin_indices_neck_collar(positions: np.ndarray, edge_med: float) -> np.ndarray:
    """Pick collar / neck-hole vertices for ``unisex_shirt.usd`` rest pose.

    See ``example_chysx_tshirt_drop``: USD uses ``Y`` along body height
    (neck toward ``max(Y)``), ``X`` across width, ``Z`` through thickness.
    Pinning ``max(Z)`` instead fixes mostly one front/back sheet — wrong.

    We take a thin band ``y >= y_max - δ_y`` and restrict ``|x - x_mid|``
    to the centre third of the bust width so sleeves stay free while the
    neckline ring stays kinematic.

    Args:
        positions: ``(N, 3)`` world-space particle positions [m].
        edge_med: Mesh median edge length [m].

    Returns:
        1-D int32 pinned particle indices.
    """
    x = positions[:, 0].astype(np.float64)
    y = positions[:, 1].astype(np.float64)

    y_lo = float(np.min(y))
    y_hi = float(np.max(y))
    y_span = max(y_hi - y_lo, 1e-6)

    x_lo = float(np.min(x))
    x_hi = float(np.max(x))
    x_mid = 0.5 * (x_lo + x_hi)
    x_extent = max(x_hi - x_lo, 1e-6)

    tol_y = min(max(3.0 * float(edge_med), 5.0e-3), 0.05 * y_span)
    tol_y = max(tol_y, 4.0e-3)

    top = y >= y_hi - tol_y

    # Collar sits near centreline; sleeves extend far in ±X.
    half_w = max(0.16 * x_extent, 6.0 * float(edge_med))
    centre_x = np.abs(x - x_mid) <= half_w

    idx = np.nonzero(top & centre_x)[0].astype(np.int32)
    if idx.size < 14:
        half_w = max(0.28 * x_extent, 10.0 * float(edge_med))
        centre_x = np.abs(x - x_mid) <= half_w
        idx = np.nonzero(top & centre_x)[0].astype(np.int32)
    if idx.size < 10:
        idx = np.nonzero(top)[0].astype(np.int32)
    if idx.size < 8:
        thr = float(np.percentile(y, 98.8))
        idx = np.nonzero(y >= thr)[0].astype(np.int32)
    if idx.size == 0:
        idx = np.array([int(np.argmax(y))], dtype=np.int32)
    return idx


class Example:
    def __init__(self, viewer, args):
        # ---- timing -----------------------------------------------------
        # 60 fps render with 5 substeps -> 300 Hz physics.  Contacts +
        # untangle keep the linear system stiff; five substeps per frame
        # stays stable through the first swing-down frames.
        self.fps = 100
        self.frame_dt = 1.0 / self.fps
        self.sim_substeps = 1
        self.sim_dt = self.frame_dt / self.sim_substeps
        self.sim_time = 0.0

        self.viewer = viewer
        self.args = args

        # ---- world ------------------------------------------------------
        # Scalar ``gravity=-9.81`` → ``model.gravity`` along −Z for Z-up.
        builder = newton.ModelBuilder(up_axis=newton.Axis.Z, gravity=-9.81)

        verts_m, tris, edge_med = _load_centered_tshirt_m()
        self._edge_med = edge_med
        # Place the (now stationary) garment at a comfortable viewing
        # height -- this matches the camera framing the gravity-on
        # variant of this scene uses.
        hang_height = 1.5

        builder.add_cloth_mesh(
            pos=wp.vec3(0.0, 0.0, hang_height),
            rot=wp.quat_identity(),
            scale=1.0,
            vel=wp.vec3(0.0, 0.0, 0.0),
            vertices=[wp.vec3(float(v[0]), float(v[1]), float(v[2])) for v in verts_m],
            indices=tris.flatten().tolist(),
            density=0.0,           # chysx redistributes lumped mass below
            tri_ke=0.0,
            tri_ka=0.0,
            tri_kd=0.0,
            edge_ke=0.0,
            edge_kd=0.0,
            particle_radius=0.4 * edge_med,
        )

        self.model = builder.finalize()

        q0 = self.model.particle_q.numpy().reshape(-1, 3)
        self._pin_indices = _pin_indices_neck_collar(q0, edge_med)
        self._pin_targets = q0[self._pin_indices].astype(np.float32).copy()

        # ---- solver -----------------------------------------------------
        # Proximity k = 1e3; untangle k = 2e3 (see file header).  Thickness 1 mm.
        self_collision_thickness = 1.0e-3  # [m]
        self_collision_stiffness = 1.0e3
        untangle_stiffness = 2.0 * self_collision_stiffness

        self.solver = newton.solvers.SolverChysX(
            self.model,
            gravity=(0.0, 0.0, -9.81),
            damping=0.05,
            fem_stretch_stiffness=5.0e2,
            fem_shear_stiffness=5.0e2,
            bending_stiffness=5.0e-4,
            pcg_iterations=50,
            surface_density=0.3,
            pin_indices=self._pin_indices.tolist(),
            pin_stiffness=1.0e9,
            self_collision_enabled=True,
            self_collision_thickness=self_collision_thickness,
            self_collision_stiffness=self_collision_stiffness,
            self_collision_max_contacts_factor=32,
            self_collision_max_ef_candidates_factor=128,
            static_contact_enabled=False,
            untangle_enabled=True,
            untangle_thickness=self_collision_thickness,
            untangle_stiffness=untangle_stiffness,
            untangle_max_contacts_factor=32,
        )

        self.solver.update_pin_targets(self._pin_targets)

        self._initial_z_extent = float(q0[:, 2].max() - q0[:, 2].min())
        print(
            f"[chysx_tshirt_hanging] bending dihedrals: "
            f"{self.solver._sim.num_bending_dihedrals()};  "
            f"pinned={len(self._pin_indices)} verts;  "
            f"initial panel sep (z extent) = {self._initial_z_extent:.3f} m;  "
            f"self_k={self_collision_stiffness:g}, "
            f"untangle_k={untangle_stiffness:g}, "
            f"thickness={self_collision_thickness:g} m"
        )

        self.state_0 = self.model.state()
        self.state_1 = self.model.state()
        self.control = self.model.control()
        self.contacts = self.model.contacts()

        self._cloth_tris = tris.copy()

        self.viewer.set_model(self.model)
        bmin = q0.min(axis=0).astype(np.float64)
        bmax = q0.max(axis=0).astype(np.float64)
        frame_z_up_camera_viewer(self.viewer, bmin, bmax)

        # ---- OBJ export (optional) -------------------------------------
        obj_out = getattr(args, "obj_out", None)
        self._obj_dir: str | None = None
        self._obj_stride = max(1, int(getattr(args, "obj_stride", 1)))
        self._frame_idx = 0
        if obj_out:
            self._obj_dir = os.path.abspath(str(obj_out))
            os.makedirs(self._obj_dir, exist_ok=True)
            print(
                f"[chysx_tshirt_hanging] OBJ export enabled -> {self._obj_dir}  "
                f"(stride {self._obj_stride})"
            )
            self._export_obj_frame(0)

        # ---- CUDA Graph capture ----------------------------------------
        self._cuda_graph = None
        self._capture_graph()

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
        # Warm-up so any lazy buffer alloc / topology rebuild lands
        # *outside* the captured region.
        self._simulate_substeps()
        wp.synchronize_device(device)

        # Print the contact count produced by the *first* substep --
        # this tells us how many self-contacts the rest pose itself
        # generates, before the garment has had a chance to deform.
        try:
            n_initial = int(self.solver._sim.self_collision_count())
            n_utg = int(self.solver._sim.untangle_count())
            print(
                f"[chysx_tshirt_hanging] after warm-up: "
                f"self_collision={n_initial}  untangle={n_utg}"
            )
        except Exception:
            pass

        with wp.ScopedCapture() as capture:
            self._simulate_substeps()
        self._cuda_graph = capture.graph

    def step(self):
        if self._cuda_graph is not None:
            wp.capture_launch(self._cuda_graph)
        else:
            self._simulate_substeps()
        self.sim_time += self.sim_substeps * self.sim_dt
        self._frame_idx += 1
        if self._obj_dir is not None and (self._frame_idx % self._obj_stride) == 0:
            self._export_obj_frame(self._frame_idx)

    def _export_obj_frame(self, frame_idx: int) -> None:
        """Write one Wavefront OBJ for the current cloth state."""

        assert self._obj_dir is not None
        q = self.state_0.particle_q.numpy().reshape(-1, 3)
        path = os.path.join(self._obj_dir, f"tshirt_hanging_{frame_idx:05d}.obj")
        with open(path, "w") as f:
            f.write(
                f"# chysx_tshirt_hanging frame {frame_idx}, "
                f"t = {self.sim_time:.6f} s\n"
            )
            np.savetxt(f, q, fmt="v %.6f %.6f %.6f")
            np.savetxt(f, self._cloth_tris + 1, fmt="f %d %d %d")

    def render(self):
        self.viewer.begin_frame(self.sim_time)
        self.viewer.log_state(self.state_0)
        self.viewer.end_frame()

    # ---- regression check --------------------------------------------

    def test_final(self):
        """Sanity-check the post-roll state under gravity + contacts."""

        q = self.state_0.particle_q.numpy().reshape(-1, 3)
        qd = self.state_0.particle_qd.numpy().reshape(-1, 3)

        if not (np.isfinite(q).all() and np.isfinite(qd).all()):
            raise ValueError("non-finite values in particle state")

        com = q.mean(axis=0)
        speed = np.linalg.norm(qd, axis=1)
        z_extent_final = float(q[:, 2].max() - q[:, 2].min())
        try:
            n_contacts = int(self.solver._sim.self_collision_count())
            n_utg = int(self.solver._sim.untangle_count())
        except Exception:
            n_contacts = -1
            n_utg = -1
        print(
            f"[chysx_tshirt_hanging] final: "
            f"CoM=({com[0]:+.4f},{com[1]:+.4f},{com[2]:+.4f}) m,  "
            f"|v| max={float(speed.max()):.4f} mean={float(speed.mean()):.4f} m/s,  "
            f"z extent {self._initial_z_extent:.4f} -> {z_extent_final:.4f} m,  "
            f"self_collision={n_contacts}  untangle={n_utg}"
        )

        bound = 10.0
        if (np.abs(q) > bound).any():
            raise ValueError(
                f"T-shirt particles escaped the {bound:.1f} m bbox; "
                f"max |q| = {float(np.abs(q).max()):.3f}"
            )

        max_speed = float(np.linalg.norm(qd, axis=1).max())
        if max_speed > 50.0:
            raise ValueError(
                f"particle speed exploded: max |v| = {max_speed:.3f} m/s"
            )

        pinned = self._pin_indices
        err = np.linalg.norm(q[pinned] - self._pin_targets, axis=1).max()
        if err > 5.0e-3:
            raise ValueError(
                f"pinned vertices drifted from targets: max |Δq| = {float(err):.4f} m"
            )


if __name__ == "__main__":
    parser = newton.examples.create_parser()
    parser.set_defaults(num_frames=600)  # 10 s at 60 fps -- enough to settle
    parser.add_argument(
        "--obj-out",
        type=str,
        default=None,
        metavar="DIR",
        help=(
            "If set, dump one Wavefront OBJ per frame into DIR "
            "(created if missing).  Names: tshirt_hanging_<frame:05d>.obj."
        ),
    )
    parser.add_argument(
        "--obj-stride",
        type=int,
        default=1,
        metavar="N",
        help="With --obj-out, write only every N-th frame.",
    )

    viewer, args = newton.examples.init(parser)
    newton.examples.run(Example(viewer, args), args)
