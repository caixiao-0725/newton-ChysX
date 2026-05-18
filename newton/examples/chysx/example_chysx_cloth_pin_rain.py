# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

###########################################################################
# Example ChysX Cloth Pin Rain
#
# A large square cloth is suspended in mid-air with **four corners pinned**
# (cloth-in-a-frame).  ``N`` smaller square patches are released from
# staggered heights and land on the taut / draped sheet, exercising chysx
# self-contact across disjoint mesh islands inside one merged triangle
# soup — same trick as ``example_chysx_cloth_stack``.  The only static
# shape is the ground plane; nothing supports the base sheet other than
# its four pinned corners.
#
# Adjust ``N`` at the command line::
#
#     python -m newton.examples chysx_cloth_pin_rain --num-small 8
#
###########################################################################

from __future__ import annotations

import argparse
import os

import numpy as np
import warp as wp

import newton
import newton.examples
from newton.examples.chysx._camera import frame_z_up_camera_viewer


def _make_square_patch(size: float, n: int) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(vertices, triangles)`` for an ``n×n`` grid square in the xy plane."""

    xs = np.linspace(-0.5 * size, 0.5 * size, n, dtype=np.float32)
    ys = np.linspace(-0.5 * size, 0.5 * size, n, dtype=np.float32)
    grid_x, grid_y = np.meshgrid(xs, ys, indexing="ij")
    verts = np.stack(
        [grid_x.flatten(), grid_y.flatten(), np.zeros(n * n, dtype=np.float32)],
        axis=1,
    ).astype(np.float32)

    tris = np.empty((2 * (n - 1) * (n - 1), 3), dtype=np.int32)
    t = 0
    for i in range(n - 1):
        for j in range(n - 1):
            v00 = i * n + j
            v10 = (i + 1) * n + j
            v01 = i * n + (j + 1)
            v11 = (i + 1) * n + (j + 1)
            tris[t] = (v00, v10, v11)
            tris[t + 1] = (v00, v11, v01)
            t += 2
    return verts, tris


def _corner_pin_indices(n: int) -> list[int]:
    """Four corners of the square grid (same indexing as ``_make_square_patch``)."""

    return [0, n - 1, (n - 1) * n, n * n - 1]


def _median_edge_length(verts: np.ndarray, tris: np.ndarray) -> float:
    lens: list[float] = []
    for tri in tris:
        a, b, c = int(tri[0]), int(tri[1]), int(tri[2])
        lens.append(float(np.linalg.norm(verts[a] - verts[b])))
        lens.append(float(np.linalg.norm(verts[b] - verts[c])))
        lens.append(float(np.linalg.norm(verts[c] - verts[a])))
    return float(np.median(np.asarray(lens, dtype=np.float64)))


def _build_merged_scene(
    *,
    base_size: float,
    base_n: int,
    small_size: float,
    small_n: int,
    num_small: int,
    base_z: float,
    drop_z0: float,
    drop_dz: float,
    xy_jitter: float,
    spread: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, list[int]]:
    """Merge base cloth + ``num_small`` falling patches into one mesh.

    Returns:
        ``(vertices, triangles, pin_indices)`` where ``pin_indices`` are
        global particle indices for the four base corners.
    """

    rng = np.random.default_rng(seed)
    base_v, base_t = _make_square_patch(base_size, base_n)
    base_v = base_v.copy()
    base_v[:, 2] = base_z

    parts_v: list[np.ndarray] = [base_v]
    parts_t: list[np.ndarray] = [base_t]
    v_off = base_v.shape[0]

    sb_v0, sb_t = _make_square_patch(small_size, small_n)
    n_sv = sb_v0.shape[0]

    for i in range(num_small):
        theta = float(rng.uniform(-np.pi, np.pi))
        c, s = np.cos(theta), np.sin(theta)
        dx = float(rng.uniform(-xy_jitter, xy_jitter))
        dy = float(rng.uniform(-xy_jitter, xy_jitter))
        gx = float(rng.uniform(-spread, spread))
        gy = float(rng.uniform(-spread, spread))

        v = sb_v0.copy()
        x, y = v[:, 0].copy(), v[:, 1].copy()
        v[:, 0] = c * x - s * y + gx + dx
        v[:, 1] = s * x + c * y + gy + dy
        v[:, 2] = drop_z0 + i * drop_dz

        parts_v.append(v)
        parts_t.append(sb_t + v_off)
        v_off += n_sv

    verts = np.concatenate(parts_v, axis=0).astype(np.float32)
    tris = np.concatenate(parts_t, axis=0).astype(np.int32)
    pins = _corner_pin_indices(base_n)
    return verts, tris, pins


class Example:
    @staticmethod
    def create_parser() -> argparse.ArgumentParser:
        p = newton.examples.create_parser()
        p.add_argument(
            "--num-small",
            type=int,
            default=5,
            metavar="N",
            help="Number of small cloth squares dropped onto the pinned base sheet.",
        )
        p.add_argument(
            "--base-n",
            type=int,
            default=41,
            metavar="N",
            help=(
                "Vertices per side on the pinned base sheet "
                "(NxN grid; default 41 -> ~18 mm cells)."
            ),
        )
        p.add_argument(
            "--small-n",
            type=int,
            default=17,
            metavar="N",
            help=(
                "Vertices per side on each falling small patch "
                "(NxN grid; default 17 -> ~9 mm cells)."
            ),
        )
        p.add_argument(
            "--seed",
            type=int,
            default=42,
            help="RNG seed for small-patch placements.",
        )
        p.add_argument(
            "--obj-out",
            type=str,
            default=None,
            metavar="DIR",
            help="Dump one OBJ per frame into DIR (use with --viewer null).",
        )
        p.add_argument(
            "--obj-stride",
            type=int,
            default=1,
            metavar="K",
            help="With --obj-out, write every K-th frame.",
        )
        return p

    def __init__(self, viewer, args):
        self.fps = 100
        self.frame_dt = 1.0 / self.fps
        self.sim_substeps = 1
        self.sim_dt = self.frame_dt / self.sim_substeps
        self.sim_time = 0.0

        self.viewer = viewer
        self.args = args

        num_small = max(0, int(getattr(args, "num_small", 5)))

        builder = newton.ModelBuilder(up_axis=newton.Axis.Z, gravity=-9.81)

        # Only static shape is the ground -- the base cloth hangs from
        # its four pinned corners with no tabletop underneath.
        builder.add_ground_plane()

        # Resolution chosen to land near multi_cloth's ~20 mm cell size
        # (50x50 over 1 m).  Defaults can be overridden with --base-n /
        # --small-n at the CLI for quick stress tests.
        self._base_size = 0.72
        self._base_n = int(getattr(args, "base_n", 41))     # ~18 mm cells
        self._small_size = 0.14
        self._small_n = int(getattr(args, "small_n", 17))   # ~9 mm cells
        self._num_small = num_small
        self._base_z = 0.08  # rest height of the suspended sheet [m]

        # Small patches hug the base sheet but must start *outside* the
        # contact buffer (~ 2 * self_thickness ~ 9 mm with the params below).
        # Anything closer triggers a kick on frame 0 from every small-cloth
        # particle and excites the high-freq buzz that this example used to
        # show.  3 cm clearance + 1 cm per-layer stagger keeps every pair of
        # vertices at least one buffer width apart at rest.
        _clear = 0.015  # [m]
        _dz = 0.006  # [m] between consecutive falling sheets

        verts, tris, pin_idx = _build_merged_scene(
            base_size=self._base_size,
            base_n=self._base_n,
            small_size=self._small_size,
            small_n=self._small_n,
            num_small=num_small,
            base_z=self._base_z,
            drop_z0=self._base_z + _clear,
            drop_dz=_dz,
            xy_jitter=0.01,
            spread=0.22 * self._base_size,
            seed=int(getattr(args, "seed", 42)),
        )

        edge_med = _median_edge_length(verts, tris)
        self._edge_med = edge_med

        self._cloth_particle_start = int(builder.particle_count)
        self._cloth_particle_count = int(verts.shape[0])
        self._cloth_tris = np.ascontiguousarray(tris, dtype=np.int32)

        builder.add_cloth_mesh(
            pos=wp.vec3(0.0, 0.0, 0.0),
            rot=wp.quat_identity(),
            scale=1.0,
            vel=wp.vec3(0.0, 0.0, 0.0),
            vertices=[wp.vec3(float(v[0]), float(v[1]), float(v[2])) for v in verts],
            indices=tris.flatten().tolist(),
            density=0.0,
            tri_ke=0.0,
            tri_ka=0.0,
            tri_kd=0.0,
            edge_ke=0.0,
            edge_kd=0.0,
            particle_radius=0.35 * edge_med,
        )

        self.model = builder.finalize()

        self_thickness = 0.2 * edge_med
        static_thickness = 0.5 * edge_med

        # Buffer caps scale with the load; cap so a single dense impact
        # cluster at the centre of the base sheet still has slack.  Higher
        # resolution multiplies the contact density, so let the broad-phase
        # cap grow a little faster than the narrow-phase one.
        cap_mult = 8 + 2 * num_small
        sc_narrow_factor = max(48, min(192, cap_mult * 8))
        sc_broad_factor = max(128, sc_narrow_factor * 3)

        # Stiffness matched to multi_cloth (which is rock-solid):
        #   self-collision k = 1e2 (cuda-cloth MultiCloth uses 1e1; bumping
        #     to 1e2 helps the much-coarser 13x13 base resist 5 falling
        #     patches without going to the buzzy 1e3 we had before).
        #   untangle_k = 2 * sc_k (style3D EF/VF ratio).  Was 2e3, way too
        #     stiff -- caused frame-0 kicks even from tiny EF candidates.
        # Coulomb friction is now POSITIVE (the C++ side guards `μ > 0`,
        # so the previous -0.35 was silently disabling friction and
        # letting layers slide-rock on the base sheet).
        # MODIFIED: Changed to positive value to enable friction with
        # the new RHS force implementation.
        sc_k = 1.0e2
        untangle_k = 3.0 * sc_k
        static_friction = 0.15  # Enable friction for testing

        self.solver = newton.solvers.SolverChysX(
            self.model,
            damping=1.0,
            fem_stretch_stiffness=1.0e3,
            fem_shear_stiffness=1.0e3,
            bending_stiffness=5.0e-4,
            pin_indices=pin_idx,
            pin_stiffness=1.0e8,
            pcg_iterations=80,
            surface_density=0.2,
            self_collision_enabled=True,
            self_collision_thickness=self_thickness,
            self_collision_stiffness=sc_k,
            self_collision_max_contacts_factor=sc_narrow_factor,
            self_collision_max_ef_candidates_factor=sc_broad_factor,
            static_contact_enabled=True,
            static_contact_thickness=static_thickness,
            static_contact_stiffness=1.0e4,
            static_contact_friction=static_friction,
            untangle_enabled=True,
            untangle_thickness=self_thickness,
            untangle_stiffness=untangle_k,
            untangle_max_contacts_factor=sc_narrow_factor,
        )

        self.state_0 = self.model.state()
        self.state_1 = self.model.state()
        self.control = self.model.control()
        self.contacts = self.model.contacts()

        self._pin_indices = list(pin_idx)
        self._initial_q = self.state_0.particle_q.numpy().reshape(-1, 3).copy()

        self.viewer.set_model(self.model)
        q = self._initial_q
        bmin = q.min(axis=0).astype(np.float64)
        bmax = q.max(axis=0).astype(np.float64)
        bmin[2] = min(float(bmin[2]), 0.0)
        frame_z_up_camera_viewer(self.viewer, bmin, bmax)

        print(
            f"[chysx_cloth_pin_rain] base {self._base_size:.2f} m ({self._base_n}×{self._base_n}),  "
            f"{num_small} small patches {self._small_size:.2f} m ({self._small_n}×{self._small_n});  "
            f"{verts.shape[0]} verts / {tris.shape[0]} tris;  "
            f"pins={pin_idx};  edge_med={edge_med*1e3:.1f} mm"
        )

        obj_out = getattr(args, "obj_out", None)
        self._obj_dir: str | None = None
        self._obj_stride = max(1, int(getattr(args, "obj_stride", 1)))
        self._frame_idx = 0
        if obj_out:
            self._obj_dir = os.path.abspath(str(obj_out))
            os.makedirs(self._obj_dir, exist_ok=True)
            print(
                f"[chysx_cloth_pin_rain] OBJ export enabled -> {self._obj_dir}  "
                f"(stride {self._obj_stride})"
            )
            self._export_obj_frame(0)

        self._cuda_graph = None
        self._capture_graph()

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

    def step(self) -> None:
        if self._cuda_graph is not None:
            wp.capture_launch(self._cuda_graph)
        else:
            self._simulate_substeps()
        self.sim_time += self.sim_substeps * self.sim_dt
        self._frame_idx += 1
        if self._obj_dir is not None and (self._frame_idx % self._obj_stride) == 0:
            self._export_obj_frame(self._frame_idx)

    def render(self) -> None:
        self.viewer.begin_frame(self.sim_time)
        self.viewer.log_state(self.state_0)
        self.viewer.end_frame()

    def _export_obj_frame(self, frame_idx: int) -> None:
        assert self._obj_dir is not None
        q = self.state_0.particle_q.numpy().reshape(-1, 3)
        s = self._cloth_particle_start
        n = self._cloth_particle_count
        verts = q[s : s + n]

        path = os.path.join(self._obj_dir, f"cloth_pin_rain_{frame_idx:05d}.obj")
        with open(path, "w") as f:
            f.write(
                f"# chysx_cloth_pin_rain frame {frame_idx}, "
                f"t = {self.sim_time:.6f} s\n"
            )
            np.savetxt(f, verts, fmt="v %.6f %.6f %.6f")
            np.savetxt(f, self._cloth_tris + 1, fmt="f %d %d %d")

    def test_final(self) -> None:
        q = self.state_0.particle_q.numpy().reshape(-1, 3)
        qd = self.state_0.particle_qd.numpy().reshape(-1, 3)

        if not (np.isfinite(q).all() and np.isfinite(qd).all()):
            raise ValueError("non-finite values in particle state")

        bound = 6.0
        if (np.abs(q) > bound).any():
            raise ValueError(
                f"cloth escaped {bound:.1f} m bbox; max |q| = {float(np.abs(q).max()):.3f}"
            )

        slack = 1.5 * float(self.solver._sim.static_contact_thickness())
        z_min = float(q[:, 2].min())
        if z_min < -slack:
            raise ValueError(
                f"fell through ground: min z = {z_min:.4f} m "
                f"(slack {-slack:.4f} m)"
            )

        max_speed = float(np.linalg.norm(qd, axis=1).max())
        if max_speed > 30.0:
            raise ValueError(f"speed exploded: max |v| = {max_speed:.3f} m/s")

        q0 = self._initial_q
        for pi in self._pin_indices:
            d = float(np.linalg.norm(q[pi] - q0[pi]))
            if d > 0.01:
                raise ValueError(f"pin {pi} drifted {d:.3f} m from rest pose")

        print(
            f"[chysx_cloth_pin_rain] test_final: min z={z_min:.4f}, "
            f"max |v|={max_speed:.4f} m/s"
        )


if __name__ == "__main__":
    parser = Example.create_parser()
    parser.set_defaults(num_frames=700)
    viewer, args = newton.examples.init(parser)
    newton.examples.run(Example(viewer, args), args)
