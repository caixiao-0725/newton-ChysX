# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

###########################################################################
# Example ChysX Multi Cloth
#
# 1:1 replication of cuda-cloth's `MultiClothCase`
# (see d:/physics/cuda-cloth/src/Simulator/MultiCloth.cpp).
#
# Scene
# -----
# `assets/quad/5-layer.obj` is a Houdini-baked stack of 5 cloth layers
# sharing one merged triangle soup -- 7000 verts, 13212 triangles,
# bbox `[-0.5, 0, -0.5] -> [0.5, 0.05, 0.5]` (a 1m x 1m square cloth
# with five sheets stacked along the cuda-cloth Y axis, 1.25 cm apart).
#
# The first 2500 vertices form the bottom layer as a 50x50 regular
# grid; cuda-cloth pins its four corners (`m_pin = {0, 49, 2450, 2499}`)
# to act as a tablecloth-in-a-frame.  The remaining four sheets sit on
# top with no anchor and settle onto the pinned base under gravity +
# self-contact + untangle -- exactly the workload chysx's
# self-collision + untangle pipeline is built for.
#
# Material parameters (cuda-cloth `MultiClothCase::Initialize`)
# -------------------------------------------------------------
#
#     thickness          = 0.005   (cloth_buffer.m_thickness)
#     proximity:
#         m_4_k          = 10.0    -> self_collision_stiffness
#         m_4_thickness  = 0.007   -> self_collision_thickness
#     untangle:
#         m_5_k          = 100.0   -> untangle_stiffness
#         m_5_thickness  = 0.005   -> untangle_thickness
#     pin_stiffness      = 1e9     (m_control_mag default)
#     stretch / bending / damping / gravity / dt come from
#     `ClothDataBuffer.h` defaults (k=1e3, bending_k=5e-4, damping=1.0,
#     gravity=-9.8, dt=0.01) since `MultiClothCase` doesn't override
#     them in the JSON loader.
#
# Loop bound
# ----------
# cuda-cloth runs `MultiClothCase::Run` for 200 steps at dt = 0.01 ->
# 2 s.  We expose `--num-frames 200` by default.
#
# Command:
#
#     python -m newton.examples chysx_multi_cloth
#     python -m newton.examples chysx_multi_cloth --no-untangle
#     python -m newton.examples chysx_multi_cloth --obj-out /tmp/multi
#
###########################################################################

from __future__ import annotations

import os

import numpy as np
import warp as wp

import newton
import newton.examples


# Pin indices that cuda-cloth sets in `MultiClothCase::Initialize`
# (corners of the 50x50 bottom layer baked into 5-layer.obj).
_PIN_INDICES = (0, 49, 2450, 2499)


def _resolve_obj(filename: str) -> str:
    """Locate a cuda-cloth mesh asset.

    Search order:
      1. ``newton/examples/assets/chysx/<filename>`` (shipped copy).
      2. ``d:/physics/cuda-cloth/assets/quad/<filename>`` (sibling
         cuda-cloth checkout that ships with the engine).
    """
    shipped = newton.examples.get_asset(f"chysx/{filename}")
    if os.path.isfile(shipped):
        return shipped
    fallback = rf"d:\physics\cuda-cloth\assets\quad\{filename}"
    if os.path.isfile(fallback):
        return fallback
    raise FileNotFoundError(
        f"{filename} not found at either {shipped!r} or {fallback!r}; "
        f"copy the file from cuda-cloth's assets/quad/ folder."
    )


def _load_obj(path: str) -> tuple[np.ndarray, np.ndarray]:
    """Parse a triangulated wavefront OBJ.

    Vertex order is preserved exactly as in the file so cuda-cloth's
    integer pin indices stay valid.  Quads are fan-triangulated; UV /
    normal indices are ignored.
    """
    verts: list[tuple[float, float, float]] = []
    tris: list[tuple[int, int, int]] = []
    with open(path) as f:
        for raw in f:
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            head, _, rest = line.partition(" ")
            if head == "v":
                xyz = rest.split()
                verts.append(
                    (float(xyz[0]), float(xyz[1]), float(xyz[2]))
                )
            elif head == "f":
                idx = [int(p.split("/")[0]) - 1 for p in rest.split()]
                if len(idx) == 3:
                    tris.append((idx[0], idx[1], idx[2]))
                else:
                    for k in range(1, len(idx) - 1):
                        tris.append((idx[0], idx[k], idx[k + 1]))
    return (
        np.asarray(verts, dtype=np.float32),
        np.asarray(tris, dtype=np.int32),
    )


def _y_up_to_z_up(verts_y_up: np.ndarray) -> np.ndarray:
    """Swap (x, y, z) -> (x, z, y) so Y-up cuda-cloth meshes line up
    with Newton's Z-up convention without changing vertex ordering."""

    out = np.empty_like(verts_y_up)
    out[:, 0] = verts_y_up[:, 0]
    out[:, 1] = verts_y_up[:, 2]
    out[:, 2] = verts_y_up[:, 1]
    return out


class Example:
    @staticmethod
    def create_parser():
        p = newton.examples.create_parser()
        p.add_argument(
            "--no-untangle",
            dest="untangle",
            action="store_false",
            help=(
                "Ablation: disable the 5-vertex untangle pass; the upper "
                "layers usually deadlock in proximity-only contact within "
                "a few seconds."
            ),
        )
        p.set_defaults(untangle=True)
        p.add_argument(
            "--obj-out",
            type=str,
            default=None,
            metavar="DIR",
            help="Dump one OBJ per simulated frame into DIR.",
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
        # cuda-cloth MultiClothCase uses dt = 0.01 s (ClothDataBuffer
        # default) with a single iteration per step; mirror that.
        self.fps = 100
        self.frame_dt = 1.0 / self.fps
        self.sim_substeps = 1
        self.sim_dt = self.frame_dt / self.sim_substeps
        self.sim_time = 0.0

        self.viewer = viewer
        self.args = args

        # ---- mesh -------------------------------------------------------
        verts_yu, tris = _load_obj(_resolve_obj("5-layer.obj"))
        if verts_yu.shape[0] != 7000:
            raise ValueError(
                f"5-layer.obj: expected 7000 verts, got {verts_yu.shape[0]} "
                "(asset out of date?)"
            )
        verts = _y_up_to_z_up(verts_yu).astype(np.float32)

        self._n_verts = int(verts.shape[0])
        self._n_tris = int(tris.shape[0])
        self._cloth_tris = np.ascontiguousarray(tris, dtype=np.int32)
        self._initial_q = verts.copy()

        # Approximate edge length from one face -- only used to pick the
        # particle radius the viewer renders.
        e0 = float(np.linalg.norm(verts[tris[0, 0]] - verts[tris[0, 1]]))
        self._edge_l = e0

        # ---- model ------------------------------------------------------
        builder = newton.ModelBuilder(up_axis=newton.Axis.Z, gravity=-9.8)
        self._cloth_particle_start = int(builder.particle_count)
        self._cloth_particle_count = self._n_verts
        builder.add_cloth_mesh(
            pos=wp.vec3(0.0, 0.0, 0.0),
            rot=wp.quat_identity(),
            scale=1.0,
            vel=wp.vec3(0.0, 0.0, 0.0),
            vertices=[wp.vec3(float(v[0]), float(v[1]), float(v[2])) for v in verts],
            indices=tris.flatten().tolist(),
            density=0.0,             # ChysX recomputes lumped mass below.
            tri_ke=0.0,
            tri_ka=0.0,
            tri_kd=0.0,
            edge_ke=0.0,
            edge_kd=0.0,
            particle_radius=0.4 * e0,
        )
        self.model = builder.finalize()

        # ---- pins -------------------------------------------------------
        self._pin_indices = np.asarray(_PIN_INDICES, dtype=np.int32)
        self._pin_targets = verts[self._pin_indices].astype(np.float32).copy()

        # ---- collision parameters (cuda-cloth MultiClothCase) ----------
        # m_4_k = 10, m_4_thickness = 0.007 (proximity self-collision)
        # m_5_k = 100, m_5_thickness = 0.005 (untangle / EF)
        sc_thickness = 0.007
        sc_stiffness = 1.0e1
        utg_thickness = 0.005
        utg_stiffness = 1.0e2

        # 5 layers x 50x50 + extras puts the worst-case proximity contact
        # density well above the chysx default of 8x particle_count;
        # mirror the untangle example's generous sizing.
        self_collision_max_contacts_factor = 64
        self_collision_max_ef_candidates_factor = 256
        untangle_max_contacts_factor = 64

        # ---- solver -----------------------------------------------------
        # cuda-cloth scatters `0.1 * area` per vertex (NOT divided by 3)
        # in `KernelComputeAllDm`; ChysX's `redistribute_mass_area_weighted`
        # divides by 3, so 0.3 reproduces the same lumped masses.
        surface_density = 0.3

        self.solver = newton.solvers.SolverChysX(
            self.model,
            damping=1.0,                    # ClothDataBuffer m_damping = 1.0
            fem_stretch_stiffness=1.0e3,    # ClothDataBuffer m_k = 1e3
            fem_shear_stiffness=1.0e3,
            bending_stiffness=5.0e-4,       # ClothDataBuffer m_bending_k = 5e-4
            pin_indices=self._pin_indices.tolist(),
            pin_stiffness=1.0e9,            # m_control_mag = 1e9
            pcg_iterations=50,
            surface_density=surface_density,
            self_collision_enabled=True,
            self_collision_thickness=sc_thickness,
            self_collision_stiffness=sc_stiffness,
            self_collision_max_contacts_factor=self_collision_max_contacts_factor,
            self_collision_max_ef_candidates_factor=self_collision_max_ef_candidates_factor,
            untangle_enabled=bool(args.untangle),
            untangle_thickness=utg_thickness,
            untangle_stiffness=utg_stiffness,
            untangle_max_contacts_factor=untangle_max_contacts_factor,
        )
        self._untangle_enabled = bool(args.untangle)

        self.state_0 = self.model.state()
        self.state_1 = self.model.state()
        self.control = self.model.control()
        self.contacts = self.model.contacts()

        # Pin targets are static throughout the run.
        self.solver.update_pin_targets(self._pin_targets)

        # ---- viewer -----------------------------------------------------
        self.viewer.set_model(self.model)
        bbox_min = verts.min(axis=0)
        bbox_max = verts.max(axis=0)
        centre = 0.5 * (bbox_min + bbox_max)
        extent = float(np.linalg.norm(bbox_max - bbox_min))
        dist = max(1.4 * extent, 1.5)
        eye = (
            float(centre[0] + 0.30 * dist),
            float(centre[1] - 1.05 * dist),
            float(centre[2] + 0.65 * dist),
        )
        cam = getattr(self.viewer, "camera", None)
        if cam is not None:
            cam.pos = cam._as_vec3(eye)
            cam.look_at(cam._as_vec3((float(centre[0]), float(centre[1]), float(centre[2]))))
        else:
            self.viewer.set_camera(
                wp.vec3(eye[0], eye[1], eye[2]),
                pitch=-25.0,
                yaw=35.0,
            )

        print(
            f"[chysx_multi_cloth] 5-layer.obj: {self._n_verts} verts / "
            f"{self._n_tris} tris;  pins={list(self._pin_indices)};  "
            f"sc(k={sc_stiffness:g}, h={sc_thickness:g}),  "
            f"untangle({'on' if self._untangle_enabled else 'off'}: "
            f"k={utg_stiffness:g}, h={utg_thickness:g})"
        )

        # ---- OBJ export -------------------------------------------------
        obj_out = getattr(args, "obj_out", None)
        self._obj_dir: str | None = None
        self._obj_stride = max(1, int(getattr(args, "obj_stride", 1)))
        self._frame_idx = 0
        if obj_out:
            self._obj_dir = os.path.abspath(str(obj_out))
            os.makedirs(self._obj_dir, exist_ok=True)
            print(
                f"[chysx_multi_cloth] OBJ export -> {self._obj_dir}  "
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
        # Warm-up so any lazy alloc / topology rebuild lands outside the
        # captured region.
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

    # ---- OBJ export -----------------------------------------------------

    def _export_obj_frame(self, frame_idx: int) -> None:
        assert self._obj_dir is not None
        q = self.state_0.particle_q.numpy().reshape(-1, 3)
        path = os.path.join(self._obj_dir, f"multi_cloth_{frame_idx:05d}.obj")
        with open(path, "w") as f:
            f.write(
                f"# chysx_multi_cloth frame {frame_idx}, t = {self.sim_time:.6f} s\n"
            )
            np.savetxt(f, q, fmt="v %.6f %.6f %.6f")
            np.savetxt(f, self._cloth_tris + 1, fmt="f %d %d %d")

    # ---- regression check ----------------------------------------------

    def test_final(self) -> None:
        q = self.state_0.particle_q.numpy().reshape(-1, 3)
        qd = self.state_0.particle_qd.numpy().reshape(-1, 3)

        if not (np.isfinite(q).all() and np.isfinite(qd).all()):
            raise ValueError("non-finite values in particle state")

        bound = 5.0
        if (np.abs(q) > bound).any():
            raise ValueError(
                f"cloth escaped {bound:.1f} m bbox; "
                f"max |q| = {float(np.abs(q).max()):.3f}"
            )

        max_speed = float(np.linalg.norm(qd, axis=1).max())
        if max_speed > 50.0:
            raise ValueError(f"speed exploded: max |v| = {max_speed:.3f}")

        # Pins must not have drifted.
        for pi, target in zip(self._pin_indices.tolist(), self._pin_targets):
            d = float(np.linalg.norm(q[int(pi)] - target))
            if d > 0.01:
                raise ValueError(
                    f"pin {int(pi)} drifted {d:.3f} m from rest pose"
                )

        n_self = self.solver._sim.self_collision_count(0)
        n_tan = (
            self.solver._sim.untangle_count(0) if self._untangle_enabled else 0
        )
        z_min = float(q[:, 2].min())
        z_max = float(q[:, 2].max())
        print(
            f"[chysx_multi_cloth] final-frame contacts: "
            f"self_collision={n_self}  untangle={n_tan};  "
            f"z range = [{z_min:.4f}, {z_max:.4f}];  "
            f"max |v| = {max_speed:.4f} m/s"
        )


if __name__ == "__main__":
    parser = Example.create_parser()
    parser.set_defaults(num_frames=200)  # cuda-cloth MultiClothCase loop
    viewer, args = newton.examples.init(parser)
    newton.examples.run(Example(viewer, args), args)
