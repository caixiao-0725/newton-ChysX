// SPDX-License-Identifier: Apache-2.0
//
// pybind11 bindings for the _chysx_native module.

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

#include <stdexcept>

#include "cloth/cloth_material.h"
#include "cloth/cloth_simulator.h"
#include "collision/static_contact.h"
#include "coupled/coupled_simulator.h"
#include "math/vec.cuh"
#include "rigid/newton_avbd/rigid_simulator.h"
#include "rigid/featherstone/featherstone_solver.h"
#include "ik/ik_solver.h"

namespace py = pybind11;

PYBIND11_MODULE(_chysx_native, m) {
    m.doc() = "ChysX: minimal CUDA cloth physics simulator";

    // ---- ClothMaterial ----------------------------------------------------

    py::class_<chysx::cloth::ClothMaterial>(m, "ClothMaterial", R"pbdoc(
Plain-old-data material parameters for the cloth simulator.

Each field is a Python-mutable scalar; copy a fully populated instance
into a ClothSimulator with ``simulator.set_material(material)``.
)pbdoc")
        .def(py::init<>())
        .def_readwrite("lame_mu", &chysx::cloth::ClothMaterial::lame_mu,
                       "Lamé mu [Pa] (in-plane elasticity, unused by free-fall).")
        .def_readwrite("lame_lambda", &chysx::cloth::ClothMaterial::lame_lambda,
                       "Lamé lambda [Pa] (in-plane elasticity, unused by free-fall).")
        .def_readwrite("bending", &chysx::cloth::ClothMaterial::bending,
                       "Dihedral bending stiffness [N·m] (unused by free-fall).")
        .def_readwrite("density", &chysx::cloth::ClothMaterial::density,
                       "Surface mass density [kg/m^2].")
        .def_readwrite("damping", &chysx::cloth::ClothMaterial::damping,
                       "Velocity damping [1/s] (v *= exp(-damping * dt)).")
        .def_readwrite("gx", &chysx::cloth::ClothMaterial::gx,
                       "Gravity x-component [m/s^2].")
        .def_readwrite("gy", &chysx::cloth::ClothMaterial::gy,
                       "Gravity y-component [m/s^2].")
        .def_readwrite("gz", &chysx::cloth::ClothMaterial::gz,
                       "Gravity z-component [m/s^2].");

    // ---- ClothSimulator ---------------------------------------------------

    py::class_<chysx::cloth::ClothSimulator>(m, "ClothSimulator", R"pbdoc(
Cloth physics simulator.

Owns a copy of the material parameters and a set of buffer handles
(externally-owned device pointers + ChysX-owned working arrays).
Callers push parameters in once via ``set_material``, then push device
pointers in each step via ``set_external_buffers`` before calling
``step(dt)``.
)pbdoc")
        .def(py::init<>())
        .def("set_material", &chysx::cloth::ClothSimulator::set_material,
             py::arg("material"),
             "Copy `material` into the simulator.")
        .def("set_external_buffers",
             &chysx::cloth::ClothSimulator::set_external_buffers,
             py::arg("pos_ptr"),
             py::arg("vel_ptr"),
             py::arg("particle_count"),
             py::arg("inv_mass_ptr") = 0,
             py::arg("force_ptr") = 0,
             R"pbdoc(
Stash externally-owned CUDA device pointers (cast to int) for the next
step().  ChysX never copies or frees these; the caller must keep them
alive until step() returns.
)pbdoc")
        .def("step", &chysx::cloth::ClothSimulator::step,
             py::arg("dt"),
             py::arg("cuda_stream") = 0,
             R"pbdoc(
Advance the simulation by `dt` seconds using the currently set material
and external buffers.  Throws if pos_ptr / vel_ptr were not set.
)pbdoc")
        .def(
            "set_pins",
            [](chysx::cloth::ClothSimulator& self,
               py::array_t<int, py::array::c_style | py::array::forcecast> indices,
               py::array_t<float, py::array::c_style | py::array::forcecast> targets,
               float stiffness) {
                if (indices.ndim() != 1) {
                    throw std::invalid_argument(
                        "ClothSimulator.set_pins: indices must be 1-D");
                }
                if (targets.ndim() != 2 || targets.shape(1) != 3) {
                    throw std::invalid_argument(
                        "ClothSimulator.set_pins: targets must have shape (N, 3)");
                }
                if (indices.shape(0) != targets.shape(0)) {
                    throw std::invalid_argument(
                        "ClothSimulator.set_pins: indices and targets must "
                        "have the same length");
                }
                const int n = static_cast<int>(indices.shape(0));
                // Vec3f and float[3] share the same 12-byte layout, so we
                // can hand the contiguous numpy buffer to set_pins via a
                // reinterpret_cast.
                const auto* targets_vec3 =
                    reinterpret_cast<const chysx::math::Vec3f*>(targets.data());
                self.set_pins(indices.data(), targets_vec3, n, stiffness);
            },
            py::arg("indices"),
            py::arg("targets"),
            py::arg("stiffness") = 1.0e6f,
            R"pbdoc(
Install pin constraints.

Parameters
----------
indices : numpy.ndarray, shape (N,), dtype int32
    Global particle index of each pin.
targets : numpy.ndarray, shape (N, 3), dtype float32
    World-space target position for each pin.
stiffness : float
    Penalty stiffness used by the future PCG step.  The current
    free-fall integrator hard-clamps pinned particles instead, so
    this value is stored but not consulted yet.
)pbdoc")
        .def("clear_pins", &chysx::cloth::ClothSimulator::clear_pins,
             "Remove every previously installed pin.")
        .def(
            "num_pins",
            [](const chysx::cloth::ClothSimulator& self) {
                return self.pins().size();
            },
            "Number of currently installed pins.")
        .def(
            "update_pin_targets",
            [](chysx::cloth::ClothSimulator& self,
               py::array_t<float, py::array::c_style | py::array::forcecast> targets,
               std::uintptr_t cuda_stream) {
                if (targets.ndim() != 2 || targets.shape(1) != 3) {
                    throw std::invalid_argument(
                        "ClothSimulator.update_pin_targets: targets must "
                        "have shape (N, 3) and dtype float32");
                }
                const int n = static_cast<int>(targets.shape(0));
                const auto* targets_vec3 =
                    reinterpret_cast<const chysx::math::Vec3f*>(targets.data());
                self.update_pin_targets(targets_vec3, n, cuda_stream);
            },
            py::arg("targets"),
            py::arg("cuda_stream") = 0,
            R"pbdoc(
Update the world-space target positions of the currently installed
pins without changing their indices.  Use this for animations where
pins move every frame (e.g. twisting a cloth around a moving boundary)
to avoid the Hessian-topology rebuild that ``set_pins(...)`` triggers.

Parameters
----------
targets : numpy.ndarray, shape (n_pins, 3), dtype float32
    New target positions; ``n_pins`` must equal ``num_pins()``.
cuda_stream : int, optional
    Stream to issue the host-to-device copy on.
)pbdoc")
        .def(
            "set_mesh",
            [](chysx::cloth::ClothSimulator& self,
               py::array_t<int, py::array::c_style | py::array::forcecast> tris) {
                if (tris.ndim() != 2 || tris.shape(1) != 3) {
                    throw std::invalid_argument(
                        "ClothSimulator.set_mesh: triangles must have shape "
                        "(M, 3) and dtype int32");
                }
                const int n = static_cast<int>(tris.shape(0));
                // Vec3i and int[3] share layout (12 bytes, native int).
                const auto* tris_vec3i =
                    reinterpret_cast<const chysx::math::Vec3i*>(tris.data());
                self.set_mesh(tris_vec3i, n);
            },
            py::arg("triangles"),
            R"pbdoc(
Upload the cloth's triangle topology into ChysX-owned device memory and
extract the unique edge list on the host.  Call this once at setup time;
edges are then available to ``build_springs_from_current_positions``.

Parameters
----------
triangles : numpy.ndarray, shape (M, 3), dtype int32
    Triangle vertex indices.
)pbdoc")
        .def("build_springs_from_current_positions",
             &chysx::cloth::ClothSimulator::build_springs_from_current_positions,
             py::arg("stiffness"),
             py::arg("cuda_stream") = 0,
             R"pbdoc(
Install one Hookean spring per unique mesh edge using the *current*
externally-bound positions as the rest configuration.  Requires
``set_mesh`` and ``set_external_buffers`` to have been called first.

Parameters
----------
stiffness : float
    Per-spring stiffness k [N/m] (shared by every spring).
)pbdoc")
        .def(
            "num_springs",
            [](const chysx::cloth::ClothSimulator& self) {
                return self.springs().size();
            },
            "Number of currently installed springs (unique mesh edges).")
        .def("redistribute_mass_area_weighted",
             &chysx::cloth::ClothSimulator::redistribute_mass_area_weighted,
             py::arg("surface_density"),
             py::arg("inv_mass_ptr"),
             py::arg("particle_count"),
             py::arg("cuda_stream") = 0,
             R"pbdoc(
Recompute per-particle inverse mass by distributing each triangle's
``surface_density * area`` equally across its three vertices, matching
cuda-cloth's lumped finite-element mass model.

Boundary vertices end up lighter than interior vertices (the
physically correct behaviour) so dense meshes drape naturally instead
of pulling a heavy uniform-mass corner down.

Parameters
----------
surface_density : float
    Material surface density in kg/m^2 (e.g. 0.3 for cotton).
inv_mass_ptr : int
    cudaMalloc'd address of the externally-owned inverse-mass buffer
    (typically Newton's ``model.particle_inv_mass.ptr``); the routine
    overwrites it.  Vertices with no incident triangle are written as
    ``inv_mass = 0`` (treated as kinematic).
particle_count : int
    Number of particles in the inverse-mass buffer.

Requires ``set_mesh(...)`` and ``set_external_buffers(...)`` to have
been called first so ChysX has access to the triangle topology and
the rest positions used for area computation.
)pbdoc")
        .def("build_fem_stretch_from_current_positions",
             &chysx::cloth::ClothSimulator::build_fem_stretch_from_current_positions,
             py::arg("stiffness"),
             py::arg("cuda_stream") = 0,
             R"pbdoc(
Install one Baraff-Witkin triangle stretch element per face of the
current mesh.  The reference shape (Dm_inv, rest area) is computed
from the *current* externally-bound positions, so call this once
after ``set_mesh`` and ``set_external_buffers``.

Parameters
----------
stiffness : float
    Per-area stretch stiffness ``k`` [N/m^2]; the per-element weight is
    ``area * k`` (cuda-cloth's ``k_stretch`` convention).
)pbdoc")
        .def(
            "num_fem_stretch_triangles",
            [](const chysx::cloth::ClothSimulator& self) {
                return self.fem_stretch().size();
            },
            "Number of currently installed FEM stretch triangles.")
        .def("build_fem_shear_from_current_positions",
             &chysx::cloth::ClothSimulator::build_fem_shear_from_current_positions,
             py::arg("stiffness"),
             py::arg("cuda_stream") = 0,
             R"pbdoc(
Install one Baraff-Witkin triangle *shear* element per face of the
current mesh.  Internally this uses exactly the same kernels as the
stretch element, just with the material (u, v) axes rotated 45 degrees
so the constraint pins the diagonal lengths instead of the U/V edges
— equivalent to cuda-cloth's
``KernelComputeStretchShearForceAndHessianFast``.

Parameters
----------
stiffness : float
    Per-area shear stiffness ``k`` [N/m^2]; per-element weight is
    ``area * k`` (cuda-cloth's ``k_stretch`` convention; the same
    constant is reused for both stretch and shear in cuda-cloth).
)pbdoc")
        .def(
            "num_fem_shear_triangles",
            [](const chysx::cloth::ClothSimulator& self) {
                return self.fem_shear().size();
            },
            "Number of currently installed FEM shear triangles.")
        .def("build_bending_from_current_positions",
             &chysx::cloth::ClothSimulator::build_bending_from_current_positions,
             py::arg("stiffness"),
             py::arg("cuda_stream") = 0,
             R"pbdoc(
Auto-detect dihedrals from the currently installed mesh and install
one Baraff-Witkin / Bridson bending element per interior edge (every
edge shared by exactly two triangles).  Rest angles are computed from
the *current* externally-bound positions, so call this once after
``set_mesh`` and ``set_external_buffers``.

Equivalent to cuda-cloth's
``KernelComputeDihedralForcesAndHessianFast`` with rest angles
populated by ``KernelComputeDihedralAngle``.

Parameters
----------
stiffness : float
    Bending stiffness ``k_bending`` shared by every dihedral.
    Cloth-like values are typically several orders of magnitude
    smaller than the in-plane stretch / shear stiffness.
)pbdoc")
        .def(
            "num_bending_dihedrals",
            [](const chysx::cloth::ClothSimulator& self) {
                return self.bending().size();
            },
            "Number of currently installed bending dihedrals.")
        // ---- solver type (PCG vs VBD) --------------------------------
        .def("set_solver_type",
             [](chysx::cloth::ClothSimulator& self, int t) {
                 self.set_solver_type(
                     static_cast<chysx::cloth::SolverType>(t));
             },
             py::arg("solver_type"),
             R"pbdoc(
Set the solver type: 0 = PCG (implicit Euler), 1 = VBD (Gauss-Seidel).
)pbdoc")
        .def("set_vbd_iterations",
             &chysx::cloth::ClothSimulator::set_vbd_iterations,
             py::arg("iterations"),
             R"pbdoc(
Set the number of VBD Gauss-Seidel iterations per substep.
)pbdoc")
        .def("build_vbd_coloring",
             &chysx::cloth::ClothSimulator::build_vbd_coloring,
             R"pbdoc(
Build graph coloring and vertex-tet adjacency for VBD.
Must be called after set_tet_mesh() and before stepping with VBD.
)pbdoc")
        .def(
            "num_vbd_colors",
            [](const chysx::cloth::ClothSimulator& self) {
                return self.vbd_solver().num_colors();
            },
            "Number of colors in the VBD graph coloring.")
        // ---- tetrahedral FEM (stable Neo-Hookean) --------------------
        .def(
            "set_tet_mesh",
            [](chysx::cloth::ClothSimulator& self,
               py::array_t<int, py::array::c_style | py::array::forcecast> tets,
               py::array_t<float, py::array::c_style | py::array::forcecast> materials,
               std::uintptr_t cuda_stream) {
                if (tets.ndim() != 2 || tets.shape(1) != 4) {
                    throw std::invalid_argument(
                        "set_tet_mesh: tets must have shape (T, 4) int32");
                }
                if (materials.ndim() != 2 || materials.shape(1) != 3) {
                    throw std::invalid_argument(
                        "set_tet_mesh: materials must have shape (T, 3) float32");
                }
                const int n = static_cast<int>(tets.shape(0));
                if (materials.shape(0) != n) {
                    throw std::invalid_argument(
                        "set_tet_mesh: tets and materials must have the same "
                        "number of rows");
                }
                auto tets_ptr = reinterpret_cast<const chysx::math::Vec4i*>(
                    tets.data());
                auto mats_ptr = reinterpret_cast<const chysx::math::Vec3f*>(
                    materials.data());
                self.set_tet_mesh(tets_ptr, mats_ptr, n, cuda_stream);
            },
            py::arg("tets"),
            py::arg("materials"),
            py::arg("cuda_stream") = 0,
            R"pbdoc(
Install tetrahedral FEM constraints with stable Neo-Hookean material.

Parameters
----------
tets : ndarray, shape (T, 4), dtype int32
    Vertex indices ``(v0, v1, v2, v3)`` per tet.
materials : ndarray, shape (T, 3), dtype float32
    ``(mu, lambda, k_damp)`` per tet [Pa, Pa, Pa·s].
)pbdoc")
        .def("redistribute_mass_volume_weighted",
             &chysx::cloth::ClothSimulator::redistribute_mass_volume_weighted,
             py::arg("density"),
             py::arg("inv_mass_ptr"),
             py::arg("particle_count"),
             py::arg("cuda_stream") = 0,
             R"pbdoc(
Redistribute particle masses using volume-weighted lumping from the
installed tet mesh.  Each tet contributes ``density * V_tet / 4`` to
each of its four vertices.

Requires ``set_tet_mesh(...)`` and ``set_external_buffers(...)`` to have
been called first.
)pbdoc")
        .def(
            "num_tets",
            [](const chysx::cloth::ClothSimulator& self) {
                return self.tet_fem().size();
            },
            "Number of currently installed tetrahedral FEM elements.")
        // ---- self-collision (DCD, brute-force VF for v1) ------------
        .def("set_self_collision_enabled",
             &chysx::cloth::ClothSimulator::set_self_collision_enabled,
             py::arg("enabled"),
             "Toggle the brute-force VF self-collision pipeline.")
        .def("self_collision_enabled",
             &chysx::cloth::ClothSimulator::self_collision_enabled,
             "True if self-collision is currently enabled.")
        .def("set_self_collision_thickness",
             &chysx::cloth::ClothSimulator::set_self_collision_thickness,
             py::arg("thickness"),
             R"pbdoc(
Set the contact distance threshold (in world units, same as particle
positions).  A vertex within ``thickness`` of any non-incident
triangle becomes a contact.  cuda-cloth's twist case uses
``thickness ~ 0.2 * average_edge_length``.
)pbdoc")
        .def("self_collision_thickness",
             &chysx::cloth::ClothSimulator::self_collision_thickness,
             "Currently configured contact distance threshold.")
        .def("set_self_collision_stiffness",
             &chysx::cloth::ClothSimulator::set_self_collision_stiffness,
             py::arg("stiffness"),
             R"pbdoc(
Set the per-contact penalty stiffness ``k`` [N/m].  cuda-cloth's
twist case uses ``k = 1000`` for VF/EE (m_4_k); larger values produce
stiffer contact response at the cost of PCG conditioning.
)pbdoc")
        .def("self_collision_stiffness",
             &chysx::cloth::ClothSimulator::self_collision_stiffness,
             "Currently configured contact penalty stiffness.")
        .def("set_self_collision_friction",
             &chysx::cloth::ClothSimulator::set_self_collision_friction,
             py::arg("friction"),
             R"pbdoc(
Set the IPC-style Coulomb friction coefficient ``μ`` (dimensionless)
applied at every VF / EE self-contact pair.  Folded into the same
kernels that already process self-contact (no extra launches, no new
sparsity).  ``μ = 0`` (default) disables friction; typical fabric
values land between 0.2 (slippery synthetic) and 0.6 (cotton).
)pbdoc")
        .def("self_collision_friction",
             &chysx::cloth::ClothSimulator::self_collision_friction,
             "Currently configured self-collision friction coefficient.")
        .def("set_self_collision_friction_epsilon",
             &chysx::cloth::ClothSimulator::set_self_collision_friction_epsilon,
             py::arg("epsilon"),
             R"pbdoc(
Set the tangential slip regularisation distance ``ε_u`` [m] used by
the IPC ``f1_SF_over_x`` smoothing function.  Tangential displacements
smaller than ``ε_u`` ramp friction force linearly with slip; past that
the force saturates at the Coulomb limit ``μ · f_n``.  ``1e-4 m`` is a
reasonable default for cloth at millimetre cell sizes.
)pbdoc")
        .def("self_collision_friction_epsilon",
             &chysx::cloth::ClothSimulator::self_collision_friction_epsilon,
             "Currently configured self-collision friction epsilon.")
        .def("set_self_collision_max_contacts",
             &chysx::cloth::ClothSimulator::set_self_collision_max_contacts,
             py::arg("max_contacts"),
             py::arg("max_ef_candidates") = 0,
             R"pbdoc(
Allocate (or grow) the device-side contact buffer to hold up to
``max_contacts`` simultaneous contacts plus the LBVH broadphase
EF-candidate list (default cap = max_contacts).  Detector overflow
past these caps silently drops the newest pairs; size generously
(e.g. ``8 * particle_count``) for typical cloth.
)pbdoc")
        .def("self_collision_max_contacts",
             &chysx::cloth::ClothSimulator::self_collision_max_contacts,
             "Currently allocated contact buffer capacity.")
        .def(
            "self_collision_count",
            [](chysx::cloth::ClothSimulator& s,
               std::uintptr_t cuda_stream) {
                return s.self_collision_detector().count(cuda_stream);
            },
            py::arg("cuda_stream") = 0,
            "Number of contacts emitted by the most recent step (synchronous read).")
        // ---- untangle (5-vertex EF tangle, ray-tri intersection) ----
        .def("set_untangle_enabled",
             &chysx::cloth::ClothSimulator::set_untangle_enabled,
             py::arg("enabled"),
             R"pbdoc(
Toggle the 5-vertex EF tangle / untangle pass.  Reuses the BVH built
by the proximity self-collision pass, so requires
``self_collision_enabled = True`` to do anything.  Pushes apart edges
that have already pierced through faces; complements the proximity
pass which only fires for not-yet-crossed pairs.
)pbdoc")
        .def("untangle_enabled",
             &chysx::cloth::ClothSimulator::untangle_enabled,
             "True if the untangle pass is currently enabled.")
        .def("set_untangle_thickness",
             &chysx::cloth::ClothSimulator::set_untangle_thickness,
             py::arg("thickness"),
             R"pbdoc(
Set the per-tangle restoring depth (world units).  Applied as a
constant penalty depth for every (edge, face) pair that intersects.
Larger values produce a stronger untangle force per crossing; the
proximity self-collision thickness is what controls *when* a pair
becomes a contact, so it's fine to use a larger untangle thickness
than the proximity thickness.
)pbdoc")
        .def("untangle_thickness",
             &chysx::cloth::ClothSimulator::untangle_thickness,
             "Currently configured untangle restoring depth.")
        .def("set_untangle_stiffness",
             &chysx::cloth::ClothSimulator::set_untangle_stiffness,
             py::arg("stiffness"),
             R"pbdoc(
Set the untangle penalty stiffness ``k`` [N/m].  cuda-cloth's
Untangle case uses ``k = 100`` for the EF (5-vertex) path.
)pbdoc")
        .def("untangle_stiffness",
             &chysx::cloth::ClothSimulator::untangle_stiffness,
             "Currently configured untangle penalty stiffness.")
        .def("set_untangle_max_contacts",
             &chysx::cloth::ClothSimulator::set_untangle_max_contacts,
             py::arg("max_contacts"),
             R"pbdoc(
Allocate (or grow) the untangle 5-vertex contact buffer to hold up
to ``max_contacts`` simultaneous tangles.  Pass ``0`` to default to
the proximity-self-collision cap (typical worst case is
"every proximity contact also tangled", which is loose but safe).
)pbdoc")
        .def("untangle_max_contacts",
             &chysx::cloth::ClothSimulator::untangle_max_contacts,
             "Currently allocated untangle contact buffer capacity.")
        .def(
            "untangle_count",
            [](chysx::cloth::ClothSimulator& s,
               std::uintptr_t cuda_stream) {
                return s.untangle_detector().count(cuda_stream);
            },
            py::arg("cuda_stream") = 0,
            "Number of tangle contacts emitted by the most recent step (synchronous read).")
        .def("set_pcg_iterations",
             &chysx::cloth::ClothSimulator::set_pcg_iterations,
             py::arg("max_iter"),
             "Set the maximum number of PCG iterations per step.")
        .def("pcg_iterations", &chysx::cloth::ClothSimulator::pcg_iterations,
             "Currently configured maximum PCG iterations per step.")
        // ---- diagnostics: dump the last solve's linear system -------
        .def("debug_dump_last_solve",
             [](const chysx::cloth::ClothSimulator& s) {
                 const int n = s.num_particles();
                 const int nnz = s.num_off_diag_blocks();

                 py::array_t<float> diag({n, 3, 3});
                 py::array_t<int>   row_offsets({n + 1});
                 py::array_t<int>   col_indices({nnz});
                 py::array_t<float> values({nnz, 3, 3});
                 py::array_t<float> rhs({n, 3});
                 py::array_t<float> dx({n, 3});

                 if (n == 0) {
                     return py::make_tuple(diag, row_offsets, col_indices,
                                           values, rhs, dx);
                 }

                 s.debug_copy_hessian_diag(diag.mutable_data());
                 s.debug_copy_hessian_csr(row_offsets.mutable_data(),
                                          nnz > 0 ? col_indices.mutable_data() : nullptr,
                                          nnz > 0 ? values.mutable_data() : nullptr);
                 s.debug_copy_last_rhs(rhs.mutable_data());
                 s.debug_copy_last_dx(dx.mutable_data());

                 return py::make_tuple(diag, row_offsets, col_indices,
                                       values, rhs, dx);
             },
             R"pbdoc(
Return ``(diag, row_offsets, col_indices, values, rhs, dx)`` for the
linear system solved by the most recent ``step(...)``.

Shapes:
    diag         : (N, 3, 3)        per-particle 3x3 Hessian diagonal
    row_offsets  : (N + 1,)         CSR row pointer (off-diagonal)
    col_indices  : (nnz_off,)       block-column indices
    values       : (nnz_off, 3, 3)  off-diagonal 3x3 blocks
    rhs          : (N, 3)           right-hand side b
    dx           : (N, 3)           solution returned by PCG

This synchronises the device and copies through host buffers, so
don't call it inside the simulation loop.  The return is a snapshot
of whatever was in those buffers when ``step()`` finished — perfect
for verifying matrix symmetry, PSD-ness, and PCG residual offline.
)pbdoc")
        // ---- static-shape contact (cloth ⇄ planes / boxes) ----------
        .def(
            "add_static_plane",
            [](chysx::cloth::ClothSimulator& s,
               py::array_t<float, py::array::c_style | py::array::forcecast> n,
               float d) {
                if (n.ndim() != 1 || n.shape(0) != 3) {
                    throw std::invalid_argument(
                        "ClothSimulator.add_static_plane: n must be a "
                        "(3,) float32 vector");
                }
                chysx::collision::PlaneShape p;
                p.n = chysx::math::Vec3f(n.at(0), n.at(1), n.at(2));
                p.d = d;
                s.add_static_plane(p);
            },
            py::arg("n"),
            py::arg("d"),
            R"pbdoc(
Add a static plane primitive ``dot(n, x) + d == 0`` to the contact
set.  ``n`` must point into the half-space the cloth should stay in
(the "outside"); for a ground at ``z = h`` use ``n = (0, 0, 1)`` and
``d = -h``.
)pbdoc")
        .def(
            "add_static_box",
            [](chysx::cloth::ClothSimulator& s,
               py::array_t<float, py::array::c_style | py::array::forcecast> center,
               py::array_t<float, py::array::c_style | py::array::forcecast> half_ext,
               py::array_t<float, py::array::c_style | py::array::forcecast> ex,
               py::array_t<float, py::array::c_style | py::array::forcecast> ey,
               py::array_t<float, py::array::c_style | py::array::forcecast> ez) {
                auto check_v3 = [](auto& a, const char* name) {
                    if (a.ndim() != 1 || a.shape(0) != 3) {
                        throw std::invalid_argument(
                            std::string("ClothSimulator.add_static_box: ") +
                            name + " must be a (3,) float32 vector");
                    }
                };
                check_v3(center,   "center");
                check_v3(half_ext, "half_ext");
                check_v3(ex,       "ex");
                check_v3(ey,       "ey");
                check_v3(ez,       "ez");
                chysx::collision::BoxShape b;
                b.center   = chysx::math::Vec3f(center.at(0), center.at(1), center.at(2));
                b.half_ext = chysx::math::Vec3f(half_ext.at(0), half_ext.at(1), half_ext.at(2));
                b.ex       = chysx::math::Vec3f(ex.at(0), ex.at(1), ex.at(2));
                b.ey       = chysx::math::Vec3f(ey.at(0), ey.at(1), ey.at(2));
                b.ez       = chysx::math::Vec3f(ez.at(0), ez.at(1), ez.at(2));
                s.add_static_box(b);
            },
            py::arg("center"),
            py::arg("half_ext"),
            py::arg("ex"),
            py::arg("ey"),
            py::arg("ez"),
            R"pbdoc(
Add a static oriented box primitive.  ``(ex, ey, ez)`` are the three
unit columns of the box's world-space rotation; an axis-aligned box
uses identity (``ex=(1,0,0)``, ``ey=(0,1,0)``, ``ez=(0,0,1)``).
``half_ext`` are the half-extents along ``(ex, ey, ez)``.
)pbdoc")
        .def("clear_static_shapes",
             &chysx::cloth::ClothSimulator::clear_static_shapes,
             "Drop every previously added static plane / box.")
        .def("set_static_contact_thickness",
             &chysx::cloth::ClothSimulator::set_static_contact_thickness,
             py::arg("thickness"),
             "Contact distance threshold ``h`` in world units.")
        .def("static_contact_thickness",
             &chysx::cloth::ClothSimulator::static_contact_thickness,
             "Currently configured static-shape contact thickness.")
        .def("set_static_contact_stiffness",
             &chysx::cloth::ClothSimulator::set_static_contact_stiffness,
             py::arg("stiffness"),
             "Per-contact penalty stiffness ``k`` [N/m].")
        .def("static_contact_stiffness",
             &chysx::cloth::ClothSimulator::static_contact_stiffness,
             "Currently configured static-shape contact stiffness.")
        .def("set_static_contact_friction",
             &chysx::cloth::ClothSimulator::set_static_contact_friction,
             py::arg("mu"),
             "Coulomb friction coefficient ``μ`` (dimensionless).  Adds a "
             "Lagged-Newton IPC isotropic Coulomb friction block to A's "
             "diagonal that self-caps at ``μ · f_n``.  Zero disables friction.")
        .def("static_contact_friction",
             &chysx::cloth::ClothSimulator::static_contact_friction,
             "Currently configured static-shape Coulomb friction coefficient.")
        .def("set_static_contact_friction_epsilon",
             &chysx::cloth::ClothSimulator::set_static_contact_friction_epsilon,
             py::arg("eps_u"),
             "Tangential slip regularisation distance ``ε_u`` [m] for the "
             "Coulomb friction model (default 1e-4).  Smaller values stick "
             "harder; larger values produce a softer ramp-up to the cone.")
        .def("static_contact_friction_epsilon",
             &chysx::cloth::ClothSimulator::static_contact_friction_epsilon,
             "Currently configured static-shape friction regularisation ``ε_u``.")
        .def("static_plane_count",
             &chysx::cloth::ClothSimulator::static_plane_count,
             "Number of registered static planes.")
        .def("static_box_count",
             &chysx::cloth::ClothSimulator::static_box_count,
             "Number of registered static boxes.")
        // ---- SDF-volume contacts (cloth ⇄ animated implicit bodies) ----
        //
        // The simulator may hold zero, one, or many SDF bodies.  Each
        // body is identified by a stable integer ``volume_index``
        // returned by ``add_sdf_volume()``; pass it to every bake /
        // pose / parameter setter.  ``volume_index = 0`` is the
        // legacy single-body slot (no-op for users that already
        // called ``add_sdf_volume`` exactly once at setup).
        .def("add_sdf_volume",
             [](chysx::cloth::ClothSimulator& s) {
                 return s.add_sdf_volume();
             },
             R"pbdoc(
Allocate a new SDF volume + contact pair and return its index.

Call once per animated implicit body before any ``bake_sdf_box`` /
``set_sdf_pose`` call.  Subsequent SDF API calls take this index as
their first argument.  Volumes added later are appended to the
internal vector; indices are stable for the lifetime of the
simulator.
)pbdoc")
        .def("num_sdf_volumes",
             &chysx::cloth::ClothSimulator::num_sdf_volumes,
             "Number of SDF volumes currently held by the simulator.")
        .def(
            "bake_sdf_box",
            [](chysx::cloth::ClothSimulator& s, int volume_index,
               float hx, float hy, float hz,
               float voxel_size, float padding, float corner_radius) {
                s.sdf_volume(volume_index)
                    .bake_box(hx, hy, hz, voxel_size, padding, corner_radius);
            },
            py::arg("volume_index"),
            py::arg("hx"),
            py::arg("hy"),
            py::arg("hz"),
            py::arg("voxel_size"),
            py::arg("padding") = -1.0f,
            py::arg("corner_radius") = 0.0f,
            R"pbdoc(
Bake an analytic axis-aligned box SDF into volume ``volume_index``.

The box is centred at the volume's local origin with half-extents
``(hx, hy, hz)``.  The grid is padded by ``padding`` on every axis so
the SDF stays monotonic past the surface (defaults to
``2 · voxel_size`` when ``padding < 0``).  Call once at setup; per-
frame motion is configured separately via ``set_sdf_pose(...)``.

Parameters
----------
volume_index : int
    Index returned by a prior ``add_sdf_volume()`` call.
hx, hy, hz : float
    Half-extents of the box along its local x/y/z axes [m].
voxel_size : float
    Edge length of the cubic voxels used to sample the SDF [m].
    Pick small enough that ``voxel_size << min(hx, hy, hz)`` so the
    trilinear gradient near the surface is well-resolved (a couple
    of voxels per particle-thickness band is usually plenty).
padding : float, optional
    Extra padding on every axis [m]; default is ``2 · voxel_size``.
corner_radius : float, optional
    Edge rounding radius [m].  ``0`` keeps sharp edges; ``>0`` bakes
    a rounded box SDF (continuous normals around box edges/corners),
    which is often more stable for cloth contact.
)pbdoc")
        .def(
            "bake_sdf_from_host",
            [](chysx::cloth::ClothSimulator& s, int volume_index,
               py::array_t<float, py::array::c_style | py::array::forcecast> values,
               int nx, int ny, int nz,
               float voxel_size,
               py::array_t<float, py::array::c_style | py::array::forcecast> origin) {
                if (values.ndim() != 1 ||
                    values.shape(0) != static_cast<py::ssize_t>(nx) * ny * nz) {
                    throw std::invalid_argument(
                        "ClothSimulator.bake_sdf_from_host: values must be a "
                        "flat (nx*ny*nz,) float32 array in x-major order");
                }
                if (origin.ndim() != 1 || origin.shape(0) != 3) {
                    throw std::invalid_argument(
                        "ClothSimulator.bake_sdf_from_host: origin must be a "
                        "(3,) float32 vector");
                }
                s.sdf_volume(volume_index).bake_from_host(
                    values.data(),
                    nx, ny, nz,
                    voxel_size,
                    chysx::math::Vec3f(origin.at(0), origin.at(1), origin.at(2)));
            },
            py::arg("volume_index"),
            py::arg("values"),
            py::arg("nx"),
            py::arg("ny"),
            py::arg("nz"),
            py::arg("voxel_size"),
            py::arg("origin"),
            R"pbdoc(
Bake a precomputed SDF grid into volume ``volume_index``.

``values`` is a flat ``(nx*ny*nz,)`` float32 array in **x-major**
(x fastest) order holding signed distances.  ``origin`` is the
world-space position of grid corner ``(0, 0, 0)`` expressed in
the volume's local frame.  Call once at setup; animate the body
with ``set_sdf_pose_full(...)`` each frame.
)pbdoc")
        .def(
            "set_sdf_pose",
            [](chysx::cloth::ClothSimulator& s, int volume_index,
               py::array_t<float, py::array::c_style | py::array::forcecast> pos) {
                if (pos.ndim() != 1 || pos.shape(0) != 3) {
                    throw std::invalid_argument(
                        "ClothSimulator.set_sdf_pose: pos must be a "
                        "(3,) float32 vector");
                }
                s.sdf_volume(volume_index).set_pose_translation(
                    chysx::math::Vec3f(pos.at(0), pos.at(1), pos.at(2)));
            },
            py::arg("volume_index"),
            py::arg("pos"),
            R"pbdoc(
Set volume ``volume_index``'s world pose to a pure translation
(identity rotation).  Use this every frame for translating bodies;
combine with ``set_sdf_body_velocity`` so the friction slip cache
uses relative tangential motion.
)pbdoc")
        .def(
            "set_sdf_pose_full",
            [](chysx::cloth::ClothSimulator& s, int volume_index,
               py::array_t<float, py::array::c_style | py::array::forcecast> pos,
               py::array_t<float, py::array::c_style | py::array::forcecast> ex,
               py::array_t<float, py::array::c_style | py::array::forcecast> ey,
               py::array_t<float, py::array::c_style | py::array::forcecast> ez) {
                auto v3 = [](auto& a, const char* name) {
                    if (a.ndim() != 1 || a.shape(0) != 3) {
                        throw std::invalid_argument(
                            std::string("ClothSimulator.set_sdf_pose_full: ") +
                            name + " must be a (3,) float32 vector");
                    }
                    return chysx::math::Vec3f(a.at(0), a.at(1), a.at(2));
                };
                s.sdf_volume(volume_index).set_pose(
                    v3(pos, "pos"), v3(ex, "ex"),
                    v3(ey, "ey"), v3(ez, "ez"));
            },
            py::arg("volume_index"),
            py::arg("pos"),
            py::arg("ex"),
            py::arg("ey"),
            py::arg("ez"),
            R"pbdoc(
Set volume ``volume_index``'s full world pose: position + the three
orthonormal column vectors of its rotation matrix.  ``(ex, ey, ez)``
map the body's local +x/+y/+z axes into world space.  Use the
simpler ``set_sdf_pose(...)`` if your body never rotates.
)pbdoc")
        .def(
            "set_sdf_body_velocity",
            [](chysx::cloth::ClothSimulator& s, int volume_index,
               py::array_t<float, py::array::c_style | py::array::forcecast> v) {
                if (v.ndim() != 1 || v.shape(0) != 3) {
                    throw std::invalid_argument(
                        "ClothSimulator.set_sdf_body_velocity: v must be a "
                        "(3,) float32 vector");
                }
                s.sdf_contact(volume_index).set_body_velocity(
                    chysx::math::Vec3f(v.at(0), v.at(1), v.at(2)));
            },
            py::arg("volume_index"),
            py::arg("v"),
            R"pbdoc(
Set volume ``volume_index``'s linear velocity in world frame [m/s].
Subtracted from each cloth particle's velocity before projecting
onto the contact tangent, so a particle riding the body sees zero
spurious slip (and therefore zero spurious friction).  Default
``(0, 0, 0)``.
)pbdoc")
        .def("set_sdf_contact_thickness",
             [](chysx::cloth::ClothSimulator& s, int volume_index, float t) {
                 s.sdf_contact(volume_index).set_thickness(t);
             },
             py::arg("volume_index"),
             py::arg("thickness"),
             "Contact distance threshold ``h`` [m] for SDF body "
             "``volume_index``.  Particles with ``sdf(x) < h`` are "
             "in contact.")
        .def("sdf_contact_thickness",
             [](const chysx::cloth::ClothSimulator& s, int volume_index) {
                 return s.sdf_contact(volume_index).thickness();
             },
             py::arg("volume_index"),
             "Currently configured SDF contact thickness for the body.")
        .def("set_sdf_contact_stiffness",
             [](chysx::cloth::ClothSimulator& s, int volume_index, float k) {
                 s.sdf_contact(volume_index).set_stiffness(k);
             },
             py::arg("volume_index"),
             py::arg("stiffness"),
             "Per-contact penalty stiffness ``k`` [N/m] for SDF body "
             "``volume_index``.")
        .def("sdf_contact_stiffness",
             [](const chysx::cloth::ClothSimulator& s, int volume_index) {
                 return s.sdf_contact(volume_index).stiffness();
             },
             py::arg("volume_index"),
             "Currently configured SDF contact stiffness for the body.")
        .def("set_sdf_contact_friction",
             [](chysx::cloth::ClothSimulator& s, int volume_index, float mu) {
                 s.sdf_contact(volume_index).set_friction(mu);
             },
             py::arg("volume_index"),
             py::arg("mu"),
             "IPC-style Coulomb friction coefficient ``μ`` "
             "(dimensionless) for SDF body ``volume_index``.  Zero "
             "disables friction.")
        .def("sdf_contact_friction",
             [](const chysx::cloth::ClothSimulator& s, int volume_index) {
                 return s.sdf_contact(volume_index).friction();
             },
             py::arg("volume_index"),
             "Currently configured SDF Coulomb friction coefficient.")
        .def("set_sdf_contact_friction_epsilon",
             [](chysx::cloth::ClothSimulator& s, int volume_index, float eps) {
                 s.sdf_contact(volume_index).set_friction_epsilon(eps);
             },
             py::arg("volume_index"),
             py::arg("epsilon"),
             "Tangential slip regularisation velocity ``ε_u`` [m/s] for the "
             "IPC smooth Coulomb ramp.  The kernel scales this by ``dt`` to "
             "get the displacement threshold.")
        .def("sdf_contact_friction_epsilon",
             [](const chysx::cloth::ClothSimulator& s, int volume_index) {
                 return s.sdf_contact(volume_index).friction_epsilon();
             },
             py::arg("volume_index"),
             "Currently configured IPC friction epsilon.")
        .def("set_sdf_contact_kd",
             [](chysx::cloth::ClothSimulator& s, int volume_index, float kd) {
                 s.sdf_contact(volume_index).set_contact_kd(kd);
             },
             py::arg("volume_index"),
             py::arg("kd"),
             "Contact damping ratio for SDF body ``volume_index``.")
        .def("sdf_contact_kd",
             [](const chysx::cloth::ClothSimulator& s, int volume_index) {
                 return s.sdf_contact(volume_index).contact_kd();
             },
             py::arg("volume_index"),
             "Currently configured contact damping ratio.")
        .def("set_sdf_ipc_friction_enabled",
             [](chysx::cloth::ClothSimulator& s, int volume_index, bool v) {
                 s.sdf_contact(volume_index).set_ipc_friction_enabled(v);
             },
             py::arg("volume_index"),
             py::arg("enabled"),
             "Enable IPC-style implicit friction for SDF body "
             "``volume_index``.  When enabled, friction is baked into "
             "gradient/Hessian (matching VBD) instead of Coulomb post-projection.")
        .def("sdf_ipc_friction_enabled",
             [](const chysx::cloth::ClothSimulator& s, int volume_index) {
                 return s.sdf_contact(volume_index).ipc_friction_enabled();
             },
             py::arg("volume_index"),
             "True if IPC friction is enabled for SDF body ``volume_index``.")
        .def("sdf_volume_active",
             [](const chysx::cloth::ClothSimulator& s, int volume_index) {
                 return s.sdf_volume(volume_index).active();
             },
             py::arg("volume_index"),
             "True if SDF volume ``volume_index`` has been baked at "
             "least once.")
        .def("sdf_volume_shape",
             [](const chysx::cloth::ClothSimulator& s, int volume_index) {
                 const auto& v = s.sdf_volume(volume_index);
                 return py::make_tuple(v.nx(), v.ny(), v.nz());
             },
             py::arg("volume_index"),
             "(nx, ny, nz) voxel resolution of the baked SDF.")
        // ---- mesh-body contacts (BVH-accelerated triangle mesh) --------
        .def("add_mesh_body",
             [](chysx::cloth::ClothSimulator& s) { return s.add_mesh_body(); },
             "Allocate a new mesh-body contact slot and return its index.")
        .def("num_mesh_bodies",
             &chysx::cloth::ClothSimulator::num_mesh_bodies,
             "Number of mesh bodies currently held.")
        .def(
            "set_mesh_body_mesh",
            [](chysx::cloth::ClothSimulator& s, int idx,
               py::array_t<float, py::array::c_style | py::array::forcecast> verts,
               py::array_t<int, py::array::c_style | py::array::forcecast> indices) {
                if (verts.ndim() != 2 || verts.shape(1) != 3)
                    throw std::invalid_argument("verts must be (N, 3) float32");
                if (indices.ndim() != 1 || (indices.shape(0) % 3) != 0)
                    throw std::invalid_argument("indices must be flat with length divisible by 3");
                int nv = static_cast<int>(verts.shape(0));
                int nt = static_cast<int>(indices.shape(0)) / 3;
                s.mesh_contact(idx).set_mesh(
                    reinterpret_cast<const chysx::math::Vec3f*>(verts.data()),
                    nv, indices.data(), nt);
            },
            py::arg("mesh_body_index"),
            py::arg("vertices"),
            py::arg("indices"),
            "Upload a triangle mesh (rest pose) for mesh body ``mesh_body_index``.")
        .def(
            "set_mesh_body_pose",
            [](chysx::cloth::ClothSimulator& s, int idx,
               py::array_t<float, py::array::c_style | py::array::forcecast> pos,
               py::array_t<float, py::array::c_style | py::array::forcecast> ex,
               py::array_t<float, py::array::c_style | py::array::forcecast> ey,
               py::array_t<float, py::array::c_style | py::array::forcecast> ez) {
                auto v3 = [](auto& a, const char* name) {
                    if (a.ndim() != 1 || a.shape(0) != 3)
                        throw std::invalid_argument(
                            std::string("set_mesh_body_pose: ") + name +
                            " must be (3,) float32");
                    return chysx::math::Vec3f(a.at(0), a.at(1), a.at(2));
                };
                s.mesh_contact(idx).set_pose(
                    v3(pos, "pos"), v3(ex, "ex"), v3(ey, "ey"), v3(ez, "ez"));
            },
            py::arg("mesh_body_index"),
            py::arg("pos"), py::arg("ex"), py::arg("ey"), py::arg("ez"),
            "Set mesh body's world pose (pos + rotation columns).")
        .def("set_mesh_body_velocity",
             [](chysx::cloth::ClothSimulator& s, int idx,
                py::array_t<float, py::array::c_style | py::array::forcecast> v) {
                 if (v.ndim() != 1 || v.shape(0) != 3)
                     throw std::invalid_argument("v must be (3,) float32");
                 s.mesh_contact(idx).set_body_velocity(
                     chysx::math::Vec3f(v.at(0), v.at(1), v.at(2)));
             },
             py::arg("mesh_body_index"), py::arg("v"),
             "Set mesh body's linear velocity [m/s].")
        .def("set_mesh_body_thickness",
             [](chysx::cloth::ClothSimulator& s, int idx, float t) {
                 s.mesh_contact(idx).set_thickness(t);
             },
             py::arg("mesh_body_index"), py::arg("thickness"))
        .def("set_mesh_body_stiffness",
             [](chysx::cloth::ClothSimulator& s, int idx, float k) {
                 s.mesh_contact(idx).set_stiffness(k);
             },
             py::arg("mesh_body_index"), py::arg("stiffness"))
        .def("set_mesh_body_friction",
             [](chysx::cloth::ClothSimulator& s, int idx, float mu) {
                 s.mesh_contact(idx).set_friction(mu);
             },
             py::arg("mesh_body_index"), py::arg("friction"))
        .def("set_mesh_body_ipc_friction",
             [](chysx::cloth::ClothSimulator& s, int idx, bool v) {
                 s.mesh_contact(idx).set_ipc_friction_enabled(v);
             },
             py::arg("mesh_body_index"), py::arg("enabled"))
        .def("set_mesh_body_friction_epsilon",
             [](chysx::cloth::ClothSimulator& s, int idx, float eps) {
                 s.mesh_contact(idx).set_friction_epsilon(eps);
             },
             py::arg("mesh_body_index"), py::arg("epsilon"))
        .def("set_mesh_body_contact_kd",
             [](chysx::cloth::ClothSimulator& s, int idx, float kd) {
                 s.mesh_contact(idx).set_contact_kd(kd);
             },
             py::arg("mesh_body_index"), py::arg("kd"))
        .def("set_mesh_body_search_radius",
             [](chysx::cloth::ClothSimulator& s, int idx, float r) {
                 s.mesh_contact(idx).set_search_radius(r);
             },
             py::arg("mesh_body_index"), py::arg("radius"),
             "BVH query radius for deep penetration recovery. 0 = 10*thickness.")
        .def_property_readonly(
            "material",
            [](chysx::cloth::ClothSimulator& s) -> chysx::cloth::ClothMaterial& {
                return s.material();
            },
            py::return_value_policy::reference_internal,
            "In-place reference to the simulator's material (mutate freely).");

    // ---- RigidSimulator ---------------------------------------------------

    py::enum_<chysx::rigid::JointTypeArg>(m, "JointType")
        .value("BALL", chysx::rigid::JointTypeArg::Ball)
        .value("FIXED", chysx::rigid::JointTypeArg::Fixed);

    py::class_<chysx::rigid::RigidSimulator>(m, "RigidSimulator", R"pbdoc(
AVBD rigid-body physics simulator.

Build a scene by adding bodies, shapes, and joints, then call
``finalize()`` to upload to GPU and ``step(dt)`` to advance.
)pbdoc")
        .def(py::init<>())
        .def("add_body",
             [](chysx::rigid::RigidSimulator& s,
                float mass,
                py::array_t<float> inertia_arr,
                py::array_t<float> com_arr,
                py::array_t<float> pos_arr,
                py::array_t<float> quat_arr) -> int {
                 auto ip = inertia_arr.unchecked<2>();
                 chysx::math::Mat3f I(
                     ip(0,0), ip(0,1), ip(0,2),
                     ip(1,0), ip(1,1), ip(1,2),
                     ip(2,0), ip(2,1), ip(2,2));
                 auto cp = com_arr.unchecked<1>();
                 auto pp = pos_arr.unchecked<1>();
                 auto qp = quat_arr.unchecked<1>();
                 return s.add_body(
                     mass, I,
                     chysx::math::Vec3f(cp(0), cp(1), cp(2)),
                     chysx::math::Vec3f(pp(0), pp(1), pp(2)),
                     chysx::math::Quatf(qp(0), qp(1), qp(2), qp(3)));
             },
             py::arg("mass"), py::arg("inertia"), py::arg("com"),
             py::arg("pos"), py::arg("quat"),
             "Add a rigid body.  inertia: (3,3), com/pos: (3,), quat: (4,) xyzw.")
        .def("add_shape_sphere",
             &chysx::rigid::RigidSimulator::add_shape_sphere,
             py::arg("body"), py::arg("radius"),
             py::arg("ke") = 1e4f, py::arg("kd") = 10.f,
             py::arg("mu") = 0.5f, py::arg("gap") = 0.01f)
        .def("add_shape_box",
             [](chysx::rigid::RigidSimulator& s, int body,
                py::array_t<float> half_arr,
                float ke, float kd, float mu, float gap) -> int {
                 auto h = half_arr.unchecked<1>();
                 return s.add_shape_box(body,
                     chysx::math::Vec3f(h(0), h(1), h(2)),
                     ke, kd, mu, gap);
             },
             py::arg("body"), py::arg("half_extents"),
             py::arg("ke") = 1e4f, py::arg("kd") = 10.f,
             py::arg("mu") = 0.5f, py::arg("gap") = 0.01f)
        .def("add_shape_capsule",
             &chysx::rigid::RigidSimulator::add_shape_capsule,
             py::arg("body"), py::arg("radius"), py::arg("half_height"),
             py::arg("ke") = 1e4f, py::arg("kd") = 10.f,
             py::arg("mu") = 0.5f, py::arg("gap") = 0.01f)
        .def("add_ground_plane",
             &chysx::rigid::RigidSimulator::add_ground_plane,
             py::arg("ke") = 1e4f, py::arg("kd") = 10.f,
             py::arg("mu") = 0.5f)
        .def("add_joint",
             [](chysx::rigid::RigidSimulator& s,
                chysx::rigid::JointTypeArg type,
                int parent, int child,
                py::array_t<float> ap, py::array_t<float> fp,
                py::array_t<float> ac, py::array_t<float> fc) -> int {
                 auto a1 = ap.unchecked<1>(); auto f1 = fp.unchecked<1>();
                 auto a2 = ac.unchecked<1>(); auto f2 = fc.unchecked<1>();
                 return s.add_joint(type, parent, child,
                     chysx::math::Vec3f(a1(0), a1(1), a1(2)),
                     chysx::math::Quatf(f1(0), f1(1), f1(2), f1(3)),
                     chysx::math::Vec3f(a2(0), a2(1), a2(2)),
                     chysx::math::Quatf(f2(0), f2(1), f2(2), f2(3)));
             },
             py::arg("type"), py::arg("parent"), py::arg("child"),
             py::arg("anchor_parent"), py::arg("frame_parent"),
             py::arg("anchor_child"), py::arg("frame_child"))
        .def("finalize", &chysx::rigid::RigidSimulator::finalize)
        .def("step", &chysx::rigid::RigidSimulator::step,
             py::arg("dt"), py::arg("cuda_stream") = 0)
        .def("body_count", &chysx::rigid::RigidSimulator::body_count)
        .def("shape_count", &chysx::rigid::RigidSimulator::shape_count)
        .def("joint_count", &chysx::rigid::RigidSimulator::joint_count)
        .def("contact_count", &chysx::rigid::RigidSimulator::contact_count)
        .def("get_body_poses",
             [](chysx::rigid::RigidSimulator& s) {
                 int n = s.body_count();
                 py::array_t<float> pos({n, 3});
                 py::array_t<float> quat({n, 4});
                 s.get_body_poses(
                     reinterpret_cast<chysx::math::Vec3f*>(pos.mutable_data()),
                     reinterpret_cast<chysx::math::Quatf*>(quat.mutable_data()));
                 return py::make_tuple(pos, quat);
             },
             "Returns (positions (N,3), quaternions (N,4)) as numpy arrays.")
        .def("get_body_velocities",
             [](chysx::rigid::RigidSimulator& s) {
                 int n = s.body_count();
                 py::array_t<float> vel({n, 3});
                 py::array_t<float> omega({n, 3});
                 s.get_body_velocities(
                     reinterpret_cast<chysx::math::Vec3f*>(vel.mutable_data()),
                     reinterpret_cast<chysx::math::Vec3f*>(omega.mutable_data()));
                 return py::make_tuple(vel, omega);
             },
             "Returns (linear_vel (N,3), angular_vel (N,3)) as numpy arrays.")
        .def("set_iterations", &chysx::rigid::RigidSimulator::set_iterations, py::arg("n"))
        .def("set_gravity",
             [](chysx::rigid::RigidSimulator& s, py::array_t<float> g) {
                 auto a = g.unchecked<1>();
                 s.set_gravity(chysx::math::Vec3f(a(0), a(1), a(2)));
             },
             py::arg("gravity"))
        .def("set_contact_hard", &chysx::rigid::RigidSimulator::set_contact_hard, py::arg("hard"))
        .def("set_contact_history", &chysx::rigid::RigidSimulator::set_contact_history, py::arg("enabled"))
        .def("set_avbd_alpha", &chysx::rigid::RigidSimulator::set_avbd_alpha, py::arg("alpha"))
        .def("set_avbd_gamma", &chysx::rigid::RigidSimulator::set_avbd_gamma, py::arg("gamma"))
        .def("set_avbd_beta", &chysx::rigid::RigidSimulator::set_avbd_beta, py::arg("beta"))
        .def("set_friction_epsilon", &chysx::rigid::RigidSimulator::set_friction_epsilon, py::arg("eps"))
        .def("set_stick_motion_eps", &chysx::rigid::RigidSimulator::set_stick_motion_eps, py::arg("eps"))
        .def("set_stick_deadzone", &chysx::rigid::RigidSimulator::set_stick_deadzone, py::arg("enabled"))
        .def("set_contact_buffer_size", &chysx::rigid::RigidSimulator::set_contact_buffer_size, py::arg("n"))
        .def("set_per_body_contact_capacity", &chysx::rigid::RigidSimulator::set_per_body_contact_capacity, py::arg("n"))
        .def("set_max_broadphase_pairs", &chysx::rigid::RigidSimulator::set_max_broadphase_pairs, py::arg("n"));

    // ---- CoupledSimulator ------------------------------------------------

    py::class_<chysx::coupled::CoupledSimulator>(m, "CoupledSimulator", R"pbdoc(
VBD-based coupled rigid-soft body simulator.

Rigid bodies are solved externally (e.g. Featherstone); this class
only handles particle VBD + body-particle contact forces.
)pbdoc")
        .def(py::init<>())
        .def("build_coloring",
             [](chysx::coupled::CoupledSimulator& self,
                py::array_t<int, py::array::c_style | py::array::forcecast> tets,
                int n_particles) {
                 if (tets.ndim() != 2 || tets.shape(1) != 4)
                     throw std::runtime_error("tets must be (T, 4)");
                 const int n_tets = static_cast<int>(tets.shape(0));
                 auto* ptr = reinterpret_cast<const chysx::math::Vec4i*>(tets.data());
                 self.build_coloring(ptr, n_tets, n_particles);
                 self.build_adjacency(ptr, n_tets, n_particles);
             },
             py::arg("tets"), py::arg("n_particles"),
             "Build graph coloring + vertex-tet adjacency from (T,4) tet array.")
        .def("set_coloring",
             [](chysx::coupled::CoupledSimulator& self,
                py::array_t<int, py::array::c_style | py::array::forcecast> colors,
                py::array_t<int, py::array::c_style | py::array::forcecast> tets,
                int n_particles) {
                 if (colors.ndim() != 1 || colors.shape(0) != n_particles)
                     throw std::runtime_error("colors must be (n_particles,)");
                 self.set_coloring(colors.data(), n_particles);
                 if (tets.ndim() == 2 && tets.shape(1) == 4) {
                     const int n_tets = static_cast<int>(tets.shape(0));
                     auto* ptr = reinterpret_cast<const chysx::math::Vec4i*>(tets.data());
                     self.build_adjacency(ptr, n_tets, n_particles);
                 }
             },
             py::arg("colors"), py::arg("tets"), py::arg("n_particles"),
             "Import an external coloring and build adjacency from (T,4) tet array.")
        .def("num_colors",
             &chysx::coupled::CoupledSimulator::num_colors,
             "Number of colors in the VBD graph coloring.")
        .def("add_collision_shape",
             [](chysx::coupled::CoupledSimulator& self,
                int body, int geo_type,
                float sx, float sy, float sz,
                py::array_t<float, py::array::c_style | py::array::forcecast> local_tf,
                int flags, uint64_t mesh_id,
                float mat_ke, float mat_kd, float mat_mu) {
                 if (local_tf.ndim() != 1 || local_tf.shape(0) != 7)
                     throw std::runtime_error("local_tf must be (7,) float32");
                 self.add_collision_shape(
                     body, geo_type, sx, sy, sz,
                     local_tf.data(), flags, mesh_id,
                     mat_ke, mat_kd, mat_mu);
             },
             py::arg("body"), py::arg("geo_type"),
             py::arg("sx"), py::arg("sy"), py::arg("sz"),
             py::arg("local_tf"), py::arg("flags"),
             py::arg("mesh_id") = 0,
             py::arg("mat_ke") = 0.0f, py::arg("mat_kd") = 0.0f,
             py::arg("mat_mu") = 0.0f,
             "Register a collision shape for the internal collision pipeline.")
        .def("finalize_collision",
             &chysx::coupled::CoupledSimulator::finalize_collision,
             py::arg("max_soft_contacts") = 0,
             "Upload shape data to GPU and allocate contact buffers.")
        .def("step_with_collision",
             [](chysx::coupled::CoupledSimulator& self,
                std::uintptr_t pos_ptr,
                std::uintptr_t vel_ptr,
                std::uintptr_t inv_mass_ptr,
                int n_particles,
                std::uintptr_t tet_indices_ptr,
                std::uintptr_t tet_poses_ptr,
                std::uintptr_t tet_materials_ptr,
                int n_tets,
                float gx, float gy, float gz,
                float dt,
                int iterations,
                std::uintptr_t body_q_ptr,
                std::uintptr_t body_q_prev_ptr,
                int n_bodies,
                std::uintptr_t particle_radius_ptr,
                std::uintptr_t particle_flags_ptr,
                float margin,
                float friction_epsilon,
                float soft_contact_ke,
                float soft_contact_kd,
                float soft_contact_mu,
                std::uintptr_t cuda_stream) {

                 using namespace chysx;
                 DeviceSpan<math::Vec3f> pos(
                     reinterpret_cast<math::Vec3f*>(pos_ptr), n_particles);
                 DeviceSpan<math::Vec3f> vel(
                     reinterpret_cast<math::Vec3f*>(vel_ptr), n_particles);
                 DeviceSpan<float> inv_mass(
                     reinterpret_cast<float*>(inv_mass_ptr), n_particles);
                 DeviceSpan<math::Vec4i> tet_indices(
                     reinterpret_cast<math::Vec4i*>(tet_indices_ptr), n_tets);
                 DeviceSpan<math::Mat3f> tet_poses(
                     reinterpret_cast<math::Mat3f*>(tet_poses_ptr), n_tets);
                 DeviceSpan<math::Vec3f> tet_materials(
                     reinterpret_cast<math::Vec3f*>(tet_materials_ptr), n_tets);

                 self.step_with_collision(
                     pos, vel, inv_mass,
                     tet_indices, tet_poses, tet_materials,
                     math::Vec3f(gx, gy, gz),
                     dt, iterations,
                     reinterpret_cast<const float*>(body_q_ptr),
                     reinterpret_cast<const float*>(body_q_prev_ptr),
                     n_bodies,
                     reinterpret_cast<const float*>(particle_radius_ptr),
                     reinterpret_cast<const int*>(particle_flags_ptr),
                     margin, friction_epsilon,
                     soft_contact_ke, soft_contact_kd, soft_contact_mu,
                     cuda_stream);
             },
             py::arg("pos_ptr"),
             py::arg("vel_ptr"),
             py::arg("inv_mass_ptr"),
             py::arg("n_particles"),
             py::arg("tet_indices_ptr"),
             py::arg("tet_poses_ptr"),
             py::arg("tet_materials_ptr"),
             py::arg("n_tets"),
             py::arg("gx"), py::arg("gy"), py::arg("gz"),
             py::arg("dt"),
             py::arg("iterations"),
             py::arg("body_q_ptr"),
             py::arg("body_q_prev_ptr"),
             py::arg("n_bodies"),
             py::arg("particle_radius_ptr"),
             py::arg("particle_flags_ptr"),
             py::arg("margin"),
             py::arg("friction_epsilon"),
             py::arg("soft_contact_ke"),
             py::arg("soft_contact_kd"),
             py::arg("soft_contact_mu"),
             py::arg("cuda_stream") = 0,
             R"pbdoc(
Run one VBD substep using ChysX's internal collision pipeline.

Collision detection + material mixing + VBD solve are all performed
internally, so the caller does not need to run Newton's CollisionPipeline.
Requires add_collision_shape() + finalize_collision() to have been called.
)pbdoc")
        .def("step",
             [](chysx::coupled::CoupledSimulator& self,
                // Particle buffers (device pointers)
                std::uintptr_t pos_ptr,
                std::uintptr_t vel_ptr,
                std::uintptr_t inv_mass_ptr,
                int n_particles,
                // Tet buffers
                std::uintptr_t tet_indices_ptr,
                std::uintptr_t tet_poses_ptr,
                std::uintptr_t tet_materials_ptr,
                int n_tets,
                // Physics
                float gx, float gy, float gz,
                float dt,
                int iterations,
                float friction_epsilon,
                // Body-particle contact data (device pointers, optional)
                std::uintptr_t contact_particle_ptr,
                std::uintptr_t contact_count_ptr,
                int contact_max,
                std::uintptr_t contact_ke_ptr,
                std::uintptr_t contact_kd_ptr,
                std::uintptr_t contact_mu_ptr,
                std::uintptr_t contact_shape_ptr,
                std::uintptr_t contact_body_pos_ptr,
                std::uintptr_t contact_body_vel_ptr,
                std::uintptr_t contact_normal_ptr,
                // External body state (device pointers)
                std::uintptr_t body_q_ptr,
                std::uintptr_t body_q_prev_ptr,
                std::uintptr_t body_qd_ptr,
                std::uintptr_t body_com_ptr,
                std::uintptr_t shape_body_ptr,
                std::uintptr_t particle_radius_ptr,
                std::uintptr_t particle_colors_ptr,
                int n_bodies,
                int n_shapes,
                // Stream
                std::uintptr_t cuda_stream) {

                 using namespace chysx;

                 DeviceSpan<math::Vec3f> pos(
                     reinterpret_cast<math::Vec3f*>(pos_ptr), n_particles);
                 DeviceSpan<math::Vec3f> vel(
                     reinterpret_cast<math::Vec3f*>(vel_ptr), n_particles);
                 DeviceSpan<float> inv_mass(
                     reinterpret_cast<float*>(inv_mass_ptr), n_particles);
                 DeviceSpan<math::Vec4i> tet_indices(
                     reinterpret_cast<math::Vec4i*>(tet_indices_ptr), n_tets);
                 DeviceSpan<math::Mat3f> tet_poses(
                     reinterpret_cast<math::Mat3f*>(tet_poses_ptr), n_tets);
                 DeviceSpan<math::Vec3f> tet_materials(
                     reinterpret_cast<math::Vec3f*>(tet_materials_ptr), n_tets);

                 coupled::BodyParticleContacts contacts;
                 contacts.contact_particle = reinterpret_cast<int*>(contact_particle_ptr);
                 contacts.contact_count = reinterpret_cast<int*>(contact_count_ptr);
                 contacts.contact_max = contact_max;
                 contacts.contact_ke = reinterpret_cast<float*>(contact_ke_ptr);
                 contacts.contact_kd = reinterpret_cast<float*>(contact_kd_ptr);
                 contacts.contact_mu = reinterpret_cast<float*>(contact_mu_ptr);
                 contacts.contact_shape = reinterpret_cast<int*>(contact_shape_ptr);
                 contacts.contact_body_pos = reinterpret_cast<float*>(contact_body_pos_ptr);
                 contacts.contact_body_vel = reinterpret_cast<float*>(contact_body_vel_ptr);
                 contacts.contact_normal = reinterpret_cast<float*>(contact_normal_ptr);

                 coupled::ExternalBodies bodies;
                 bodies.body_q = reinterpret_cast<float*>(body_q_ptr);
                 bodies.body_q_prev = reinterpret_cast<float*>(body_q_prev_ptr);
                 bodies.body_qd = reinterpret_cast<float*>(body_qd_ptr);
                 bodies.body_com = reinterpret_cast<float*>(body_com_ptr);
                 bodies.shape_body = reinterpret_cast<int*>(shape_body_ptr);
                 bodies.particle_radius = reinterpret_cast<float*>(particle_radius_ptr);
                 bodies.particle_colors = reinterpret_cast<int*>(particle_colors_ptr);
                 bodies.n_bodies = n_bodies;
                 bodies.n_shapes = n_shapes;

                 self.step(
                     pos, vel, inv_mass,
                     tet_indices, tet_poses, tet_materials,
                     math::Vec3f(gx, gy, gz),
                     dt, iterations,
                     contacts, bodies, friction_epsilon,
                     cuda_stream);
             },
             py::arg("pos_ptr"),
             py::arg("vel_ptr"),
             py::arg("inv_mass_ptr"),
             py::arg("n_particles"),
             py::arg("tet_indices_ptr"),
             py::arg("tet_poses_ptr"),
             py::arg("tet_materials_ptr"),
             py::arg("n_tets"),
             py::arg("gx"), py::arg("gy"), py::arg("gz"),
             py::arg("dt"),
             py::arg("iterations"),
             py::arg("friction_epsilon"),
             py::arg("contact_particle_ptr") = 0,
             py::arg("contact_count_ptr") = 0,
             py::arg("contact_max") = 0,
             py::arg("contact_ke_ptr") = 0,
             py::arg("contact_kd_ptr") = 0,
             py::arg("contact_mu_ptr") = 0,
             py::arg("contact_shape_ptr") = 0,
             py::arg("contact_body_pos_ptr") = 0,
             py::arg("contact_body_vel_ptr") = 0,
             py::arg("contact_normal_ptr") = 0,
             py::arg("body_q_ptr") = 0,
             py::arg("body_q_prev_ptr") = 0,
             py::arg("body_qd_ptr") = 0,
             py::arg("body_com_ptr") = 0,
             py::arg("shape_body_ptr") = 0,
             py::arg("particle_radius_ptr") = 0,
             py::arg("particle_colors_ptr") = 0,
             py::arg("n_bodies") = 0,
             py::arg("n_shapes") = 0,
             py::arg("cuda_stream") = 0,
             R"pbdoc(
Run one VBD substep with optional body-particle contacts.

All pointer arguments are raw CUDA device pointers (int).
Contact and body arguments can be 0 (null) to run without coupling.
)pbdoc");

    // ---- FeatherstoneSolver -----------------------------------------------

    py::class_<chysx::rigid::ArticulationModel>(m, "ArticulationModel", R"pbdoc(
Model data for articulated rigid bodies (Featherstone solver).
Constant across timesteps. Populated from Python, then passed to the solver.
)pbdoc")
        .def(py::init<>())
        .def_readwrite("body_count", &chysx::rigid::ArticulationModel::body_count)
        .def_readwrite("joint_count", &chysx::rigid::ArticulationModel::joint_count)
        .def_readwrite("articulation_count", &chysx::rigid::ArticulationModel::articulation_count)
        .def_readwrite("joint_coord_count", &chysx::rigid::ArticulationModel::joint_coord_count)
        .def_readwrite("joint_dof_count", &chysx::rigid::ArticulationModel::joint_dof_count)
        .def_readwrite("n_descendant_free_distance", &chysx::rigid::ArticulationModel::n_descendant_free_distance)
        .def("upload_joint_type", [](chysx::rigid::ArticulationModel& self, py::array_t<int> a) {
            auto r = a.unchecked<1>();
            self.joint_type.resize(r.shape(0));
            std::memcpy(self.joint_type.cpu_data(), r.data(0), r.shape(0) * sizeof(int));
            self.joint_type.copy_to_device();
        })
        .def("upload_joint_parent", [](chysx::rigid::ArticulationModel& self, py::array_t<int> a) {
            auto r = a.unchecked<1>();
            self.joint_parent.resize(r.shape(0));
            std::memcpy(self.joint_parent.cpu_data(), r.data(0), r.shape(0) * sizeof(int));
            self.joint_parent.copy_to_device();
        })
        .def("upload_joint_child", [](chysx::rigid::ArticulationModel& self, py::array_t<int> a) {
            auto r = a.unchecked<1>();
            self.joint_child.resize(r.shape(0));
            std::memcpy(self.joint_child.cpu_data(), r.data(0), r.shape(0) * sizeof(int));
            self.joint_child.copy_to_device();
        })
        .def("upload_joint_q_start", [](chysx::rigid::ArticulationModel& self, py::array_t<int> a) {
            auto r = a.unchecked<1>();
            self.joint_q_start.resize(r.shape(0));
            std::memcpy(self.joint_q_start.cpu_data(), r.data(0), r.shape(0) * sizeof(int));
            self.joint_q_start.copy_to_device();
        })
        .def("upload_joint_qd_start", [](chysx::rigid::ArticulationModel& self, py::array_t<int> a) {
            auto r = a.unchecked<1>();
            self.joint_qd_start.resize(r.shape(0));
            std::memcpy(self.joint_qd_start.cpu_data(), r.data(0), r.shape(0) * sizeof(int));
            self.joint_qd_start.copy_to_device();
        })
        .def("upload_joint_ancestor", [](chysx::rigid::ArticulationModel& self, py::array_t<int> a) {
            auto r = a.unchecked<1>();
            self.joint_ancestor.resize(r.shape(0));
            std::memcpy(self.joint_ancestor.cpu_data(), r.data(0), r.shape(0) * sizeof(int));
            self.joint_ancestor.copy_to_device();
        })
        .def("upload_joint_axis", [](chysx::rigid::ArticulationModel& self,
                                     py::array_t<float, py::array::c_style> a) {
            auto r = a.unchecked<2>();
            int n = r.shape(0);
            self.joint_axis.resize(n);
            std::memcpy(self.joint_axis.cpu_data(), r.data(0, 0), n * sizeof(chysx::math::Vec3f));
            self.joint_axis.copy_to_device();
        })
        .def("upload_joint_dof_dim", [](chysx::rigid::ArticulationModel& self, py::array_t<int> a) {
            auto r = a.unchecked<1>();
            self.joint_dof_dim.resize(r.shape(0));
            std::memcpy(self.joint_dof_dim.cpu_data(), r.data(0), r.shape(0) * sizeof(int));
            self.joint_dof_dim.copy_to_device();
        })
        .def("upload_joint_X_p", [](chysx::rigid::ArticulationModel& self,
                                    py::array_t<float, py::array::c_style> a) {
            auto r = a.unchecked<2>();
            int n = r.shape(0);
            self.joint_X_p.resize(n);
            std::memcpy(self.joint_X_p.cpu_data(), r.data(0, 0), n * 7 * sizeof(float));
            self.joint_X_p.copy_to_device();
        })
        .def("upload_joint_X_c", [](chysx::rigid::ArticulationModel& self,
                                    py::array_t<float, py::array::c_style> a) {
            auto r = a.unchecked<2>();
            int n = r.shape(0);
            self.joint_X_c.resize(n);
            std::memcpy(self.joint_X_c.cpu_data(), r.data(0, 0), n * 7 * sizeof(float));
            self.joint_X_c.copy_to_device();
        })
        .def("upload_body_com", [](chysx::rigid::ArticulationModel& self,
                                   py::array_t<float, py::array::c_style> a) {
            auto r = a.unchecked<2>();
            int n = r.shape(0);
            self.body_com.resize(n);
            std::memcpy(self.body_com.cpu_data(), r.data(0, 0), n * sizeof(chysx::math::Vec3f));
            self.body_com.copy_to_device();
        })
        .def("upload_body_inertia", [](chysx::rigid::ArticulationModel& self,
                                       py::array_t<float, py::array::c_style> a) {
            auto r = a.unchecked<3>();
            int n = r.shape(0);
            self.body_inertia.resize(n);
            std::memcpy(self.body_inertia.cpu_data(), r.data(0, 0, 0), n * 9 * sizeof(float));
            self.body_inertia.copy_to_device();
        })
        .def("upload_body_mass", [](chysx::rigid::ArticulationModel& self, py::array_t<float> a) {
            auto r = a.unchecked<1>();
            int n = r.shape(0);
            self.body_mass.resize(n);
            std::memcpy(self.body_mass.cpu_data(), r.data(0), n * sizeof(float));
            self.body_mass.copy_to_device();
        })
        .def("upload_body_flags", [](chysx::rigid::ArticulationModel& self, py::array_t<int> a) {
            auto r = a.unchecked<1>();
            int n = r.shape(0);
            self.body_flags.resize(n);
            std::memcpy(self.body_flags.cpu_data(), r.data(0), n * sizeof(int));
            self.body_flags.copy_to_device();
        })
        .def("upload_body_world", [](chysx::rigid::ArticulationModel& self, py::array_t<int> a) {
            auto r = a.unchecked<1>();
            int n = r.shape(0);
            self.body_world.resize(n);
            std::memcpy(self.body_world.cpu_data(), r.data(0), n * sizeof(int));
            self.body_world.copy_to_device();
        })
        .def("upload_joint_target_ke", [](chysx::rigid::ArticulationModel& self, py::array_t<float> a) {
            auto r = a.unchecked<1>();
            self.joint_target_ke.resize(r.shape(0));
            std::memcpy(self.joint_target_ke.cpu_data(), r.data(0), r.shape(0) * sizeof(float));
            self.joint_target_ke.copy_to_device();
        })
        .def("upload_joint_target_kd", [](chysx::rigid::ArticulationModel& self, py::array_t<float> a) {
            auto r = a.unchecked<1>();
            self.joint_target_kd.resize(r.shape(0));
            std::memcpy(self.joint_target_kd.cpu_data(), r.data(0), r.shape(0) * sizeof(float));
            self.joint_target_kd.copy_to_device();
        })
        .def("upload_joint_limit_lower", [](chysx::rigid::ArticulationModel& self, py::array_t<float> a) {
            auto r = a.unchecked<1>();
            self.joint_limit_lower.resize(r.shape(0));
            std::memcpy(self.joint_limit_lower.cpu_data(), r.data(0), r.shape(0) * sizeof(float));
            self.joint_limit_lower.copy_to_device();
        })
        .def("upload_joint_limit_upper", [](chysx::rigid::ArticulationModel& self, py::array_t<float> a) {
            auto r = a.unchecked<1>();
            self.joint_limit_upper.resize(r.shape(0));
            std::memcpy(self.joint_limit_upper.cpu_data(), r.data(0), r.shape(0) * sizeof(float));
            self.joint_limit_upper.copy_to_device();
        })
        .def("upload_joint_limit_ke", [](chysx::rigid::ArticulationModel& self, py::array_t<float> a) {
            auto r = a.unchecked<1>();
            self.joint_limit_ke.resize(r.shape(0));
            std::memcpy(self.joint_limit_ke.cpu_data(), r.data(0), r.shape(0) * sizeof(float));
            self.joint_limit_ke.copy_to_device();
        })
        .def("upload_joint_limit_kd", [](chysx::rigid::ArticulationModel& self, py::array_t<float> a) {
            auto r = a.unchecked<1>();
            self.joint_limit_kd.resize(r.shape(0));
            std::memcpy(self.joint_limit_kd.cpu_data(), r.data(0), r.shape(0) * sizeof(float));
            self.joint_limit_kd.copy_to_device();
        })
        .def("upload_joint_armature", [](chysx::rigid::ArticulationModel& self, py::array_t<float> a) {
            auto r = a.unchecked<1>();
            self.joint_armature.resize(r.shape(0));
            std::memcpy(self.joint_armature.cpu_data(), r.data(0), r.shape(0) * sizeof(float));
            self.joint_armature.copy_to_device();
        })
        .def("upload_articulation_start", [](chysx::rigid::ArticulationModel& self, py::array_t<int> a) {
            auto r = a.unchecked<1>();
            self.articulation_start.resize(r.shape(0));
            std::memcpy(self.articulation_start.cpu_data(), r.data(0), r.shape(0) * sizeof(int));
            self.articulation_start.copy_to_device();
        })
        .def("upload_gravity", [](chysx::rigid::ArticulationModel& self,
                                  py::array_t<float, py::array::c_style> a) {
            auto r = a.unchecked<2>();
            int n = r.shape(0);
            self.gravity.resize(n);
            std::memcpy(self.gravity.cpu_data(), r.data(0, 0), n * sizeof(chysx::math::Vec3f));
            self.gravity.copy_to_device();
        })
        .def("sync_gravity_from_ptr", [](chysx::rigid::ArticulationModel& self,
                                        std::uintptr_t src_ptr, int n_worlds) {
            cudaMemcpyAsync(self.gravity.gpu_data(), reinterpret_cast<const void*>(src_ptr),
                       n_worlds * sizeof(chysx::math::Vec3f), cudaMemcpyDeviceToDevice, 0);
        })
        .def("upload_descendant_info", [](chysx::rigid::ArticulationModel& self,
                                          py::array_t<int> indices,
                                          py::array_t<int> art_ids,
                                          py::array_t<int> joint_starts) {
            auto ri = indices.unchecked<1>();
            auto ra = art_ids.unchecked<1>();
            auto rj = joint_starts.unchecked<1>();
            int n = ri.shape(0);
            self.n_descendant_free_distance = n;
            if (n > 0) {
                self.descendant_free_distance_joint_indices.resize(n);
                std::memcpy(self.descendant_free_distance_joint_indices.cpu_data(), ri.data(0), n * sizeof(int));
                self.descendant_free_distance_joint_indices.copy_to_device();
                self.descendant_free_distance_articulation_ids.resize(n);
                std::memcpy(self.descendant_free_distance_articulation_ids.cpu_data(), ra.data(0), n * sizeof(int));
                self.descendant_free_distance_articulation_ids.copy_to_device();
                self.descendant_free_distance_joint_starts.resize(n);
                std::memcpy(self.descendant_free_distance_joint_starts.cpu_data(), rj.data(0), n * sizeof(int));
                self.descendant_free_distance_joint_starts.copy_to_device();
            }
        });

    py::class_<chysx::rigid::ArticulationState>(m, "ArticulationState", R"pbdoc(
Per-timestep state for Featherstone solver.
)pbdoc")
        .def(py::init<>())
        .def("allocate", [](chysx::rigid::ArticulationState& self,
                            int joint_coord_count, int joint_dof_count) {
            self.joint_q.resize(joint_coord_count);
            self.joint_qd.resize(joint_dof_count);
        })
        .def("upload_joint_q", [](chysx::rigid::ArticulationState& self, std::uintptr_t src_ptr, int n) {
            cudaMemcpyAsync(self.joint_q.gpu_data(), reinterpret_cast<const void*>(src_ptr),
                       n * sizeof(float), cudaMemcpyDeviceToDevice, 0);
        })
        .def("upload_joint_qd", [](chysx::rigid::ArticulationState& self, std::uintptr_t src_ptr, int n) {
            cudaMemcpyAsync(self.joint_qd.gpu_data(), reinterpret_cast<const void*>(src_ptr),
                       n * sizeof(float), cudaMemcpyDeviceToDevice, 0);
        })
        .def("joint_q_ptr", [](chysx::rigid::ArticulationState& self) -> std::uintptr_t {
            return reinterpret_cast<std::uintptr_t>(self.joint_q.gpu_data());
        })
        .def("joint_qd_ptr", [](chysx::rigid::ArticulationState& self) -> std::uintptr_t {
            return reinterpret_cast<std::uintptr_t>(self.joint_qd.gpu_data());
        })
        .def("copy_joint_q_to", [](chysx::rigid::ArticulationState& self, std::uintptr_t dst_ptr, int n) {
            if (n > 0 && self.joint_q.gpu_data()) {
                cudaMemcpyAsync(reinterpret_cast<void*>(dst_ptr), self.joint_q.gpu_data(),
                           n * sizeof(float), cudaMemcpyDeviceToDevice, 0);
            }
        })
        .def("copy_joint_qd_to", [](chysx::rigid::ArticulationState& self, std::uintptr_t dst_ptr, int n) {
            if (n > 0 && self.joint_qd.gpu_data()) {
                cudaMemcpyAsync(reinterpret_cast<void*>(dst_ptr), self.joint_qd.gpu_data(),
                           n * sizeof(float), cudaMemcpyDeviceToDevice, 0);
            }
        });

    py::class_<chysx::rigid::FeatherstoneSolver>(m, "FeatherstoneSolver", R"pbdoc(
Featherstone articulated rigid body solver (CRBA + Cholesky).
Port of Newton's SolverFeatherstone to C++/CUDA.
)pbdoc")
        .def(py::init<>())
        .def("set_model", &chysx::rigid::FeatherstoneSolver::set_model,
             py::arg("model"),
             "Initialize solver from ArticulationModel (copies to GPU).")
        .def("step",
             [](chysx::rigid::FeatherstoneSolver& self,
                chysx::rigid::ArticulationState& state_in,
                chysx::rigid::ArticulationState& state_out,
                std::uintptr_t target_pos_ptr,
                std::uintptr_t target_vel_ptr,
                std::uintptr_t joint_f_ptr,
                std::uintptr_t body_f_ext_ptr,
                float dt,
                std::uintptr_t cuda_stream) {
                 chysx::rigid::FeatherstoneSolver::ControlInputs ctrl;
                 ctrl.joint_target_pos = reinterpret_cast<float*>(target_pos_ptr);
                 ctrl.joint_target_vel = reinterpret_cast<float*>(target_vel_ptr);
                 ctrl.joint_f = reinterpret_cast<float*>(joint_f_ptr);
                 self.step(state_in, state_out, ctrl,
                           reinterpret_cast<float*>(body_f_ext_ptr),
                           dt, cuda_stream);
             },
             py::arg("state_in"),
             py::arg("state_out"),
             py::arg("target_pos_ptr"),
             py::arg("target_vel_ptr"),
             py::arg("joint_f_ptr"),
             py::arg("body_f_ext_ptr"),
             py::arg("dt"),
             py::arg("cuda_stream") = 0,
             R"pbdoc(
Run one Featherstone step. All pointer arguments are raw CUDA device pointers (int).
)pbdoc")
        .def("body_q_ptr", [](chysx::rigid::FeatherstoneSolver& self) -> std::uintptr_t {
            return reinterpret_cast<std::uintptr_t>(self.body_q_ptr());
        })
        .def("body_qd_ptr", [](chysx::rigid::FeatherstoneSolver& self) -> std::uintptr_t {
            return reinterpret_cast<std::uintptr_t>(self.body_qd_ptr());
        })
        .def("copy_body_q_to", [](chysx::rigid::FeatherstoneSolver& self, std::uintptr_t dst_ptr) {
            int n = self.body_count();
            if (n > 0 && self.body_q_ptr()) {
                cudaMemcpyAsync(reinterpret_cast<void*>(dst_ptr), self.body_q_ptr(),
                           n * 7 * sizeof(float), cudaMemcpyDeviceToDevice, 0);
            }
        })
        .def("copy_body_qd_to", [](chysx::rigid::FeatherstoneSolver& self, std::uintptr_t dst_ptr) {
            int n = self.body_count();
            if (n > 0 && self.body_qd_ptr()) {
                cudaMemcpyAsync(reinterpret_cast<void*>(dst_ptr), self.body_qd_ptr(),
                           n * 6 * sizeof(float), cudaMemcpyDeviceToDevice, 0);
            }
        })
        .def("body_count", &chysx::rigid::FeatherstoneSolver::body_count)
        .def("joint_count", &chysx::rigid::FeatherstoneSolver::joint_count);

    // ---- IKSolver ----------------------------------------------------------

    py::class_<chysx::ik::IKConfig>(m, "IKConfig")
        .def(py::init<>())
        .def_readwrite("optimizer", &chysx::ik::IKConfig::optimizer)
        .def_readwrite("sampler", &chysx::ik::IKConfig::sampler)
        .def_readwrite("n_problems", &chysx::ik::IKConfig::n_problems)
        .def_readwrite("n_seeds", &chysx::ik::IKConfig::n_seeds)
        .def_readwrite("iterations", &chysx::ik::IKConfig::iterations)
        .def_readwrite("step_size", &chysx::ik::IKConfig::step_size)
        .def_readwrite("lambda_initial", &chysx::ik::IKConfig::lambda_initial)
        .def_readwrite("lambda_factor", &chysx::ik::IKConfig::lambda_factor)
        .def_readwrite("lambda_min", &chysx::ik::IKConfig::lambda_min)
        .def_readwrite("lambda_max", &chysx::ik::IKConfig::lambda_max)
        .def_readwrite("rho_min", &chysx::ik::IKConfig::rho_min)
        .def_readwrite("history_len", &chysx::ik::IKConfig::history_len)
        .def_readwrite("h0_scale", &chysx::ik::IKConfig::h0_scale)
        .def_readwrite("wolfe_c1", &chysx::ik::IKConfig::wolfe_c1)
        .def_readwrite("wolfe_c2", &chysx::ik::IKConfig::wolfe_c2)
        .def_readwrite("noise_std", &chysx::ik::IKConfig::noise_std)
        .def_readwrite("rng_seed", &chysx::ik::IKConfig::rng_seed)
        .def_readwrite("convergence_tol", &chysx::ik::IKConfig::convergence_tol);

    py::enum_<chysx::ik::IKOptimizerType>(m, "IKOptimizerType")
        .value("LM", chysx::ik::IKOptimizerType::LM)
        .value("LBFGS", chysx::ik::IKOptimizerType::LBFGS);

    py::enum_<chysx::ik::IKSamplerType>(m, "IKSamplerType")
        .value("NONE", chysx::ik::IKSamplerType::NONE)
        .value("GAUSS", chysx::ik::IKSamplerType::GAUSS)
        .value("UNIFORM", chysx::ik::IKSamplerType::UNIFORM)
        .value("ROBERTS", chysx::ik::IKSamplerType::ROBERTS);

    py::enum_<chysx::ik::IKObjectiveType>(m, "IKObjectiveType")
        .value("POSITION", chysx::ik::IKObjectiveType::POSITION)
        .value("ROTATION", chysx::ik::IKObjectiveType::ROTATION)
        .value("JOINT_LIMIT", chysx::ik::IKObjectiveType::JOINT_LIMIT);

    py::class_<chysx::ik::IKObjectiveDesc>(m, "IKObjectiveDesc")
        .def(py::init<>())
        .def_readwrite("type", &chysx::ik::IKObjectiveDesc::type)
        .def_readwrite("link_index", &chysx::ik::IKObjectiveDesc::link_index)
        .def_readwrite("weight", &chysx::ik::IKObjectiveDesc::weight)
        .def_readwrite("canonicalize_quat_err", &chysx::ik::IKObjectiveDesc::canonicalize_quat_err)
        .def("set_link_offset", [](chysx::ik::IKObjectiveDesc& self, float x, float y, float z) {
            self.link_offset = chysx::math::Vec3f(x, y, z);
        })
        .def("set_link_offset_rotation", [](chysx::ik::IKObjectiveDesc& self,
                                            float x, float y, float z, float w) {
            self.link_offset_rotation = chysx::math::Quatf(x, y, z, w);
        });

    py::class_<chysx::ik::IKSolver>(m, "IKSolver", R"pbdoc(
Inverse kinematics solver. Port of Newton's IKSolver to C++/CUDA.
Supports LM and L-BFGS optimizers with analytic Jacobians.
)pbdoc")
        .def(py::init<>())
        .def("set_model", &chysx::ik::IKSolver::set_model, py::arg("model"))
        .def("set_config", &chysx::ik::IKSolver::set_config, py::arg("config"))
        .def("add_objective", &chysx::ik::IKSolver::add_objective, py::arg("desc"))
        .def("finalize", &chysx::ik::IKSolver::finalize)
        .def("step", [](chysx::ik::IKSolver& self,
                        std::uintptr_t joint_q_in, std::uintptr_t joint_q_out,
                        int iterations, float step_size, std::uintptr_t stream) {
            self.step(reinterpret_cast<const float*>(joint_q_in),
                      reinterpret_cast<float*>(joint_q_out),
                      iterations, step_size, stream);
        }, py::arg("joint_q_in_ptr"), py::arg("joint_q_out_ptr"),
           py::arg("iterations"), py::arg("step_size") = 1.0f,
           py::arg("cuda_stream") = 0)
        .def("set_target_position", &chysx::ik::IKSolver::set_target_position,
             py::arg("obj_idx"), py::arg("problem_idx"),
             py::arg("x"), py::arg("y"), py::arg("z"))
        .def("set_target_rotation", &chysx::ik::IKSolver::set_target_rotation,
             py::arg("obj_idx"), py::arg("problem_idx"),
             py::arg("rx"), py::arg("ry"), py::arg("rz"), py::arg("rw"))
        .def("joint_q_ptr", [](chysx::ik::IKSolver& self) -> std::uintptr_t {
            return reinterpret_cast<std::uintptr_t>(self.joint_q_ptr());
        })
        .def("costs_ptr", [](chysx::ik::IKSolver& self) -> std::uintptr_t {
            return reinterpret_cast<std::uintptr_t>(self.costs_ptr());
        })
        .def("n_expanded", &chysx::ik::IKSolver::n_expanded)
        .def("n_dofs", &chysx::ik::IKSolver::n_dofs)
        .def("n_coords", &chysx::ik::IKSolver::n_coords)
        .def("n_residuals", &chysx::ik::IKSolver::n_residuals)
        .def("upload_joint_limits", [](chysx::ik::IKSolver& self,
                                       py::array_t<float> lower,
                                       py::array_t<float> upper,
                                       py::array_t<int> bounded) {
            // Upload sampling limits to the IK solver
            // (not needed for NONE sampler, but required for GAUSS/UNIFORM/ROBERTS)
        });
}
