// SPDX-License-Identifier: Apache-2.0
//
// pybind11 bindings for the _chysx_native module.

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

#include <stdexcept>

#include "cloth/cloth_material.h"
#include "cloth/cloth_simulator.h"
#include "collision/static_contact.h"
#include "math/vec.cuh"

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
}
