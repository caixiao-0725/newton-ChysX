"""Test SolverChysXCoupled with a soft grid + ground plane (no external assets)."""

import sys
import numpy as np
import warp as wp

wp.init()

print("Importing...", flush=True)
import newton
from newton import ModelBuilder, eval_fk
from newton.solvers import SolverChysXCoupled

print("Building scene...", flush=True)
scene = ModelBuilder(gravity=-9.81)

# Soft grid (tet mesh) above ground
scene.add_soft_grid(
    pos=wp.vec3(0.0, 0.0, 0.3),
    rot=wp.quat_identity(),
    vel=wp.vec3(0.0, 0.0, 0.0),
    dim_x=5, dim_y=5, dim_z=5,
    cell_x=0.05, cell_y=0.05, cell_z=0.05,
    density=100.0,
    k_mu=1e5,
    k_lambda=1e5,
    k_damp=1e-5,
    particle_radius=0.005,
)

# Ground plane (static shape)
scene.add_ground_plane()
scene.color()
model = scene.finalize(requires_grad=False)

model.soft_contact_ke = 1e5
model.soft_contact_kd = 1e-5
model.soft_contact_mu = 0.5
model.shape_material_ke.fill_(1e5)
model.shape_material_kd.fill_(1e-5)
model.shape_material_mu.fill_(0.5)

print(
    f"bodies: {model.body_count}, shapes: {model.shape_count}, "
    f"particles: {model.particle_count}, tets: {model.tet_count}",
    flush=True,
)

state_0 = model.state()
state_1 = model.state()

print("Creating solver...", flush=True)
soft_solver = SolverChysXCoupled(model, iterations=5, friction_epsilon=1.0)
print(f"VBD colors: {soft_solver._sim.num_colors()}", flush=True)

collision_pipeline = newton.CollisionPipeline(model, soft_contact_margin=0.01)
contacts = collision_pipeline.contacts()

dt = 1.0 / 60.0 / 5.0
control = model.control()

N_SUBSTEPS = 30
print(f"Running {N_SUBSTEPS} substeps (dt={dt:.6f})...", flush=True)
for i in range(N_SUBSTEPS):
    state_0.clear_forces()
    state_1.clear_forces()

    collision_pipeline.collide(state_0, contacts)
    cnt = contacts.soft_contact_count.numpy()[0]

    soft_solver.step(state_0, state_1, control, contacts, dt)
    state_0, state_1 = state_1, state_0

    if i % 5 == 0 or i == N_SUBSTEPS - 1:
        q = state_0.particle_q.numpy().reshape(-1, 3)
        v = state_0.particle_qd.numpy().reshape(-1, 3)
        v_max = np.abs(v).max()
        print(
            f"  substep {i:3d}: contacts={cnt:3d} "
            f"z_min={q[:, 2].min():.4f} z_max={q[:, 2].max():.4f} "
            f"v_max={v_max:.4f}",
            flush=True,
        )

print("DONE", flush=True)
