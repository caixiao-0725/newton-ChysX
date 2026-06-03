"""Quick test of SolverChysXCoupled with Franka + duck scene."""

import sys
import numpy as np
import warp as wp

wp.init()

print("Importing newton...", flush=True)
import newton
import newton.utils
from newton import ModelBuilder, eval_fk
from newton.solvers import SolverChysXCoupled, SolverFeatherstone

print("Setting up scene...", flush=True)
scene = ModelBuilder(gravity=-9.81)

franka = ModelBuilder()
asset_path = newton.utils.download_asset("franka_emika_panda")
print(f"Asset downloaded: {asset_path}", flush=True)
franka.add_urdf(
    str(asset_path / "urdf" / "fr3_franka_hand.urdf"),
    xform=wp.transform((-0.5, -0.5, -0.1), wp.quat_identity()),
    floating=False,
    scale=1.0,
    enable_self_collisions=False,
    collapse_fixed_joints=True,
    force_show_colliders=False,
)
franka.joint_q[:6] = [0.0, 0.0, 0.0, -1.59695, 0.0, 2.5307]
scene.add_world(franka)

scene.add_shape_box(
    -1,
    wp.transform(wp.vec3(0.0, -0.5, 0.1), wp.quat_identity()),
    hx=0.4,
    hy=0.4,
    hz=0.1,
)

print("Loading duck...", flush=True)
from pxr import Usd

duck_path = newton.utils.download_asset("manipulation_objects/rubber_duck")
print(f"Duck asset: {duck_path}", flush=True)
usd_stage = Usd.Stage.Open(str(duck_path / "model.usda"))
prim = usd_stage.GetPrimAtPath("/root/Model/TetMesh")
tetmesh = newton.TetMesh.create_from_usd(prim)
scene.add_soft_mesh(
    pos=wp.vec3(0.0, -0.5, 0.23),
    rot=wp.quat_identity(),
    scale=1.0,
    vel=wp.vec3(0.0, 0.0, 0.0),
    mesh=tetmesh,
    density=100.0,
    k_mu=1e6,
    k_lambda=1e6,
    k_damp=1e-6,
    particle_radius=0.005,
)

scene.color()
scene.add_ground_plane()
model = scene.finalize(requires_grad=False)

model.soft_contact_ke = 2e6
model.soft_contact_kd = 1e-7
model.soft_contact_mu = 0.5
model.shape_material_ke.fill_(2e6)
model.shape_material_kd.fill_(1e-7)
model.shape_material_mu.fill_(1.5)

print(
    f"bodies: {model.body_count}, shapes: {model.shape_count}, "
    f"particles: {model.particle_count}",
    flush=True,
)

state_0 = model.state()
state_1 = model.state()

print("Creating Featherstone solver...", flush=True)
robot_solver = SolverFeatherstone(model, update_mass_matrix_interval=10)

print("Creating ChysX coupled solver...", flush=True)
soft_solver = SolverChysXCoupled(model, iterations=5, friction_epsilon=1.0)
print(f"VBD colors: {soft_solver._sim.num_colors()}", flush=True)

collision_pipeline = newton.CollisionPipeline(model, soft_contact_margin=0.01)
contacts = collision_pipeline.contacts()

print("Running eval_fk...", flush=True)
eval_fk(model, model.joint_q, model.joint_qd, state_0)

dt = 1.0 / 60.0 / 10.0
gravity_zero = wp.zeros(1, dtype=wp.vec3)
gravity_earth = wp.array(wp.vec3(0.0, 0.0, -9.81), dtype=wp.vec3)
control = model.control()
target_qd = wp.zeros_like(state_0.joint_qd)

N_SUBSTEPS = 5
print(f"Running {N_SUBSTEPS} substeps...", flush=True)
for i in range(N_SUBSTEPS):
    state_0.clear_forces()
    state_1.clear_forces()

    pc = model.particle_count
    model.particle_count = 0
    model.gravity.assign(gravity_zero)
    model.shape_contact_pair_count = 0
    state_0.joint_qd.assign(target_qd)
    robot_solver.step(state_0, state_1, control, None, dt)
    model.particle_count = pc
    model.gravity.assign(gravity_earth)

    collision_pipeline.collide(state_0, contacts)
    cnt = contacts.soft_contact_count.numpy()[0]

    soft_solver.step(state_0, state_1, control, contacts, dt)
    state_0, state_1 = state_1, state_0

    q = state_0.particle_q.numpy().reshape(-1, 3)
    print(
        f"  substep {i}: contacts={cnt} z_min={q[:, 2].min():.4f} z_max={q[:, 2].max():.4f}",
        flush=True,
    )

print("DONE", flush=True)
