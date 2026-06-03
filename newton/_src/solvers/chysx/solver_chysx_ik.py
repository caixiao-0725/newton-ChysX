# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""ChysX-native IK solver wrapper.

Mirrors Newton's ``IKSolver`` API but runs entirely in C++/CUDA via the
ChysX ``_chysx_native.IKSolver`` backend.  Only ANALYTIC Jacobian mode
is supported; autodiff is not available.
"""

from __future__ import annotations

import numpy as np
import warp as wp


class SolverChysXIK:
    """Python wrapper around the ChysX C++/CUDA IK solver.

    Args:
        model: Newton articulation model.
        n_problems: Number of IK problems to solve in parallel.
        objectives: List of objective descriptor dicts. Each has keys:
            ``type`` (``"position"`` / ``"rotation"`` / ``"joint_limit"``),
            ``link_index``, ``link_offset``, ``link_offset_rotation``,
            ``weight``, ``target_positions``, ``target_rotations``,
            ``canonicalize_quat_err``.
        optimizer: ``"lm"`` or ``"lbfgs"`` (default ``"lm"``).
        n_seeds: Number of initial guesses per problem.
        sampler: ``"none"`` / ``"gauss"`` / ``"uniform"`` / ``"roberts"``.
        lambda_initial: LM initial damping.
        iterations: Default iteration count.
    """

    def __init__(
        self,
        model,
        n_problems: int = 1,
        objectives: list | None = None,
        *,
        optimizer: str = "lm",
        n_seeds: int = 1,
        sampler: str = "none",
        lambda_initial: float = 0.1,
        iterations: int = 12,
        step_size: float = 1.0,
        convergence_tol: float = 0.0,
        **kwargs,
    ):
        import chysx

        self._model = model
        self._n_problems = n_problems
        self._n_coords = model.joint_coord_count
        self._n_dofs = model.joint_dof_count

        # Build C++ ArticulationModel if not already done
        self._art_model = self._build_articulation_model(model)

        # Config
        self._solver = chysx.IKSolver()
        cfg = chysx.IKConfig()
        cfg.optimizer = (
            chysx.IKOptimizerType.LBFGS if optimizer.lower() == "lbfgs"
            else chysx.IKOptimizerType.LM
        )
        cfg.sampler = {
            "none": chysx.IKSamplerType.NONE,
            "gauss": chysx.IKSamplerType.GAUSS,
            "uniform": chysx.IKSamplerType.UNIFORM,
            "roberts": chysx.IKSamplerType.ROBERTS,
        }.get(sampler.lower(), chysx.IKSamplerType.NONE)
        cfg.n_problems = n_problems
        cfg.n_seeds = n_seeds
        cfg.iterations = iterations
        cfg.step_size = step_size
        cfg.lambda_initial = lambda_initial
        cfg.convergence_tol = convergence_tol

        for k, v in kwargs.items():
            if hasattr(cfg, k):
                setattr(cfg, k, v)

        self._solver.set_model(self._art_model)
        self._solver.set_config(cfg)

        # Add objectives
        self._obj_descs = []
        self._obj_targets = []
        if objectives:
            for obj_spec in objectives:
                self._add_objective(obj_spec)

        self._solver.finalize()

        self._iterations = iterations
        self._step_size = step_size

        # Warp buffers for joint_q I/O
        self._joint_q_in_wp = None
        self._joint_q_out_wp = None

    def _build_articulation_model(self, model):
        """Upload Newton model data to a ChysX ArticulationModel."""
        import chysx

        am = chysx.ArticulationModel()
        am.body_count = model.body_count
        am.joint_count = model.joint_count
        am.articulation_count = model.articulation_count
        am.joint_coord_count = model.joint_coord_count
        am.joint_dof_count = model.joint_dof_count

        am.upload_joint_type(model.joint_type.numpy())
        am.upload_joint_parent(model.joint_parent.numpy())
        am.upload_joint_child(model.joint_child.numpy())
        am.upload_joint_q_start(model.joint_q_start.numpy())
        am.upload_joint_qd_start(model.joint_qd_start.numpy())
        am.upload_joint_axis(model.joint_axis.numpy())

        joint_dof_dim_np = model.joint_dof_dim.numpy().flatten().astype(np.int32)
        am.upload_joint_dof_dim(joint_dof_dim_np)

        am.upload_joint_X_p(model.joint_X_p.numpy().reshape(-1, 7))
        am.upload_joint_X_c(model.joint_X_c.numpy().reshape(-1, 7))
        am.upload_body_com(model.body_com.numpy())
        am.upload_body_mass(model.body_mass.numpy())
        inertia = model.body_inertia.numpy().reshape(-1, 3, 3).astype(np.float32)
        am.upload_body_inertia(np.ascontiguousarray(inertia))
        am.upload_body_flags(np.zeros(model.body_count, dtype=np.int32))
        am.upload_body_world(np.zeros(model.body_count, dtype=np.int32))

        am.upload_joint_limit_lower(model.joint_limit_lower.numpy())
        am.upload_joint_limit_upper(model.joint_limit_upper.numpy())
        am.upload_joint_limit_ke(model.joint_limit_ke.numpy())
        am.upload_joint_limit_kd(model.joint_limit_kd.numpy())
        am.upload_joint_target_ke(model.joint_target_ke.numpy())
        am.upload_joint_target_kd(model.joint_target_kd.numpy())
        am.upload_joint_armature(model.joint_armature.numpy())

        articulation_start = model.articulation_start.numpy()
        am.articulation_count = len(articulation_start) - 1
        am.upload_articulation_start(articulation_start)

        return am

    def _add_objective(self, spec):
        """Add a single objective from a spec dict or Newton objective object."""
        import chysx

        desc = chysx.IKObjectiveDesc()

        obj = spec
        cls_name = type(obj).__name__

        if hasattr(obj, "__class__"):
            if "Position" in cls_name and hasattr(obj, "link_index"):
                desc.type = chysx.IKObjectiveType.POSITION
                desc.link_index = obj.link_index
                if hasattr(obj, "link_offset"):
                    lo = obj.link_offset
                    desc.set_link_offset(float(lo[0]), float(lo[1]), float(lo[2]))
                desc.weight = getattr(obj, "weight", 1.0)
                self._obj_descs.append(desc)
                self._solver.add_objective(desc)
                idx = len(self._obj_descs) - 1
                if hasattr(obj, "_target_positions") and obj._target_positions is not None:
                    tp = obj._target_positions.numpy()
                    for pi in range(len(tp)):
                        self._solver.set_target_position(idx, pi, float(tp[pi][0]), float(tp[pi][1]), float(tp[pi][2]))
                self._obj_targets.append(("position", obj))
            elif "Rotation" in cls_name and hasattr(obj, "link_index"):
                desc.type = chysx.IKObjectiveType.ROTATION
                desc.link_index = obj.link_index
                if hasattr(obj, "link_offset_rotation"):
                    lor = obj.link_offset_rotation
                    desc.set_link_offset_rotation(float(lor[0]), float(lor[1]), float(lor[2]), float(lor[3]))
                desc.weight = getattr(obj, "weight", 1.0)
                desc.canonicalize_quat_err = getattr(obj, "canonicalize_quat_err", True)
                self._obj_descs.append(desc)
                self._solver.add_objective(desc)
                idx = len(self._obj_descs) - 1
                if hasattr(obj, "_target_rotations") and obj._target_rotations is not None:
                    tr = obj._target_rotations.numpy()
                    for pi in range(len(tr)):
                        self._solver.set_target_rotation(idx, pi, float(tr[pi][0]), float(tr[pi][1]), float(tr[pi][2]), float(tr[pi][3]))
                self._obj_targets.append(("rotation", obj))
            elif "Limit" in cls_name:
                desc.type = chysx.IKObjectiveType.JOINT_LIMIT
                desc.weight = getattr(obj, "weight", 10.0)
                self._obj_descs.append(desc)
                self._solver.add_objective(desc)
                self._obj_targets.append(("joint_limit", obj))
            else:
                raise ValueError(f"Unknown objective type: {spec}")
        else:
            raise ValueError(f"Unknown objective type: {spec}")

    def set_target_position(self, obj_idx: int, problem_idx: int, pos):
        """Update the target position for a position objective."""
        if isinstance(pos, wp.vec3):
            self._solver.set_target_position(obj_idx, problem_idx, float(pos[0]), float(pos[1]), float(pos[2]))
        else:
            self._solver.set_target_position(obj_idx, problem_idx, float(pos[0]), float(pos[1]), float(pos[2]))

    def set_target_rotation(self, obj_idx: int, problem_idx: int, rot):
        """Update the target rotation for a rotation objective."""
        if isinstance(rot, wp.vec4):
            self._solver.set_target_rotation(obj_idx, problem_idx, float(rot[0]), float(rot[1]), float(rot[2]), float(rot[3]))
        else:
            self._solver.set_target_rotation(obj_idx, problem_idx, float(rot[0]), float(rot[1]), float(rot[2]), float(rot[3]))

    def step(
        self,
        joint_q_in: wp.array,
        joint_q_out: wp.array,
        iterations: int | None = None,
        step_size: float | None = None,
    ):
        """Run IK solver.

        Args:
            joint_q_in: Input joint coordinates [n_problems, n_coords] or [1, n_coords].
            joint_q_out: Output buffer, same shape. May alias joint_q_in.
            iterations: Override iteration count.
            step_size: Override step size.
        """
        iters = iterations if iterations is not None else self._iterations
        ss = step_size if step_size is not None else self._step_size

        in_ptr = joint_q_in.ptr
        out_ptr = joint_q_out.ptr

        self._solver.step(in_ptr, out_ptr, iters, ss)
