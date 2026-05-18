# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Camera helpers for Z-up ViewerGL examples."""

from __future__ import annotations

from typing import Sequence

import numpy as np
import warp as wp


def frame_z_up_camera_viewer(
    viewer,
    bbox_min: Sequence[float] | np.ndarray,
    bbox_max: Sequence[float] | np.ndarray,
    *,
    dist_scale: float = 1.15,
    dist_floor: float = 1.25,
    eye_offset: tuple[float, float, float] = (0.25, -1.05, 0.85),
    target: Sequence[float] | np.ndarray | None = None,
) -> None:
    """Frame ``bbox`` for :class:`newton.viewer.ViewerGL` with Z-up convention.

    With ``up_axis = Z``, ``pitch = yaw = 0`` looks along **+X**, not along
    −Z.  Guessed fixed ``pos`` + Euler angles often leave the cloth outside
    the opening frustum.  We place the eye in the (+X, −Y, +Z) quadrant
    relative to the bbox diagonal and call :meth:`Camera.look_at` at the
    bbox centre (or ``target``).

    Args:
        viewer: Typically ``ViewerGL``; ``ViewerNull`` has no ``camera`` —
            falls back to :meth:`~newton.viewer.ViewerBase.set_camera` with
            angles derived from ``eye → target``.
        bbox_min: Axis-aligned bbox minimum (world [m]).
        bbox_max: Axis-aligned bbox maximum (world [m]).
        dist_scale: Eye distance as ``max(dist_scale * extent, dist_floor)``.
        dist_floor: Minimum eye distance [m].
        eye_offset: Scaled offset added to ``centre`` for eye position.
        target: Optional look-at point; default bbox centre.
    """
    bmin = np.asarray(bbox_min, dtype=np.float64).reshape(3)
    bmax = np.asarray(bbox_max, dtype=np.float64).reshape(3)
    centre = 0.5 * (bmin + bmax)
    extent = float(np.linalg.norm(bmax - bmin))
    dist = max(dist_scale * extent, dist_floor)
    ox, oy, oz = eye_offset
    eye = (
        float(centre[0] + ox * dist),
        float(centre[1] + oy * dist),
        float(centre[2] + oz * dist),
    )
    if target is None:
        tgt = (float(centre[0]), float(centre[1]), float(centre[2]))
    else:
        t = np.asarray(target, dtype=np.float64).reshape(3)
        tgt = (float(t[0]), float(t[1]), float(t[2]))

    cam = getattr(viewer, "camera", None)
    if cam is not None:
        cam.pos = cam._as_vec3(eye)
        cam.look_at(cam._as_vec3(tgt))
        return

    direction = np.asarray(tgt, dtype=np.float64) - np.asarray(eye, dtype=np.float64)
    dn = float(np.linalg.norm(direction))
    if dn < 1e-8:
        direction = np.array([-0.25, 1.05, -0.85], dtype=np.float64)
        dn = float(np.linalg.norm(direction))
    direction /= dn
    pitch = float(np.rad2deg(np.arcsin(np.clip(direction[2], -1.0, 1.0))))
    yaw = float(np.rad2deg(np.arctan2(direction[1], direction[0])))
    viewer.set_camera(wp.vec3(*eye), pitch, yaw)
