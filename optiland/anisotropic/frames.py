"""Interface Frames

The interface frame of a ray at a surface and the rotation of 6 x 6
constitutive matrices into it.

Conventions (equation register E-11 of the project notes; Lekner, J. Phys.:
Condens. Matter 3, 6121 (1991)):

* ẑ is the unit surface normal n̂, from the incident medium A into the exit
  medium B.
* x̂ is along the tangential part of the real incident wave vector, so the
  tangential wave vector of every child is K x̂ (K complex for an absorbing A).
  At normal incidence x̂ is a reference vector projected on the surface.
* ŷ = ẑ × x̂. The rotation R has the rows x̂, ŷ, ẑ: v_frame = R v_global, and a
  tensor rotates as X_frame = R X_global Rᵀ.
"""

from __future__ import annotations

from typing import Any

import numpy as np

import optiland.backend as be
from optiland.materials.anisotropic import _complex

Array = Any

__all__ = ["interface_frame", "rotate_constitutive", "to_global"]

#: A tangential wave vector below this (times max(1, |k|)) is normal incidence.
NORMAL_INCIDENCE_RTOL = 1e-12


def _real_vectors(x: Any) -> Array:
    """Return x as real vectors of the active backend, shape (N, 3)."""
    if hasattr(x, "detach") and be.get_backend() == "numpy":
        x = x.detach().cpu().numpy()
    v = be.asarray(x)
    if len(v.shape) == 1:
        v = be.reshape(v, (1, 3))
    return v


def _complex_vectors(x: Any) -> Array:
    """Return x as complex vectors of the active backend, shape (N, 3)."""
    v = _complex(x)
    if len(v.shape) == 1:
        v = be.reshape(v, (1, 3))
    return v


def _sum(x: Array, axis: int) -> Array:
    """Return ``be.sum`` along an axis, typed as an array."""
    return be.sum(x, axis=axis)


def _dot(a: Array, b: Array) -> Array:
    """Return the dot product of vectors on the last axis."""
    return _sum(a * b, axis=-1)


def _unit(v: Array) -> Array:
    """Return v / |v| on the last axis (a zero vector stays zero)."""
    norm = be.sqrt(_dot(v, v))
    safe = be.where(norm > 0, norm, be.ones_like(norm))
    return v / safe[..., None]


def interface_frame(
    normal: Any,
    k_in: Any,
    x_ref: Any = None,
) -> tuple[Array, Array]:
    """Return the interface frame of each ray and its tangential wave number.

    Args:
        normal: Unit surface normals from medium A into medium B, shape (N, 3)
            or (3,). They are normalized here.
        k_in: Incident wave vectors in units of k0, shape (N, 3) or (3,).
            Complex values are allowed (an absorbing incident medium).
        x_ref: Reference vectors for x̂ at normal incidence, shape (N, 3) or
            (3,). If None, the global x̂ is used (the global ŷ where the normal
            is within 25.8° of x̂). Only the part tangential to the surface
            is used.

    Returns:
        A tuple (R, K). R is the real rotation with the rows x̂, ŷ, ẑ, shape
        (N, 3, 3). K = k_in · x̂ is the complex tangential wave number, shape
        (N,).
    """
    n_hat = _real_vectors(normal)
    k = _complex_vectors(k_in)
    n_rays = max(n_hat.shape[0], k.shape[0])
    n_hat = _unit(be.broadcast_to(n_hat, (n_rays, 3)))
    k = be.broadcast_to(k, (n_rays, 3))

    k_real = be.real(k)
    k_t = k_real - _dot(k_real, n_hat)[:, None] * n_hat
    k_t_norm = be.sqrt(_dot(k_t, k_t))
    k_norm = be.sqrt(be.real(_dot(k, be.real(k) - 1j * be.imag(k))))
    oblique = k_t_norm > NORMAL_INCIDENCE_RTOL * be.maximum(
        k_norm, be.ones_like(k_norm)
    )

    if x_ref is None:
        near_x = be.abs(n_hat[:, 0]) > 0.9
        x_axis = be.zeros_like(n_hat) + be.asarray(np.array([1.0, 0.0, 0.0]))
        y_axis = be.zeros_like(n_hat) + be.asarray(np.array([0.0, 1.0, 0.0]))
        ref = be.where(near_x[:, None], y_axis, x_axis)
    else:
        ref = be.broadcast_to(_real_vectors(x_ref), (n_rays, 3))
    ref_t = ref - _dot(ref, n_hat)[:, None] * n_hat

    x_hat = be.where(oblique[:, None], _unit(k_t), _unit(ref_t))
    y_hat = be.cross(n_hat, x_hat)
    rotation = be.stack([x_hat, y_hat, n_hat], axis=-2)
    tangential = _dot(k, be.to_complex(x_hat))
    return rotation, tangential


def _block_rotation(rotation: Array) -> Array:
    """Return diag(R, R) of each rotation, shape (N, 6, 6), real."""
    zeros = be.zeros_like(rotation)
    top = be.concatenate([rotation, zeros], axis=-1)
    bottom = be.concatenate([zeros, rotation], axis=-1)
    return be.concatenate([top, bottom], axis=-2)


def rotate_constitutive(matrix: Any, rotation: Array) -> Array:
    """Rotate 6 x 6 constitutive matrices into the interface frame.

    Each 3 x 3 block X of [[ε, ξ], [ζ, μ]] becomes R X Rᵀ.

    Args:
        matrix: Constitutive matrices in the global frame, shape (N, 6, 6) or
            (6, 6), complex.
        rotation: Frame rotations from ``interface_frame``, shape (N, 3, 3).

    Returns:
        The matrices in the frame, shape (N, 6, 6), complex.
    """
    m = _complex(matrix)
    if len(m.shape) == 2:
        m = be.reshape(m, (1, 6, 6))
    r6 = be.to_complex(_block_rotation(rotation))
    r6_t = be.transpose(r6, (0, 2, 1))
    return be.matmul(be.matmul(r6, m), r6_t)


def to_global(vectors: Array, rotation: Array) -> Array:
    """Rotate frame vectors back to the global frame, v_global = Rᵀ v_frame.

    Args:
        vectors: Vectors in the frame, shape (N, M, 3) (M vectors per ray),
            real or complex.
        rotation: Frame rotations, shape (N, 3, 3).

    Returns:
        The global vectors, shape (N, M, 3), of the dtype of ``vectors``.
    """
    r = rotation
    if "complex" in str(vectors.dtype):
        r = be.to_complex(r)
    return be.matmul(vectors, r)
