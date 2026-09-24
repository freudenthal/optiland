"""Mueller and Stokes outputs of polarized rays

This module computes Stokes vectors, Mueller matrices, diattenuation and
retardance from the polarization ray-tracing (PRT) matrix of each ray.

Conventions:

* Time dependence exp(-iωt).
* The PRT matrix P maps the 3D field at the entrance to the 3D field at the
  exit, P k_in = k_out (Yun, Crabtree and Chipman, Appl. Opt. 50, 2855
  (2011)). ``PolarizedRays.p`` holds it; the power factor of the refractions
  is ``PolarizedRays.flux_factor``.
* A Jones vector (Ex, Ey) is given in a right-handed frame (x, y, k). The
  Stokes vector is S0 = |Ex|² + |Ey|², S1 = |Ex|² - |Ey|²,
  S2 = 2 Re(Ex Ey*), S3 = 2 Im(Ex Ey*) = Im((E x E*) · k). S3 > 0 is right
  circular polarization as seen by an observer who looks toward the source
  (the "RCP" state of :func:`optiland.rays.create_polarization`).
* The input frame of a ray is the frame of its input field
  (``PolarizedRays.get_input_basis``): x = s, y = p. The output frame is a
  detector frame: x is the projection of a reference vector ``x_ref`` onto the
  plane normal to k_out, and y = k_out x x.
* The Mueller matrix of a Jones matrix J is M = A (J ⊗ J*) A⁻¹, with A the
  matrix that gives the Stokes vector above from E ⊗ E*.

References:

* G. Yun, K. Crabtree, R. A. Chipman, "Three-dimensional polarization
  ray-tracing calculus I: definition and diattenuation," Appl. Opt. 50,
  2855-2865 (2011).
* G. Yun, S. C. McClain, R. A. Chipman, "Three-dimensional polarization
  ray-tracing calculus II: retardance," Appl. Opt. 50, 2866-2874 (2011).
* S.-Y. Lu, R. A. Chipman, "Interpretation of Mueller matrices based on polar
  decomposition," J. Opt. Soc. Am. A 13, 1106-1113 (1996).
* J. J. Gil, E. Bernabeu, "Depolarization and polarization indices of an
  optical system," Opt. Acta 33, 185-189 (1986).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

import optiland.backend as be

if TYPE_CHECKING:
    from optiland.rays import PolarizationState, PolarizedRays

Array = Any

__all__ = [
    "depolarization_index",
    "detector_frame",
    "diattenuation",
    "diattenuation_from_mueller",
    "jones_from_prt",
    "jones_to_mueller",
    "mueller_matrix",
    "retardance",
    "stokes_at_detector",
    "stokes_from_jones_vector",
    "stokes_from_state",
    "transverse_frame",
]


def _dot(a: Array, b: Array) -> Array:
    """Row-wise dot product of two (N, 3) arrays."""
    return be.sum(a * b, axis=1)


def _unit(v: Array) -> Array:
    return v / be.unsqueeze_last(be.linalg.norm(v, axis=1))


def _rows(v: Array, n: int) -> Array:
    """Broadcast a 3-vector or an (N, 3) array to (N, 3)."""
    v = be.array(v)
    if v.ndim == 1:
        v = be.broadcast_to(v, (n, 3))
    return v


def _complex(re: Any, im: Any) -> Array:
    """Return the complex backend array re + i im from real array-likes."""
    return be.to_complex(be.array(re)) + 1j * be.to_complex(be.array(im))


def detector_frame(k: Array, x_ref: Array = (1.0, 0.0, 0.0)) -> tuple[Array, Array]:
    """Return the detector frame (x, y) for the ray directions k.

    x is the projection of ``x_ref`` onto the plane normal to k, normalized;
    y = k x x. Then (x, y, k) is right-handed.

    Args:
        k: (N, 3) unit ray directions.
        x_ref: The reference vector, a 3-vector or (N, 3). It must not be
            parallel to a ray.

    Returns:
        tuple: (x, y), each (N, 3).

    Raises:
        ValueError: If ``x_ref`` is parallel to a ray direction.
    """
    k = be.array(k)
    ref = _rows(x_ref, k.shape[0])
    x = ref - be.unsqueeze_last(_dot(ref, k)) * k
    norm = be.linalg.norm(x, axis=1)
    if be.any(norm < 1e-12):
        raise ValueError("x_ref is parallel to a ray direction.")
    x = x / be.unsqueeze_last(norm)
    y = be.cross(k, x)
    return x, y


def transverse_frame(k: Array) -> tuple[Array, Array]:
    """Return a right-handed transverse frame (x, y) for any ray direction.

    The reference is the global x axis, or the global y axis for a ray within
    about 26 degrees of the x axis.

    Args:
        k: (N, 3) unit ray directions.

    Returns:
        tuple: (x, y), each (N, 3).
    """
    k = be.array(k)
    n = k.shape[0]
    ex = be.broadcast_to(be.array([1.0, 0.0, 0.0]), (n, 3))
    ey = be.broadcast_to(be.array([0.0, 1.0, 0.0]), (n, 3))
    near_x = be.unsqueeze_last(be.abs(k[:, 0]) > 0.9)
    return detector_frame(k, be.where(near_x, ey, ex))


def jones_from_prt(
    prt: Array, x_in: Array, y_in: Array, x_out: Array, y_out: Array
) -> Array:
    """Return the 2 x 2 Jones matrix of a PRT matrix in stated frames.

    J[i, j] = e_out_i · (P e_in_j) with e_in = (x_in, y_in) and
    e_out = (x_out, y_out). The frame vectors are real.

    Args:
        prt: (N, 3, 3) PRT matrices.
        x_in, y_in: (N, 3) input frame vectors, transverse to k_in.
        x_out, y_out: (N, 3) output frame vectors, transverse to k_out.

    Returns:
        be.ndarray: (N, 2, 2) complex Jones matrices.
    """
    col_x = be.sum(prt * be.to_complex(x_in)[:, None, :], axis=2)
    col_y = be.sum(prt * be.to_complex(y_in)[:, None, :], axis=2)
    x_out = be.to_complex(x_out)
    y_out = be.to_complex(y_out)
    n = prt.shape[0]
    J = be.to_complex(be.zeros((n, 2, 2)))
    J[:, 0, 0] = _dot(x_out, col_x)
    J[:, 0, 1] = _dot(x_out, col_y)
    J[:, 1, 0] = _dot(y_out, col_x)
    J[:, 1, 1] = _dot(y_out, col_y)
    return J


def _stokes_matrices() -> tuple[Array, Array]:
    """Return A and A⁻¹, with S = A (E ⊗ E*) and
    E ⊗ E* = (ExEx*, ExEy*, EyEx*, EyEy*)."""
    a = _complex(
        [[1.0, 0.0, 0.0, 1.0], [1.0, 0.0, 0.0, -1.0], [0.0, 1.0, 1.0, 0.0], [0.0] * 4],
        [[0.0] * 4, [0.0] * 4, [0.0] * 4, [0.0, -1.0, 1.0, 0.0]],
    )
    a_inv = _complex(
        [
            [0.5, 0.5, 0.0, 0.0],
            [0.0, 0.0, 0.5, 0.0],
            [0.0, 0.0, 0.5, 0.0],
            [0.5, -0.5, 0.0, 0.0],
        ],
        [[0.0] * 4, [0.0, 0.0, 0.0, 0.5], [0.0, 0.0, 0.0, -0.5], [0.0] * 4],
    )
    return a, a_inv


def jones_to_mueller(J: Array) -> Array:
    """Return the Mueller matrices M = A (J ⊗ J*) A⁻¹ of Jones matrices.

    Args:
        J: (N, 2, 2) Jones matrices.

    Returns:
        be.ndarray: (N, 4, 4) real Mueller matrices.
    """
    J = be.to_complex(J)
    Jc = be.real(J) - 1j * be.imag(J)
    n = J.shape[0]
    kron = be.reshape(J[:, :, None, :, None] * Jc[:, None, :, None, :], (n, 4, 4))
    A, A_inv = _stokes_matrices()
    M = be.matmul(be.matmul(be.broadcast_to(A, (n, 4, 4)), kron), A_inv)
    return be.real(M)


def stokes_from_jones_vector(E: Array) -> Array:
    """Return the Stokes vectors of Jones vectors (Ex, Ey).

    Args:
        E: (N, 2) complex Jones vectors.

    Returns:
        be.ndarray: (N, 4) Stokes vectors.
    """
    E = be.to_complex(E)
    ex, ey = E[:, 0], E[:, 1]
    cross = ex * be.conj(ey)
    s0 = be.abs(ex) ** 2 + be.abs(ey) ** 2
    s1 = be.abs(ex) ** 2 - be.abs(ey) ** 2
    return be.stack([s0, s1, 2 * be.real(cross), 2 * be.imag(cross)], axis=1)


def stokes_from_state(state: PolarizationState | None) -> Array:
    """Return the input Stokes vector of a polarization state.

    The Jones vector is (Ex e^{iφx}, Ey e^{iφy}) in the input frame (s, p) of
    the ray. An unpolarized state (or None) gives (1, 0, 0, 0).

    Args:
        state: The polarization state.

    Returns:
        be.ndarray: (4,) Stokes vector.
    """
    if state is None or not state.is_polarized:
        return be.array([1.0, 0.0, 0.0, 0.0])
    ex = (state.Ex or 0.0) * complex(np.exp(1j * (state.phase_x or 0.0)))
    ey = (state.Ey or 0.0) * complex(np.exp(1j * (state.phase_y or 0.0)))
    E = _complex([[ex.real, ey.real]], [[ex.imag, ey.imag]])
    return stokes_from_jones_vector(E)[0]


def _path_jones(rays: PolarizedRays, x_ref: Array) -> Array:
    x_in, y_in = rays.get_input_basis()
    k_out = be.stack([rays.L, rays.M, rays.N], axis=1)
    x_out, y_out = detector_frame(k_out, x_ref)
    return jones_from_prt(rays.p, x_in, y_in, x_out, y_out)


def mueller_matrix(rays: PolarizedRays, x_ref: Array = (1.0, 0.0, 0.0)) -> Array:
    """Return the Mueller matrix of the traced path of each ray.

    The input frame is the frame of the input field of the ray (x = s,
    y = p of ``PolarizedRays.get_input_basis``); the output frame is the
    detector frame of ``x_ref`` (:func:`detector_frame`). M is in power
    units: it includes ``rays.flux_factor``, so M[0, 0] is the transmittance
    of the path for unpolarized light. Scalar losses that are not in the PRT
    matrix (bulk absorption, ``SimpleCoating``) are not included.

    Args:
        rays: Traced polarized rays.
        x_ref: The reference vector of the detector frame.

    Returns:
        be.ndarray: (N, 4, 4) Mueller matrices.
    """
    M = jones_to_mueller(_path_jones(rays, x_ref))
    return M * rays.flux_factor[:, None, None]


def stokes_at_detector(
    rays: PolarizedRays,
    state: PolarizationState | None,
    x_ref: Array = (1.0, 0.0, 0.0),
) -> Array:
    """Return the Stokes vector of each ray in the detector frame of ``x_ref``.

    S = M S_in i_0, with M of :func:`mueller_matrix` and S_in of
    :func:`stokes_from_state`. S0 equals the intensity that
    ``PolarizedRays.update_intensity`` gives for a path without scalar
    losses.

    Args:
        rays: Traced polarized rays.
        state: The input polarization state (None for unpolarized).
        x_ref: The reference vector of the detector frame.

    Returns:
        be.ndarray: (N, 4) Stokes vectors.
    """
    M = mueller_matrix(rays, x_ref)
    s_in = be.to_complex(stokes_from_state(state)).real
    S = be.sum(M * s_in[None, None, :], axis=2)
    return S * be.unsqueeze_last(rays._i0)


def diattenuation(prt: Array, k_in: Array, k_out: Array) -> Array:
    """Return the diattenuation of PRT matrices (Yun et al. 2011, part I).

    D = (Λ1² - Λ2²) / (Λ1² + Λ2²), with Λ1 >= Λ2 the singular values of the
    transverse part of P. D is 0 for a non-diattenuating path and 1 for a
    polarizer. The value does not depend on the choice of frames.

    Args:
        prt: (N, 3, 3) PRT matrices.
        k_in: (N, 3) input ray directions.
        k_out: (N, 3) output ray directions.

    Returns:
        be.ndarray: (N,) diattenuation.
    """
    x_in, y_in = transverse_frame(k_in)
    x_out, y_out = transverse_frame(k_out)
    J = jones_from_prt(prt, x_in, y_in, x_out, y_out)
    return _jones_diattenuation(J)


def _jones_diattenuation(J: Array) -> Array:
    total = be.sum(be.sum(be.abs(J) ** 2, axis=2), axis=1)  # Λ1² + Λ2²
    det = J[:, 0, 0] * J[:, 1, 1] - J[:, 0, 1] * J[:, 1, 0]  # |det| = Λ1 Λ2
    ratio = 4 * be.abs(det) ** 2 / total**2
    return be.sqrt(be.clip(1 - ratio, 0.0, 1.0))


def retardance(prt: Array, q: Array, k_in: Array) -> Array:
    """Return the retardance of PRT matrices (Yun et al. 2011, part II).

    The retardance is the phase difference of the eigenpolarizations of the
    unitary factor of P after the geometrical transformation Q is removed:
    Q is the PRT matrix of the same path without polarization effects
    (``PolarizedRays.q``, the parallel transport of the ray frame). The value
    is in [0, π]. It is not defined for a matrix with a zero singular value in
    the transverse plane (an ideal polarizer); the result is then NaN.

    Args:
        prt: (N, 3, 3) PRT matrices.
        q: (N, 3, 3) geometrical transformations of the same paths.
        k_in: (N, 3) input ray directions.

    Returns:
        be.ndarray: (N,) retardance in radians.
    """
    x_in, y_in = transverse_frame(k_in)
    q = be.real(be.to_complex(q))
    x_out = be.sum(q * x_in[:, None, :], axis=2)
    y_out = be.sum(q * y_in[:, None, :], axis=2)
    J = jones_from_prt(prt, x_in, y_in, x_out, y_out)

    # Unitary polar factor of a 2 x 2 matrix: U = (J + e^{iφ} adj(J)^H) /
    # (Λ1 + Λ2), with φ = arg det J; det U = e^{iφ}.
    a, b = J[:, 0, 0], J[:, 0, 1]
    c, d = J[:, 1, 0], J[:, 1, 1]
    det = a * d - b * c
    abs_det = be.abs(det)
    valid = abs_det > 1e-15 * be.sum(be.sum(be.abs(J) ** 2, axis=2), axis=1)
    phase = det / be.where(valid, abs_det, be.ones_like(abs_det))
    norm = be.sqrt(be.sum(be.sum(be.abs(J) ** 2, axis=2), axis=1) + 2 * abs_det)
    u00 = (a + phase * be.conj(d)) / norm
    u01 = (b - phase * be.conj(c)) / norm
    u10 = (c - phase * be.conj(b)) / norm
    u11 = (d + phase * be.conj(a)) / norm

    # V = U / sqrt(det U) is in SU(2): V = cos(δ/2) I + i sin(δ/2) n·σ. The
    # atan2 form keeps the precision near δ = 0 and δ = π.
    root = be.sqrt(phase)
    half_trace = (u00 + u11) / (2 * root)
    off = be.sqrt(
        be.abs((u00 - u11) / (2 * root)) ** 2
        + (be.abs(u01) ** 2 + be.abs(u10) ** 2) / 2
    )
    delta = 2 * be.arctan2(off, be.abs(half_trace))
    return be.where(valid, delta, be.full_like(delta, float("nan")))


def diattenuation_from_mueller(M: Array) -> Array:
    """Return the diattenuation of Mueller matrices (Lu and Chipman 1996).

    D = sqrt(m01² + m02² + m03²) / m00.

    Args:
        M: (N, 4, 4) Mueller matrices.

    Returns:
        be.ndarray: (N,) diattenuation.
    """
    d = be.sqrt(M[:, 0, 1] ** 2 + M[:, 0, 2] ** 2 + M[:, 0, 3] ** 2)
    return d / M[:, 0, 0]


def depolarization_index(M: Array) -> Array:
    """Return the depolarization index of Mueller matrices (Gil and Bernabeu).

    DI = sqrt(Σ m_ij² - m00²) / (√3 m00): 1 for a non-depolarizing Mueller
    matrix, 0 for an ideal depolarizer.

    Args:
        M: (N, 4, 4) Mueller matrices.

    Returns:
        be.ndarray: (N,) depolarization index.
    """
    total = be.sum(be.sum(M**2, axis=2), axis=1)
    m00 = M[:, 0, 0]
    return be.sqrt(be.clip(total - m00**2, 0.0, None)) / (3**0.5 * m00)
