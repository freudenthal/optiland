"""Eigenmodes of a Homogeneous Medium

The four plane-wave eigenmodes of a medium with a 6 x 6 constitutive matrix for
a given tangential wave number, in the interface frame, sorted and scaled.

Conventions:

* Time dependence exp(-iωt); a mode is exp(i k0 (K x + q z)). Wave vectors are
  in units of k0.
* Tellegen form with H' = η0 H: D' = ε E + ξ H', B' = ζ E + μ H'. Maxwell for
  a plane wave: k × E = B' and k × H' = -D'.
* ψ = (E_x, E_y, H'_x, H'_y). With k = (K, 0, q) the z rows give
  D'_z + K H'_y = 0 and B'_z - K E_y = 0, which fix E_z and H'_z; the
  transverse rows give q E_x = B'_y + K E_z, q E_y = -B'_x,
  q H'_x = -D'_y + K H'_z and q H'_y = D'_x. Thus q ψ = Δ ψ (Berreman 1972).
* Sorting (equation register E-09): a mode is forward (it carries power or
  decays into +z) if Im q > 1E-9 max(1, |q|), else if S_z > 0. The sign of
  Re q is never used: a forward extraordinary mode can have Re q < 0. A grazing
  mode (|S_z| <= 1E-12 |ψ|²) falls back to Re q > 0; if the rule does not give
  two forward modes, the two modes with the larger Re q + Im q are forward. Both
  fallbacks set ``fallback``.
* Column order (f1, f2, b1, b2): two forward modes, then two backward modes;
  in each direction the s-like mode (the larger |E_y| / |ψ|) first. An |E_y|
  tie (1E-9 relative; the circular modes of a chiral medium at normal
  incidence) goes to the larger |q|.
* A degenerate pair (|q1 - q2| <= 1E-9 max(1, |q1|): an isotropic medium, the
  optic axis of a uniaxial medium, a binormal of a biaxial medium) has no
  unique eigenvectors. Its columns are the canonical basis of the pair: E_x = 0
  (s-like) and E_y = 0 (p-like), and ``degenerate`` is set.
* Flux orthogonality: in a lossless medium a propagating pair that is not
  degenerate is made flux-orthogonal (the Löwdin map of its 2 x 2 flux Gram
  matrix). The exact modes have no cross flux; near a degeneracy the ``eig``
  vectors mix by about 1E-16 |Δ| / |q1 - q2|, which would break the sum of the
  per-mode powers.
* Phase: an s-like column has E_y real >= 0, a p-like column H'_y real >= 0
  (if that component is below 1E-12 |ψ|, H'_x or E_x instead). Scale: the
  flux |S_z| = 1 (S = Re(E × H'*) / 2), or |ψ| = 1 for a mode that carries no
  power along z (evanescent).

References:

* D. W. Berreman, "Optics in stratified and anisotropic media: 4 x 4-matrix
  formulation," J. Opt. Soc. Am. 62, 502-510 (1972).
* J. Lekner, "Reflection and refraction by uniaxial crystals," J. Phys.:
  Condens. Matter 3, 6121-6133 (1991): the ordinary and extraordinary modes
  and the sign of the forward root.
* M. V. Berry, M. R. Jeffrey, "Conical diffraction: Hamilton's diabolical point
  at the heart of crystal optics," Prog. Opt. 50, 13-50 (2007): the
  degeneracy at a binormal.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

import optiland.backend as be
from optiland.anisotropic.frames import (
    _complex_vectors,
    _sum,
    _unit,
    interface_frame,
    rotate_constitutive,
    to_global,
)
from optiland.materials.anisotropic import _complex

Array = Any

__all__ = [
    "Eigenmodes",
    "PlaneWaveModes",
    "berreman_delta",
    "medium_modes",
    "plane_wave_modes",
]

#: A mode with |Im q| above this (times max(1, |q|)) is evanescent or absorbed.
FORWARD_RTOL = 1e-9
#: A mode with |S_z| at or below this (times |ψ|²) is grazing.
GRAZING_RTOL = 1e-12
#: Two modes of one direction with |q1 - q2| at or below this are degenerate.
DEGENERATE_RTOL = 1e-9
#: Two |E_y| / |ψ| values within this (relative) are a tie.
EY_TIE_RTOL = 1e-9
#: A mode with |S_z| at or below this (times |ψ|²) carries no power along z.
FLUX_RTOL = 1e-14
#: An entry of ε, μ below this (times the matrix norm) is zero for the
#: isotropic test.
ISOTROPIC_RTOL = 1e-12
#: A phase reference component below this (times |ψ|) is not used.
PHASE_RTOL = 1e-12

# Constant matrices (NumPy; converted to the backend on use).
# The transverse components (E_x, E_y, H'_x, H'_y) in F = (E, H').
_TRANSVERSE = [0, 1, 3, 4]
_E4 = np.eye(4)
_E3 = np.eye(3)
# Isotropic columns (f_s, f_p, b_s, b_p) = _ISO_1 + (q/μ) _ISO_QM + (q/n) _ISO_QN
# + (ε/n) _ISO_EN.
_ISO_1 = np.zeros((4, 4))
_ISO_1[1, 0], _ISO_1[1, 2] = 1.0, 1.0
_ISO_QM = np.zeros((4, 4))
_ISO_QM[2, 0], _ISO_QM[2, 2] = -1.0, 1.0
_ISO_QN = np.zeros((4, 4))
_ISO_QN[0, 1], _ISO_QN[0, 3] = 1.0, -1.0
_ISO_EN = np.zeros((4, 4))
_ISO_EN[3, 1], _ISO_EN[3, 3] = 1.0, 1.0
_ISO_SIGN = np.array([1.0, 1.0, -1.0, -1.0])
# The expected isotropic matrix: ε _EPS_BLOCK + μ _MU_BLOCK.
_EPS_BLOCK = np.diag([1.0, 1.0, 1.0, 0.0, 0.0, 0.0])
_MU_BLOCK = np.diag([0.0, 0.0, 0.0, 1.0, 1.0, 1.0])
# A matrix with four distinct eigenvalues: the stand-in for isotropic rows in
# ``eig`` (their columns come from the closed form).
_DUMMY_DELTA = np.diag([1.0, 2.0, -1.0, -2.0])
# Phase reference rows of ψ per column (s-like: E_y, then H'_x; p-like: H'_y,
# then E_x).
# Entry [row, column] = 1 selects the row of ψ for that column.
_PHASE_MAIN = np.zeros((4, 4))
_PHASE_MAIN[[1, 1], [0, 2]] = 1.0
_PHASE_MAIN[[3, 3], [1, 3]] = 1.0
_PHASE_ALT = np.zeros((4, 4))
_PHASE_ALT[[2, 2], [0, 2]] = 1.0
_PHASE_ALT[[0, 0], [1, 3]] = 1.0


def _c(x: np.ndarray) -> Array:
    """Return a NumPy constant as a complex backend array."""
    return _complex(x)


def _conj(z: Array) -> Array:
    """Return the complex conjugate (not a lazy Torch conjugate, R-22)."""
    return be.real(z) - 1j * be.imag(z)


def _abs2(z: Array) -> Array:
    """Return |z|² as a real array."""
    return be.real(z) ** 2 + be.imag(z) ** 2


def _flux_z(psi: Array) -> Array:
    """Return S_z = Re(E_x H'_y* - E_y H'_x*) / 2 of columns ψ (N, 4, M)."""
    return 0.5 * be.real(
        psi[:, 0, :] * _conj(psi[:, 3, :]) - psi[:, 1, :] * _conj(psi[:, 2, :])
    )


def _poynting(e: Array, h: Array) -> Array:
    """Return S = Re(E × H'*) / 2 of vectors on the last axis."""
    return 0.5 * be.real(be.cross(e, _conj(h)))


def _column_scale(psi: Array) -> Array:
    """Return the flux scale of each column: sqrt|S_z|, or |ψ| without flux."""
    flux = be.abs(_flux_z(psi))
    norm2 = _sum(_abs2(psi), axis=-2)
    has_flux = flux > FLUX_RTOL * norm2
    return be.sqrt(be.where(has_flux, flux, norm2)), has_flux


# ---------------------------------------------------------------------------
# The Berreman matrix
# ---------------------------------------------------------------------------


def _z_rows(tangential: Array, matrix: Array) -> tuple[Array, Array]:
    """Return the row vectors e_z, h_z with E_z = e_z · ψ and H'_z = h_z · ψ.

    From D'_z + K H'_y = 0 and B'_z - K E_y = 0, solved for (E_z, H'_z).
    """
    k = tangential[:, None]
    mt = matrix[:, :, _TRANSVERSE]  # (N, 6, 4): M F without E_z, H'_z
    e4 = _c(_E4)
    rhs_d = -mt[:, 2, :] - k * e4[3]
    rhs_b = -mt[:, 5, :] + k * e4[1]
    a, b = matrix[:, 2, 2][:, None], matrix[:, 2, 5][:, None]
    c, d = matrix[:, 5, 2][:, None], matrix[:, 5, 5][:, None]
    det = a * d - b * c
    e_z = (d * rhs_d - b * rhs_b) / det
    h_z = (-c * rhs_d + a * rhs_b) / det
    return e_z, h_z


def berreman_delta(tangential: Any, matrix: Any) -> Array:
    """Return the Berreman matrix Δ of each medium: q ψ = Δ ψ.

    Args:
        tangential: Tangential wave numbers K in units of k0, shape (N,) or
            scalar, complex allowed.
        matrix: Constitutive matrices [[ε, ξ], [ζ, μ]] in the interface frame,
            shape (N, 6, 6) or (6, 6).

    Returns:
        Δ, shape (N, 4, 4), complex, for ψ = (E_x, E_y, H'_x, H'_y).
    """
    m, k = _inputs(tangential, matrix)
    return _delta(k, m)[0]


def _inputs(tangential: Any, matrix: Any) -> tuple[Array, Array]:
    """Return (M (N, 6, 6), K (N,)) as complex backend arrays."""
    m = _complex(matrix)
    if len(m.shape) == 2:
        m = be.reshape(m, (1, 6, 6))
    k = be.reshape(_complex(tangential), (-1,))
    n_rows = max(m.shape[0], k.shape[0])
    return be.broadcast_to(m, (n_rows, 6, 6)), be.broadcast_to(k, (n_rows,))


def _delta(tangential: Array, matrix: Array) -> tuple[Array, Array, Array]:
    """Return (Δ, e_z, h_z) for complex backend inputs of equal length."""
    e_z, h_z = _z_rows(tangential, matrix)
    # (N, 6, 4): the rows of M F = (D', B') as coefficients of ψ.
    mf = (
        matrix[:, :, _TRANSVERSE]
        + matrix[:, :, 2:3] * e_z[:, None, :]
        + matrix[:, :, 5:6] * h_z[:, None, :]
    )
    k = tangential[:, None]
    rows = [
        mf[:, 4, :] + k * e_z,  # q E_x = B'_y + K E_z
        -mf[:, 3, :],  # q E_y = -B'_x
        -mf[:, 1, :] + k * h_z,  # q H'_x = -D'_y + K H'_z
        mf[:, 0, :],  # q H'_y = D'_x
    ]
    delta = be.concatenate([row[:, None, :] for row in rows], axis=1)
    return delta, e_z, h_z


# ---------------------------------------------------------------------------
# Eigenmodes
# ---------------------------------------------------------------------------


@dataclass
class Eigenmodes:
    """The sorted, flux-scaled eigenmodes of media in the interface frame.

    Column and mode order (f1, f2, b1, b2): forward s-like, forward p-like,
    backward s-like, backward p-like. All vectors are in the interface frame.

    Attributes:
        q: Normal wave numbers, shape (N, 4), complex.
        psi: Columns (E_x, E_y, H'_x, H'_y), shape (N, 4, 4) (column = mode).
        E: Electric fields, shape (N, 4, 3), complex.
        H: Magnetic fields H' = η0 H, shape (N, 4, 3), complex.
        S: Poynting vectors Re(E × H'*) / 2, shape (N, 4, 3), real.
        evanescent: True for a mode with no power flux along z, shape (N, 4).
        degenerate: True for a degenerate pair (forward, backward), shape
            (N, 2).
        fallback: True if the sorting used a fallback rule, shape (N,).
        isotropic: True for a medium that is isotropic to 1E-12, shape (N,).
    """

    q: Array
    psi: Array
    E: Array
    H: Array
    S: Array
    evanescent: Array
    degenerate: Array
    fallback: Array
    isotropic: Array


def _is_isotropic(matrix: Array) -> Array:
    """Return True for each matrix that is ε I ⊕ μ I to 1E-12 (relative)."""
    expected = matrix[:, 0, 0][:, None, None] * _c(_EPS_BLOCK) + matrix[:, 3, 3][
        :, None, None
    ] * _c(_MU_BLOCK)
    n_rows = matrix.shape[0]
    deviation = _sum(be.reshape(_abs2(matrix - expected), (n_rows, 36)), axis=-1)
    scale = _sum(be.reshape(_abs2(matrix), (n_rows, 36)), axis=-1)
    return deviation <= ISOTROPIC_RTOL**2 * scale


def _isotropic_columns(tangential: Array, matrix: Array) -> tuple[Array, Array]:
    """Return (q, ψ) of the closed-form isotropic modes (f_s, f_p, b_s, b_p).

    s± = (0, 1, ∓q/μ, 0), p± = (±q/n, 0, 0, ε/n) with n = sqrt(εμ) and q the
    principal root of εμ - K².
    """
    eps, mu = matrix[:, 0, 0], matrix[:, 3, 3]
    n = be.sqrt(eps * mu)
    q = be.sqrt(eps * mu - tangential**2)
    psi = (
        _c(_ISO_1)[None]
        + (q / mu)[:, None, None] * _c(_ISO_QM)[None]
        + (q / n)[:, None, None] * _c(_ISO_QN)[None]
        + (eps / n)[:, None, None] * _c(_ISO_EN)[None]
    )
    return q[:, None] * _c(_ISO_SIGN)[None], psi


def _rank_less(values: Array) -> Array:
    """Return the number of entries before each entry, sorted in descending order.

    Equal values are ordered by index. Shape (N, 4) -> (N, 4) of floats.
    """
    vi, vj = values[:, :, None], values[:, None, :]
    idx = be.asarray(np.arange(4.0))
    earlier = idx[None, None, :] < idx[None, :, None]  # j < i
    before = (vj > vi) | ((vj == vi) & earlier)
    return _sum(
        be.where(before, be.ones_like(vi * vj), be.zeros_like(vi * vj)), axis=-1
    )


def _eig_columns(delta: Array) -> tuple[Array, Array, Array]:
    """Return (q, ψ, fallback) of the sorted eigenvectors of Δ (not scaled)."""
    q, v = be.linalg.eig(delta)
    norm2 = _sum(_abs2(v), axis=-2)
    flux = _flux_z(v)
    q_abs = be.abs(q)
    one = be.ones_like(q_abs)
    evan = be.abs(be.imag(q)) > FORWARD_RTOL * be.maximum(one, q_abs)
    graze = (~evan) & (be.abs(flux) <= GRAZING_RTOL * norm2)
    forward = be.where(evan, be.imag(q) > 0, be.where(graze, be.real(q) > 0, flux > 0))
    count = _sum(be.where(forward, one, 0.0 * one), axis=-1)
    bad = count != 2.0
    by_rank = _rank_less(be.real(q) + be.imag(q)) < 2
    forward = be.where(bad[:, None], by_rank, forward)
    n_graze = _sum(be.where(graze, one, 0.0 * one), axis=-1)
    fallback = bad | (n_graze > 0)

    # Position of each mode: 2 for a backward mode, plus its rank in its group.
    ey = be.sqrt(_abs2(v[:, 1, :]) / norm2)
    ey_i, ey_j = ey[:, :, None], ey[:, None, :]
    qa_i, qa_j = q_abs[:, :, None], q_abs[:, None, :]
    idx = be.asarray(np.arange(4.0))
    earlier = idx[None, None, :] < idx[None, :, None]
    same = forward[:, :, None] == forward[:, None, :]
    tie = be.abs(ey_i - ey_j) <= EY_TIE_RTOL * be.maximum(ey_i, ey_j)
    before = same & (
        ((~tie) & (ey_j > ey_i))
        | (tie & (qa_j > qa_i))
        | (tie & (qa_j == qa_i) & earlier)
    )
    ones = be.ones_like(ey_i * ey_j)
    rank = _sum(be.where(before, ones, 0.0 * ones), axis=-1)
    position = 2.0 * be.where(forward, 0.0 * one, one) + rank
    permutation = be.to_complex(
        be.where(
            position[:, :, None] == idx[None, None, :],
            be.ones_like(ones),
            be.zeros_like(ones),
        )
    )  # (N, 4, 4): mode i goes to column position_i
    v_sorted = be.matmul(v, permutation)
    q_sorted = be.matmul(q[:, None, :], permutation)[:, 0, :]
    return q_sorted, v_sorted, fallback


def _canonical(a: Array, b: Array) -> tuple[Array, Array]:
    """Return the (E_x = 0, E_y = 0) basis of the span of columns a, b (N, 4)."""
    s = a * b[:, 0:1] - b * a[:, 0:1]
    p = a * b[:, 1:2] - b * a[:, 1:2]

    def unit(x: Array) -> Array:
        norm = be.sqrt(_sum(_abs2(x), axis=-1))
        return x / be.to_complex(be.where(norm > 0, norm, be.ones_like(norm)))[:, None]

    return unit(s), unit(p)


def _columns(parts: list[Array]) -> Array:
    """Return the matrix with the columns ``parts`` (each (N, 4)), (N, 4, 4)."""
    return be.concatenate([col[:, :, None] for col in parts], axis=-1)


def _degenerate_pairs(q: Array, psi: Array) -> tuple[Array, Array, Array]:
    """Replace each degenerate pair by its canonical basis; return (q, ψ, flags)."""
    cols = [psi[:, :, i] for i in range(4)]
    qs = [q[:, i] for i in range(4)]
    flags = []
    for i, j in ((0, 1), (2, 3)):
        scale = be.maximum(be.abs(qs[i]), be.ones_like(be.abs(qs[i])))
        deg = be.abs(qs[i] - qs[j]) <= DEGENERATE_RTOL * scale
        s, p = _canonical(cols[i], cols[j])
        cols[i] = be.where(deg[:, None], s, cols[i])
        cols[j] = be.where(deg[:, None], p, cols[j])
        mean = 0.5 * (qs[i] + qs[j])
        qs[i] = be.where(deg, mean, qs[i])
        qs[j] = be.where(deg, mean, qs[j])
        flags.append(deg)
    q_out = be.concatenate([x[:, None] for x in qs], axis=-1)
    return q_out, _columns(cols), be.concatenate([f[:, None] for f in flags], axis=-1)


def _phase_and_scale(psi: Array) -> tuple[Array, Array]:
    """Apply the phase rule and the flux scale to columns ψ (N, 4, 4)."""
    norm = be.sqrt(_sum(_abs2(psi), axis=-2))  # (N, 4)
    main = _sum(psi * _c(_PHASE_MAIN)[None], axis=-2)
    alt = _sum(psi * _c(_PHASE_ALT)[None], axis=-2)
    ref = be.where(be.abs(main) > PHASE_RTOL * norm, main, alt)
    mag = be.abs(ref)
    phase = be.where(
        mag > PHASE_RTOL * norm,
        _conj(ref) / be.to_complex(be.where(mag > 0, mag, be.ones_like(mag))),
        be.to_complex(be.ones_like(mag)),
    )
    psi = psi * phase[:, None, :]
    scale, has_flux = _column_scale(psi)
    return psi / be.to_complex(scale)[:, None, :], ~has_flux


def _is_transparent(matrix: Array) -> Array:
    """Return True for each matrix that is Hermitian to 1E-12 (lossless)."""
    n_rows = matrix.shape[0]
    adjoint = _conj(be.transpose(matrix, (0, 2, 1)))
    deviation = _sum(be.reshape(_abs2(matrix - adjoint), (n_rows, 36)), axis=-1)
    scale = _sum(be.reshape(_abs2(matrix), (n_rows, 36)), axis=-1)
    return deviation <= ISOTROPIC_RTOL**2 * scale


def _flux_orthogonalize(
    psi: Array, transparent: Array, evanescent: Array, degenerate: Array
) -> Array:
    """Make each propagating pair of a lossless medium flux-orthogonal.

    In a lossless medium two propagating modes with q1 ≠ q2 carry no cross
    flux: ψ1ᴴ Σ ψ2 = 0 with S_z = ψᴴ Σ ψ. Near a degeneracy the eigenvectors of
    ``eig`` mix by about 1E-16 |Δ| / |q1 - q2|, so the per-mode powers do not
    add. The symmetric (Löwdin) map Ψ G^(-1/2) with G = σ Ψᴴ Σ Ψ of the pair
    (σ = ±1, the flux sign) removes the cross flux with the smallest change;
    each column stays a mode to about 1E-16 |Δ|. Flagged degenerate pairs
    keep the canonical basis.
    """
    sigma = _c(0.25 * np.fliplr(np.diag([1.0, -1.0, -1.0, 1.0])))
    gram = be.matmul(_conj(be.transpose(psi, (0, 2, 1))), be.matmul(sigma[None], psi))
    cols = [psi[:, :, i] for i in range(4)]
    for pair, (i, j) in enumerate(((0, 1), (2, 3))):
        g11, g12 = gram[:, i, i], gram[:, i, j]
        g21, g22 = gram[:, j, i], gram[:, j, j]
        sign = be.where(
            be.real(g11) < 0, -be.ones_like(be.real(g11)), be.ones_like(be.real(g11))
        )
        sign_c = be.to_complex(sign)
        g11, g12, g21, g22 = sign_c * g11, sign_c * g12, sign_c * g21, sign_c * g22
        det = be.real(g11 * g22 - g12 * g21)
        ok = (
            transparent
            & ~evanescent[:, i]
            & ~evanescent[:, j]
            & ~degenerate[:, pair]
            & (be.real(sign_c * gram[:, j, j]) > 0)
            & (det > 0)
        )
        root = be.sqrt(be.where(ok, det, be.ones_like(det)))
        t = be.sqrt(be.where(ok, be.real(g11 + g22) + 2 * root, be.ones_like(det)))
        scale = be.to_complex(root * t)
        # G^(-1/2) = adj(G + sI) / (s t), s = sqrt(det G), t = sqrt(tr G + 2s).
        root_c = be.to_complex(root)
        x11, x12 = (g22 + root_c) / scale, -g12 / scale
        x21, x22 = -g21 / scale, (g11 + root_c) / scale
        a, b = cols[i], cols[j]
        new_a = a * x11[:, None] + b * x21[:, None]
        new_b = a * x12[:, None] + b * x22[:, None]
        cols[i] = be.where(ok[:, None], new_a, a)
        cols[j] = be.where(ok[:, None], new_b, b)
    return _columns(cols)


def _modes(tangential: Array, matrix: Array) -> Eigenmodes:
    """Return the eigenmodes for complex backend inputs of equal length."""
    delta, e_z, h_z = _delta(tangential, matrix)
    iso = _is_isotropic(matrix)
    q, psi = _isotropic_columns(tangential, matrix)
    degenerate = be.concatenate([iso[:, None], iso[:, None]], axis=-1)
    fallback = iso & ~iso
    if not bool(be.all(iso)):  # eig only when a row is not isotropic
        dummy = be.to_complex(be.zeros_like(be.real(delta))) + _c(_DUMMY_DELTA)[None]
        q_eig, psi_eig, fallback = _eig_columns(
            be.where(iso[:, None, None], dummy, delta)
        )
        q_eig, psi_eig, degenerate = _degenerate_pairs(q_eig, psi_eig)
        q = be.where(iso[:, None], q, q_eig)
        psi = be.where(iso[:, None, None], psi, psi_eig)
        degenerate = iso[:, None] | degenerate
        fallback = fallback & ~iso

    psi, evanescent = _phase_and_scale(psi)
    if not bool(be.all(iso)):
        transparent = _is_transparent(matrix)
        psi = _flux_orthogonalize(psi, transparent, evanescent, degenerate)
        psi, evanescent = _phase_and_scale(psi)
    e_zm = _sum(e_z[:, :, None] * psi, axis=1)  # (N, 4 modes)
    h_zm = _sum(h_z[:, :, None] * psi, axis=1)
    e = be.concatenate(
        [be.transpose(psi[:, 0:2, :], (0, 2, 1)), e_zm[:, :, None]], axis=-1
    )
    h = be.concatenate(
        [be.transpose(psi[:, 2:4, :], (0, 2, 1)), h_zm[:, :, None]], axis=-1
    )
    return Eigenmodes(
        q=q,
        psi=psi,
        E=e,
        H=h,
        S=_poynting(e, h),
        evanescent=evanescent,
        degenerate=degenerate,
        fallback=fallback,
        isotropic=iso,
    )


def medium_modes(tangential: Any, matrix: Any) -> Eigenmodes:
    """Return the sorted, flux-scaled eigenmodes of media in the interface frame.

    Args:
        tangential: Tangential wave numbers K (k = (K, 0, q)) in units of k0,
            shape (N,) or scalar, complex allowed.
        matrix: Constitutive matrices [[ε, ξ], [ζ, μ]] in the interface frame,
            shape (N, 6, 6) or (6, 6).

    Returns:
        The eigenmodes, in the order (f1, f2, b1, b2) (see the module
        docstring for the sorting, the degenerate basis and the scale).
    """
    m, k = _inputs(tangential, matrix)
    return _modes(k, m)


# ---------------------------------------------------------------------------
# Plane-wave modes along a wave normal
# ---------------------------------------------------------------------------


@dataclass
class PlaneWaveModes:
    """The two forward plane-wave modes of a medium along given wave normals.

    Attributes:
        index: Refractive indices of the two modes (k = index d̂), shape
            (N, 2), complex. Order: s-like, p-like in the frame of d̂.
        k: Wave vectors in units of k0, global frame, shape (N, 2, 3).
        E: Electric fields, global frame, shape (N, 2, 3), flux-scaled
            (|S · d̂| = 1).
        H: Magnetic fields H' = η0 H, global frame, shape (N, 2, 3).
        S: Poynting vectors, global frame, shape (N, 2, 3).
        degenerate: True if the two indices are equal (an optic axis),
            shape (N,).
    """

    index: Array
    k: Array
    E: Array
    H: Array
    S: Array
    degenerate: Array


def plane_wave_modes(direction: Any, matrix: Any, x_ref: Any = None) -> PlaneWaveModes:
    """Return the two plane-wave modes of a medium along wave normals d̂.

    The modes are the forward eigenmodes of the frame with ẑ = d̂ and K = 0.

    Args:
        direction: Wave normals, shape (N, 3) or (3,) (normalized here).
        matrix: Constitutive matrices in the global frame, shape (N, 6, 6) or
            (6, 6).
        x_ref: Reference vectors for the frame x̂ (the s-like / p-like labels of
            a degenerate pair), shape (N, 3) or (3,). Default: the global x̂.

    Returns:
        The two forward modes along each d̂.
    """
    d_hat = _unit(be.real(_complex_vectors(direction)))
    rotation, _ = interface_frame(d_hat, d_hat, x_ref)
    m = rotate_constitutive(matrix, rotation)
    n_rows = max(m.shape[0], rotation.shape[0])
    m = be.broadcast_to(m, (n_rows, 6, 6))
    rotation = be.broadcast_to(rotation, (n_rows, 3, 3))
    modes = _modes(be.to_complex(be.zeros((n_rows,))), m)
    index = modes.q[:, :2]
    k_frame = index[:, :, None] * _c(_E3)[2][None, None, :]
    return PlaneWaveModes(
        index=index,
        k=to_global(k_frame, rotation),
        E=to_global(modes.E[:, :2, :], rotation),
        H=to_global(modes.H[:, :2, :], rotation),
        S=to_global(modes.S[:, :2, :], rotation),
        degenerate=modes.degenerate[:, 0],
    )
