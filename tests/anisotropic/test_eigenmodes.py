"""Tests of the eigenmodes of homogeneous media (closed forms, both backends).

The closed forms are from J. Lekner, J. Phys.: Condens. Matter 3, 6121 (1991)
(Eqs. 20-31) and Born and Wolf ch. 15 (the uniaxial and biaxial indices).
"""

from __future__ import annotations

import math

import numpy as np
import pytest

import optiland.backend as be
from optiland.anisotropic import (
    berreman_delta,
    medium_modes,
    plane_wave_modes,
)

from ..utils import assert_allclose

# Calcite, Ghosh 1999 at 589.3 nm.
CALCITE = (1.6583434042, 1.4861300612)
# KTP, Kato and Takaoka 2002 at 632.8 nm.
KTP = (1.7619723936, 1.7712901659, 1.8648041298)
KTP_BINORMAL_DEG = 18.2279131512


def _np(x):
    return np.asarray(be.to_numpy(x))


def _max_error(a, b):
    """Return max |a - b| of complex arrays."""
    return float(np.max(np.abs(np.asarray(a) - np.asarray(b))))


def _six(eps, mu=None, xi=None, zeta=None):
    m = np.zeros((6, 6), dtype=complex)
    m[:3, :3] = eps
    m[3:, 3:] = np.eye(3) if mu is None else mu
    if xi is not None:
        m[:3, 3:] = xi
    if zeta is not None:
        m[3:, :3] = zeta
    return m


def _uniaxial(n_o, n_e, axis):
    c = np.asarray(axis, dtype=float)
    c = c / np.linalg.norm(c)
    return n_o**2 * np.eye(3) + (n_e**2 - n_o**2) * np.outer(c, c)


def _random_axis(rng):
    v = rng.normal(size=3)
    return v / np.linalg.norm(v)


def _maxwell_residual(k, e, h, m):
    """Relative residual of k × E = ζE + μH' and k × H' = -(εE + ξH')."""
    eps, xi, zeta, mu = m[:3, :3], m[:3, 3:], m[3:, :3], m[3:, 3:]
    r1 = np.cross(k, e) - (zeta @ e + mu @ h)
    r2 = np.cross(k, h) + (eps @ e + xi @ h)
    scale = np.linalg.norm(e) + np.linalg.norm(h)
    return (np.linalg.norm(r1) + np.linalg.norm(r2)) / scale


# --------------------------------------------------------------------- Δ


def test_modes_solve_maxwell_random_bianisotropic(set_test_backend):
    """Each mode of Δ solves the plane-wave Maxwell system (Berreman 1972).

    Random complex [[ε, ξ], [ζ, μ]] and K; relative residual 1E-12.
    """
    rng = np.random.default_rng(7)
    for _ in range(10):
        a = rng.normal(size=(6, 6)) + 1j * rng.normal(size=(6, 6))
        m = 0.3 * a + np.diag([2.5, 2.2, 2.8, 1.2, 1.1, 1.3])
        k_t = rng.uniform(0.0, 1.0)
        modes = medium_modes(k_t, m)
        q, e, h = _np(modes.q)[0], _np(modes.E)[0], _np(modes.H)[0]
        delta = _np(berreman_delta(k_t, m))[0]
        assert (
            _max_error(np.sort_complex(q), np.sort_complex(np.linalg.eigvals(delta)))
            < 1e-12
        )
        for j in range(4):
            k = np.array([k_t, 0.0, q[j]])
            assert _maxwell_residual(k, e[j], h[j], m) < 1e-12


def test_isotropic_closed_form(set_test_backend):
    """An isotropic medium: q = ±sqrt(n² - K²), s = E_y only, p = E_x, E_z only,
    flux |S_z| = 1, the pair flagged degenerate (lossless and lossy)."""
    for n in (1.5, 1.5 + 0.1j):
        k_t = 0.6
        modes = medium_modes(k_t, _six(n**2 * np.eye(3)))
        q = np.sqrt(n**2 - k_t**2)
        assert _max_error(_np(modes.q)[0], [q, q, -q, -q]) < 1e-15
        e = _np(modes.E)[0]
        assert np.all(np.abs(e[[0, 2], 0]) < 1e-15)  # s: E_x = 0
        assert np.all(np.abs(e[[0, 2], 2]) < 1e-15)  # s: E_z = 0
        assert np.all(np.abs(e[[1, 3], 1]) < 1e-15)  # p: E_y = 0
        assert_allclose(np.abs(_np(modes.S)[0, :, 2]), np.ones(4), rtol=0, atol=1e-14)
        assert np.all(_np(modes.degenerate)[0])
        assert bool(_np(modes.isotropic)[0])
        assert not bool(_np(modes.fallback)[0])


def test_phase_rule_and_flux_scale(set_test_backend):
    """s-like columns have E_y real >= 0, p-like columns H'_y real >= 0; each
    propagating mode has |S_z| = 1; an evanescent mode has |ψ| = 1."""
    rng = np.random.default_rng(3)
    eps = _uniaxial(1.6, 1.9, _random_axis(rng))
    for k_t in (0.4, 1.7):  # 1.7 > n_o: the ordinary pair is evanescent
        modes = medium_modes(k_t, _six(eps))
        psi = _np(modes.psi)[0]
        ey, hy = psi[1, [0, 2]], psi[3, [1, 3]]
        assert np.all(np.abs(ey.imag) < 1e-15) and np.all(ey.real >= 0)
        assert np.all(np.abs(hy.imag) < 1e-15) and np.all(hy.real >= 0)
        evanescent = _np(modes.evanescent)[0]
        flux = np.abs(_np(modes.S)[0, :, 2])
        norm = np.linalg.norm(psi, axis=0)
        assert_allclose(flux[~evanescent], 1.0, rtol=0, atol=1e-13)
        assert_allclose(norm[evanescent], 1.0, rtol=0, atol=1e-13)
    assert evanescent.sum() == 2


# --------------------------------------------------------------------- uniaxial


def _lekner_q(k_t, eps_o, eps_e, c):
    """Lekner 1991 Eqs. (21), (23): the forward q_o and q_e."""
    a, b, g = c
    de = eps_e - eps_o
    q_o = np.sqrt(eps_o - k_t**2 + 0j)
    d = eps_o * (eps_e * (eps_o + g**2 * de) - (eps_e - b**2 * de) * k_t**2)
    q_e = (np.sqrt(d + 0j) - a * g * k_t * de) / (eps_o + g**2 * de)
    return q_o, q_e


def test_uniaxial_q_closed_form(set_test_backend):
    """The forward q of a uniaxial medium equal Lekner's closed forms (Eqs.
    20-24) for random axes, lossless and lossy; the ordinary mode has E ⊥ ĉ."""
    rng = np.random.default_rng(11)
    for trial in range(40):
        n_o = rng.uniform(1.4, 2.4) + (0.02j if trial % 3 == 0 else 0.0)
        n_e = n_o + rng.uniform(-0.4, 0.4)
        c = _random_axis(rng)
        k_t = rng.uniform(0.0, 1.3)
        modes = medium_modes(k_t, _six(_uniaxial(n_o, n_e, c)))
        q, e = _np(modes.q)[0], _np(modes.E)[0]
        q_o, q_e = _lekner_q(k_t, n_o**2, n_e**2, c)
        ordinary = int(
            np.argmin([abs(e[j] @ c) / np.linalg.norm(e[j]) for j in (0, 1)])
        )
        assert abs(q[ordinary] - q_o) < 1e-13
        assert abs(q[1 - ordinary] - q_e) < 1e-13


def test_uniaxial_fields(set_test_backend):
    """k · D = 0 for every mode; E_o ⊥ ĉ; S_o ∥ k_o and S_e ∥ ε k_e for a
    lossless crystal (Lekner 1991 Eqs. 30, 31)."""
    rng = np.random.default_rng(5)
    for _ in range(20):
        c = _random_axis(rng)
        eps = _uniaxial(1.65, 1.49, c)
        k_t = rng.uniform(0.0, 1.2)
        modes = medium_modes(k_t, _six(eps))
        q, e, s = _np(modes.q)[0], _np(modes.E)[0], _np(modes.S)[0]
        for j in range(4):
            k = np.array([k_t, 0.0, q[j]])
            d = eps @ e[j]
            assert abs(k @ d) < 1e-13 * np.linalg.norm(d)
            s_hat = s[j] / np.linalg.norm(s[j])
            if abs(e[j] @ c) < 1e-12 * np.linalg.norm(e[j]):  # ordinary
                direction = np.real(k)
            else:
                direction = np.real(eps @ k)
            direction = direction / np.linalg.norm(direction)
            assert np.linalg.norm(np.cross(s_hat, direction)) < 1e-13


def test_forward_mode_with_negative_re_q(set_test_backend):
    """A forward mode with Re q < 0: n_o 1.5, n_e 2.6, axis at
    0.9328571428571428 rad from n̂ in the plane of incidence, K = 1.9612731829573937.

    The forward e mode has q = -1.0440997527 (printed to 10 decimals) and
    S_z > 0; the forward o mode is evanescent (K > n_o). A Re q sort fails here.
    """
    phi = 0.9328571428571428
    c = np.array([math.sin(phi), 0.0, math.cos(phi)])
    modes = medium_modes(1.9612731829573937, _six(_uniaxial(1.5, 2.6, c)))
    q, s = _np(modes.q)[0], _np(modes.S)[0]
    evanescent = _np(modes.evanescent)[0]
    assert evanescent[0] and not evanescent[1]  # f1 = o (s-like), f2 = e
    assert q[0].imag > 0
    assert abs(q[1] - (-1.0440997527)) < 5e-11
    assert s[1, 2] > 0
    assert q[1].real < q[2].real  # the backward e mode has the larger Re q


def test_absorbing_forward_modes_decay(set_test_backend):
    """In an absorbing crystal the two forward modes have Im q > 0."""
    rng = np.random.default_rng(2)
    for _ in range(10):
        eps = _uniaxial(1.6 + 0.05j, 1.8 + 0.01j, _random_axis(rng))
        q = _np(medium_modes(rng.uniform(0, 1.2), _six(eps)).q)[0]
        assert np.all(q[:2].imag > 0) and np.all(q[2:].imag < 0)


# --------------------------------------------------------------------- degeneracy


def test_degenerate_flags(set_test_backend):
    """Degenerate pairs: the calcite optic axis and the KTP
    binormal are flagged; 1E-3 rad off them and active quartz are not."""
    n_o, n_e = CALCITE
    axis = _six(_uniaxial(n_o, n_e, [0.0, 0.0, 1.0]))
    assert bool(_np(medium_modes(0.0, axis).degenerate)[0, 0])
    tilted = _six(_uniaxial(n_o, n_e, [math.sin(1e-3), 0.0, math.cos(1e-3)]))
    assert not bool(_np(medium_modes(0.0, tilted).degenerate)[0, 0])

    n_a, n_b, n_g = KTP
    ktp = _six(np.diag([n_a**2, n_b**2, n_g**2]))
    v = math.radians(KTP_BINORMAL_DEG)
    on = plane_wave_modes([math.sin(v), 0.0, math.cos(v)], ktp)
    assert bool(_np(on.degenerate)[0])
    assert_allclose(_np(on.index)[0], [n_b, n_b], rtol=0, atol=1e-9)
    off = plane_wave_modes([math.sin(v + 1e-3), 0.0, math.cos(v + 1e-3)], ktp)
    assert not bool(_np(off.degenerate)[0])


def test_active_quartz_on_axis(set_test_backend):
    """Quartz in the Tellegen form along its axis: the modes
    are not degenerate and n = sqrt(n_o² + κ²) ± κ exactly; the circular
    modes tie in |E_y|, so the larger index is first."""
    n_o, n_e, kappa, ratio = 1.5442057388, 1.5533, 3.5574359e-05, -0.525
    alpha = np.diag([kappa, kappa, (2 * ratio - 1) * kappa])
    eps = np.diag([n_o**2, n_o**2, n_e**2]) + alpha @ alpha.T
    m = _six(eps, xi=1j * alpha, zeta=-1j * alpha.T)
    modes = plane_wave_modes([0.0, 0.0, 1.0], m)
    root = math.sqrt(n_o**2 + kappa**2)
    assert_allclose(
        _np(modes.index)[0], [root + kappa, root - kappa], rtol=0, atol=1e-14
    )
    assert not bool(_np(modes.degenerate)[0])


# --------------------------------------------------------------------- plane-wave modes


def test_plane_wave_modes_uniaxial_indices(set_test_backend):
    """Along a wave normal at θ from the axis: n_o and n_e(θ) with
    1/n_e(θ)² = cos²θ/n_o² + sin²θ/n_e²."""
    n_o, n_e = CALCITE
    c = np.array([0.0, 0.0, 1.0])
    m = _six(_uniaxial(n_o, n_e, c))
    for theta in (0.2, 0.7, 1.2, math.pi / 2):
        d = np.array([math.sin(theta), 0.0, math.cos(theta)])
        index = np.sort(_np(plane_wave_modes(d, m).index)[0].real)
        n_theta = 1.0 / math.sqrt(
            math.cos(theta) ** 2 / n_o**2 + math.sin(theta) ** 2 / n_e**2
        )
        assert_allclose(index, np.sort([n_o, n_theta]), rtol=0, atol=1e-14)


def test_plane_wave_modes_biaxial_fresnel_equation(set_test_backend):
    """KTP along random wave normals: both indices solve Fresnel's equation of
    wave normals Σ u_i² n_i² Π_(j≠i) (n² - n_j²) = 0."""
    rng = np.random.default_rng(13)
    ni = np.array(KTP)
    m = _six(np.diag(ni**2))
    for _ in range(20):
        u = _random_axis(rng)

        def fresnel(n, u=u):
            return sum(
                u[i] ** 2
                * ni[i] ** 2
                * np.prod([n**2 - ni[j] ** 2 for j in range(3) if j != i])
                for i in range(3)
            )

        index = _np(plane_wave_modes(u, m).index)[0].real
        for n in index:
            # The residual as an index error: |F(n)| / |F'(n)| <= 1E-14.
            h = 1e-6
            slope = (fresnel(n + h) - fresnel(n - h)) / (2 * h)
            assert abs(fresnel(n)) / abs(slope) < 1e-14


def test_walkoff_calcite_45(set_test_backend):
    """The e ray of calcite with k at 45° from the axis walks off
    by 6.2323695095° (tan ρ = tan θ (n_e² - n_o²)/(n_e² + n_o² tan² θ)).

    The indices are printed to 10 decimals, which moves ρ by up to 2E-9°: the
    printed ρ is checked to 5E-9°, the closed form with the same indices to
    1E-12°."""
    n_o, n_e = CALCITE
    m = _six(_uniaxial(n_o, n_e, [1.0, 0.0, 1.0]))
    modes = plane_wave_modes([0.0, 0.0, 1.0], m)
    s, k = _np(modes.S)[0], _np(modes.k)[0].real
    rho = [
        math.degrees(math.atan2(np.linalg.norm(np.cross(k[j], s[j])), k[j] @ s[j]))
        for j in (0, 1)
    ]
    assert min(rho) < 1e-12
    assert abs(max(rho) - 6.2323695095) < 5e-9
    t = 1.0
    closed = math.degrees(
        math.atan(abs(t * (n_e**2 - n_o**2) / (n_e**2 + n_o**2 * t**2)))
    )
    assert abs(max(rho) - closed) < 1e-12


# --------------------------------------------------------------------- gradients


def test_torch_gradient_of_extraordinary_q():
    """Torch: dq_e/dε_e of the forward e mode equals a central finite
    difference to 1E-7 relative (away from the degeneracy)."""
    torch = pytest.importorskip("torch")
    be.set_backend("torch")
    be.set_precision("float64")
    try:
        c = np.array([0.3, 0.2, 0.9])
        c = c / np.linalg.norm(c)
        cc = torch.tensor(np.outer(c, c), dtype=torch.complex128)
        c_t = torch.tensor(c, dtype=torch.complex128)
        eye = torch.eye(3, dtype=torch.complex128)

        def q_e(eps_e):
            eps = 2.25 * eye + (eps_e.to(torch.complex128) - 2.25) * cc
            modes = medium_modes(0.5, torch.block_diag(eps, eye))
            j = int(torch.argmax(torch.abs(modes.E[0, :2, :] @ c_t)))  # E not ⊥ ĉ
            return modes.q[0, j].real

        x = torch.tensor(2.89, dtype=torch.float64, requires_grad=True)
        (grad,) = torch.autograd.grad(q_e(x), x)
        h = 1e-6
        up = q_e(torch.tensor(2.89 + h, dtype=torch.float64))
        down = q_e(torch.tensor(2.89 - h, dtype=torch.float64))
        fd = (up - down) / (2 * h)
        assert abs(grad.item() - fd.item()) < 1e-7 * abs(fd.item())
    finally:
        be.set_backend("numpy")
