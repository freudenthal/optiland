"""Tests of the interface solver (closed forms, both backends).

Closed forms: the Fresnel coefficients of the fixed ``JonesFresnel`` (isotropic
limit; the comparison with ``JonesFresnel`` itself is a cross-check of the
integration branch, because this branch does not contain the fix), Lekner 1991
Eqs. (34), (35), (42) (isotropic | uniaxial; t_se = +2 q1 A / D, the printed
sign is a misprint), and the project's oracle cases 1, 2, 6 and 10 (calcite
walk-off, the Wollaston prism, a forward mode with Re q < 0, the Glan-Taylor
surfaces).
"""

from __future__ import annotations

import math

import numpy as np
import pytest

import optiland.backend as be
from optiland.anisotropic import (
    constitutive_matrix,
    plane_wave_modes,
    solve_interface,
)
from optiland.materials import IdealMaterial
from optiland.materials.anisotropic import UniaxialMaterial

CALCITE = (1.6583434042, 1.4861300612)  # Ghosh 1999, 589.3 nm (oracle case 1)
CALCITE_HANDBOOK = (1.65835, 1.48640)  # oracle case 10 (Chipman 2019 ch. 22)


def _np(x):
    return np.asarray(be.to_numpy(x))


def _max_error(a, b):
    return float(np.max(np.abs(np.asarray(a) - np.asarray(b))))


def _six(eps):
    m = np.zeros((6, 6), dtype=complex)
    m[:3, :3] = eps
    m[3:, 3:] = np.eye(3)
    return m


def _iso(n):
    return _six(n**2 * np.eye(3))


def _uniaxial(n_o, n_e, axis):
    c = np.asarray(axis, dtype=float)
    c = c / np.linalg.norm(c)
    return _six(n_o**2 * np.eye(3) + (n_e**2 - n_o**2) * np.outer(c, c))


def _biaxial(n, rotation):
    return _six(rotation @ np.diag(np.asarray(n) ** 2) @ rotation.T)


def _random_unit(rng):
    v = rng.normal(size=3)
    return v / np.linalg.norm(v)


def _random_rotation(rng):
    q, r = np.linalg.qr(rng.normal(size=(3, 3)))
    q = q * np.sign(np.diag(r))
    return q if np.linalg.det(q) > 0 else -q


# --------------------------------------------------------------------- isotropic


def _fresnel_jones(n1, n2, theta, reflect):
    """The Jones entries (s, p) of the fixed ``JonesFresnel`` (stage 030 of the
    project: exp(-iωt), n + ik, p = k × s on each side), shape (N, 2).

    N = n2 / n1, root = the principal sqrt(N² - sin²θ);
    r_s = (cos θ - root) / (cos θ + root), r_p = (N² cos θ - root) / (N² cos θ + root),
    t_s = 2 cos θ / (cos θ + root), t_p = 2 N cos θ / (N² cos θ + root).
    """
    rel = n2 / n1
    cos = np.cos(theta)
    root = np.sqrt(rel**2 - np.sin(theta) ** 2 + 0j)
    if reflect:
        return np.stack(
            [
                (cos - root) / (cos + root),
                (rel**2 * cos - root) / (rel**2 * cos + root),
            ],
            axis=-1,
        )
    return np.stack(
        [2 * cos / (cos + root), 2 * rel * cos / (rel**2 * cos + root)], axis=-1
    )


@pytest.mark.parametrize(
    "n1, n2",
    [(1.0, 1.5), (1.5, 1.0), (1.0, 1.5 + 0.2j), (1.7, 1.2 + 0.05j)],
    ids=["external", "internal-TIR", "absorbing", "internal-absorbing"],
)
def test_isotropic_limit_equals_fresnel(set_test_backend, n1, n2):
    """Isotropic A and B: the children equal the Fresnel coefficients of the
    fixed ``JonesFresnel`` to 1E-12, at every angle, the TIR phases included.

    In the Optiland frame s is shared by the input and the outputs and
    p = k × s on each side, so E_r = J_ss s (s input) and E_r = J_pp k̂_r × s
    (p input), E_t = J_ss s and E_t = J_pp (k_t × s) / n2 (k_t complex).
    """
    theta = np.radians(np.array([0.0, 5.0, 20.0, 35.0, 41.0, 45.0, 60.0, 75.0, 89.0]))
    k_hat = np.stack([np.sin(theta), 0 * theta, np.cos(theta)], axis=-1)
    s = np.array([0.0, 1.0, 0.0])
    k_r = k_hat * np.array([1.0, 1.0, -1.0])
    j_r = _fresnel_jones(n1, n2, theta, True)
    j_t = _fresnel_jones(n1, n2, theta, False)
    for pol in ("s", "p"):
        e_in = np.tile(s, (theta.size, 1)) if pol == "s" else np.cross(k_hat, s)
        result = solve_interface([0.0, 0.0, 1.0], _iso(n1), _iso(n2), n1 * k_hat, e_in)
        e = _np(result.E)
        k_t = _np(result.k)[:, 2]
        e_r, e_t = e[:, :2].sum(axis=1), e[:, 2:].sum(axis=1)
        if pol == "s":
            exp_r = j_r[:, 0, None] * s
            exp_t = j_t[:, 0, None] * s
        else:
            exp_r = j_r[:, 1, None] * np.cross(k_r, s)
            exp_t = j_t[:, 1, None] * np.cross(k_t, s) / n2
        assert _max_error(e_r, exp_r) < 1e-12
        assert _max_error(e_t, exp_t) < 1e-12
        assert _max_error(_np(result.k)[:, 0], n1 * k_r) < 1e-14
        if np.isreal(n1) and np.isreal(n2):
            assert np.max(np.abs(_np(result.balance))) < 1e-12


def test_isotropic_prt_is_the_yun_form(set_test_backend):
    """For an isotropic interface the transmitted children sum to the Yun,
    Crabtree and Chipman PRT on transverse fields; P_j k̂_in = k̂_j; det P_j = 0."""
    n1, n2 = 1.0, 1.5
    theta = np.radians(np.array([10.0, 40.0, 70.0]))
    k_hat = np.stack([np.sin(theta), 0 * theta, np.cos(theta)], axis=-1)
    s = np.tile([0.0, 1.0, 0.0], (theta.size, 1))
    p_in = np.cross(k_hat, s)
    result = solve_interface([0.0, 0.0, 1.0], _iso(n1), _iso(n2), n1 * k_hat, s)
    prt, k = _np(result.prt), _np(result.k)
    j_t = _fresnel_jones(n1, n2, theta, False)
    k_t = k[:, 2].real / np.linalg.norm(k[:, 2].real, axis=-1, keepdims=True)
    p_t = np.cross(k_t, s)
    p_sum = prt[:, 2] + prt[:, 3]
    for i in range(theta.size):
        yun = np.column_stack([s[i], p_t[i], k_t[i]]) @ np.diag(
            [j_t[i, 0], j_t[i, 1], 1.0]
        )
        yun = yun @ np.linalg.inv(np.column_stack([s[i], p_in[i], k_hat[i]]))
        for e in (s[i], p_in[i]):
            assert _max_error(p_sum[i] @ e, yun @ e) < 1e-13
        for j in range(4):
            k_j = k[i, j].real / np.linalg.norm(k[i, j].real)
            assert _max_error(prt[i, j] @ k_hat[i], k_j) < 1e-13
            assert abs(np.linalg.det(prt[i, j])) < 1e-13


# --------------------------------------------------------------------- Lekner 1991


def _lekner(n1, theta, n_o, n_e, c):
    """Lekner 1991 Eqs. (27), (28), (34), (35), (42) (register E-12, R-27)."""
    eps_o, eps_e = n_o**2, n_e**2
    a, b, g = c
    de = eps_e - eps_o
    k_t, q1 = n1 * math.sin(theta), n1 * math.cos(theta)
    q_o = np.sqrt(eps_o - k_t**2 + 0j)
    d = eps_o * (eps_e * (eps_o + g**2 * de) - (eps_e - b**2 * de) * k_t**2)
    q_e = (np.sqrt(d + 0j) - a * g * k_t * de) / (eps_o + g**2 * de)
    e_o = np.array([-b * q_o, a * q_o - g * k_t, b * k_t])
    e_e = np.array(
        [a * q_o**2 - g * q_e * k_t, b * eps_o, g * (eps_o - q_e**2) - a * q_e * k_t]
    )
    tan = math.tan(theta)
    big_a = (q_o + q1 + k_t * tan) * e_o[0] - k_t * e_o[2]
    big_b = (q_e + q1 + k_t * tan) * e_e[0] - k_t * e_e[2]
    den = (q1 + q_e) * big_a * e_e[1] - (q1 + q_o) * big_b * e_o[1]
    q_t = q1 + k_t * tan
    return {
        "r_ss": ((q1 - q_e) * big_a * e_e[1] - (q1 - q_o) * big_b * e_o[1]) / den,
        "r_sp": 2 * n1 * (big_a * e_e[0] - big_b * e_o[0]) / den,
        "r_pp": (2 * q_t / den)
        * ((q1 + q_e) * e_o[0] * e_e[1] - (q1 + q_o) * e_e[0] * e_o[1])
        - 1,
        "r_ps": 2 * n1 * (q_e - q_o) * e_o[1] * e_e[1] / den,
        # t_se = +2 q1 A / D: Lekner prints -2 q1 A / D (a misprint, R-27).
        "t_s": (-2 * q1 * big_b / den) * e_o + (2 * q1 * big_a / den) * e_e,
        "t_p": (2 * n1 * (q1 + q_e) * e_e[1] / den) * e_o
        + (-2 * n1 * (q1 + q_o) * e_o[1] / den) * e_e,
    }


def test_lekner_isotropic_to_uniaxial(set_test_backend):
    """Reflected and transmitted fields of isotropic | uniaxial equal Lekner
    1991 for random axes, lossless and lossy, 1E-12."""
    rng = np.random.default_rng(17)
    for trial in range(30):
        c = _random_unit(rng)
        n1 = rng.uniform(1.0, 1.8)
        theta = rng.uniform(0.01, 1.4)
        n_o = rng.uniform(1.4, 2.4) + (0.03j if trial % 3 == 0 else 0.0)
        n_e = n_o + rng.uniform(-0.3, 0.3)
        lek = _lekner(n1, theta, n_o, n_e, c)
        k_hat = np.array([math.sin(theta), 0.0, math.cos(theta)])
        s = np.array([0.0, 1.0, 0.0])
        p_in = np.array([math.cos(theta), 0.0, -math.sin(theta)])
        p_r = np.array([math.cos(theta), 0.0, math.sin(theta)])
        for pol, e_in in (("s", s), ("p", p_in)):
            result = solve_interface(
                [0.0, 0.0, 1.0], _iso(n1), _uniaxial(n_o, n_e, c), n1 * k_hat, e_in
            )
            e = _np(result.E)[0]
            e_r, e_t = e[:2].sum(axis=0), e[2:].sum(axis=0)
            if pol == "s":
                assert abs(e_r @ s - lek["r_ss"]) < 1e-12
                assert abs(e_r @ p_r - lek["r_sp"]) < 1e-12
                assert _max_error(e_t, lek["t_s"]) < 1e-12
            else:
                assert abs(e_r @ p_r - lek["r_pp"]) < 1e-12
                assert abs(e_r @ s - lek["r_ps"]) < 1e-12
                assert _max_error(e_t, lek["t_p"]) < 1e-12


# --------------------------------------------------------------------- energy


def test_energy_balance_lossless(set_test_backend):
    """Anisotropic A (an incident eigenmode) and B at random normals: the child
    powers sum to 1 and R + T = 1 to 1E-12 (register E-10)."""
    rng = np.random.default_rng(23)
    for trial in range(40):
        if trial % 2:
            ma = _uniaxial(
                rng.uniform(1.5, 1.8), rng.uniform(1.5, 1.8), _random_unit(rng)
            )
        else:
            ma = _biaxial([1.62, 1.70, 1.85], _random_rotation(rng))
        mb = (
            _biaxial([1.40, 1.55, 2.0], _random_rotation(rng))
            if trial % 3
            else _iso(1.0)
        )
        normal = _random_unit(rng)
        d = _random_unit(rng)
        d = d if d @ normal > 0.3 else (d + 2 * normal) / np.linalg.norm(d + 2 * normal)
        modes = plane_wave_modes(d, ma)
        mode = trial % 4 // 2
        k_in, e_in = _np(modes.k)[0, mode], _np(modes.E)[0, mode]
        if np.real(_np(modes.S)[0, mode]) @ normal <= 0:
            continue
        result = solve_interface(normal, ma, mb, k_in, e_in)
        assert abs(_np(result.residual)[0]) < 1e-12
        assert abs(_np(result.power)[0].sum() - 1.0) < 1e-12
        assert abs(_np(result.balance)[0]) < 1e-12
        assert not _np(result.fallback)[0].any()


def test_absorbing_exit_summed_flux(set_test_backend):
    """An absorbing crystal B behind a lossless A: R + T of the summed fields is
    1 to 1E-12 (flux is continuous at the surface); the per-mode fluxes are
    reported but are not additive (R-34)."""
    rng = np.random.default_rng(29)
    for _ in range(10):
        mb = _uniaxial(1.6 + 0.2j, 1.9 + 0.05j, _random_unit(rng))
        theta = rng.uniform(0.0, 1.3)
        k_hat = np.array([math.sin(theta), 0.0, math.cos(theta)])
        e_in = np.cross(k_hat, _random_unit(rng))
        result = solve_interface([0.0, 0.0, 1.0], _iso(1.0), mb, k_hat, e_in)
        assert abs(_np(result.balance)[0]) < 1e-12
        assert np.all(np.isfinite(_np(result.power)[0]))


# --------------------------------------------------------------------- TIR


def test_total_internal_reflection_out_of_a_crystal(set_test_backend):
    """Calcite (axis in the face) into air beyond the critical angle: both
    transmitted children are evanescent (power 0, no ray) and R = 1."""
    n_o, n_e = CALCITE
    ma = _uniaxial(n_o, n_e, [0.0, 1.0, 0.0])
    theta = math.radians(50.0)
    d = np.array([math.sin(theta), 0.0, math.cos(theta)])
    modes = plane_wave_modes(d, ma)
    for mode in (0, 1):
        result = solve_interface(
            [0.0, 0.0, 1.0], ma, _iso(1.0), _np(modes.k)[0, mode], _np(modes.E)[0, mode]
        )
        power = _np(result.power)[0]
        assert np.all(_np(result.evanescent)[0, 2:])
        assert np.all(power[2:] == 0.0)
        assert np.all(np.isnan(_np(result.ray)[0, 2:]))
        assert abs(power[:2].sum() - 1.0) < 1e-12
        assert abs(_np(result.reflectance)[0] - 1.0) < 1e-12


def test_total_internal_reflection_into_a_crystal_case6(set_test_backend):
    """Oracle case 6: n1 = 2.2 at K = 1.9612731829573937 into n_o 1.5, n_e 2.6
    (axis 0.9328571428571428 rad from n̂): the o child is evanescent, the e
    child has q = -1.0440997527 and S · n̂ > 0, and the powers sum to 1."""
    phi = 0.9328571428571428
    mb = _uniaxial(1.5, 2.6, [math.sin(phi), 0.0, math.cos(phi)])
    k_t = 1.9612731829573937
    k_in = np.array([k_t, 0.0, math.sqrt(2.2**2 - k_t**2)])
    for e_in in (np.array([0.0, 1.0, 0.0]), np.cross(k_in / 2.2, [0.0, 1.0, 0.0])):
        result = solve_interface([0.0, 0.0, 1.0], _iso(2.2), mb, k_in, e_in)
        k, s = _np(result.k)[0], _np(result.S)[0]
        assert _np(result.evanescent)[0, 2] and not _np(result.evanescent)[0, 3]
        assert abs(k[3, 2] - (-1.0440997527)) < 5e-11
        assert s[3, 2] > 0 or abs(_np(result.amplitude)[0, 3]) < 1e-12
        assert abs(_np(result.power)[0].sum() - 1.0) < 1e-12


# --------------------------------------------------------------------- degeneracy


def test_optic_axis_crossing_is_continuous(set_test_backend):
    """Calcite with the axis along n̂: a sweep of the incidence angle through 0
    (the degenerate case) gives finite, continuous children, and the per-child
    powers sum to 1 to 1E-12 also next to the axis (the flux-orthogonal pair)."""
    n_o, n_e = CALCITE
    mb = _uniaxial(n_o, n_e, [0.0, 0.0, 1.0])
    angles = np.array([-1e-3, -1e-6, -1e-12, 0.0, 1e-12, 1e-6, 1e-3])
    k_in = np.stack([np.sin(angles), 0 * angles, np.cos(angles)], axis=-1)
    e_in = np.cross(k_in, np.tile([0.6, 0.8, 0.0], (angles.size, 1)))
    result = solve_interface(
        [0.0, 0.0, 1.0], _iso(1.0), mb, k_in, e_in, x_ref=[1.0, 0.0, 0.0]
    )
    e_t = _np(result.E)[:, 2:].sum(axis=1)
    power = _np(result.power)
    assert np.all(np.isfinite(e_t)) and np.all(np.isfinite(power))
    degenerate = _np(result.degenerate)[:, 2]
    assert degenerate[3] and not degenerate[0] and not degenerate[-1]
    zero = e_t[3]
    for i, a in enumerate(angles):
        assert np.linalg.norm(e_t[i] - zero) <= 2.0 * abs(a) + 1e-14
    assert np.max(np.abs(power.sum(axis=1) - 1.0)) < 1e-12


# --------------------------------------------------------------------- oracle cases


def test_calcite_displacer_walkoff_case1(set_test_backend):
    """Oracle case 1: air into calcite with the axis at 45° in the x-z plane at
    normal incidence; the e ray shifts by -0.1092064213 mm per mm (the indices
    are printed to 10 decimals: 1E-9), the o ray does not shift."""
    n_o, n_e = CALCITE
    mb = _uniaxial(n_o, n_e, [1.0, 0.0, 1.0])
    for e_in in ([1.0, 0.0, 0.0], [0.0, 1.0, 0.0]):
        result = solve_interface([0.0, 0.0, 1.0], _iso(1.0), mb, [0.0, 0.0, 1.0], e_in)
        power = _np(result.power)[0]
        ray = _np(result.ray)[0]
        child = 2 + int(np.argmax(power[2:]))
        shift = ray[child, 0] / ray[child, 2]
        if e_in[0]:  # x: the e mode
            assert abs(shift - (-0.1092064213)) < 1e-9
            assert (
                abs(math.degrees(_np(result.walkoff)[0, child]) - 6.2323695095) < 5e-9
            )
        else:
            assert abs(shift) < 1e-15


def _glan_taylor_step(ma, mb, normal, k_in, e_in):
    result = solve_interface(normal, ma, mb, k_in, e_in)
    power = _np(result.power)[0]
    j = 2 + int(np.argmax(power[2:]))
    e_t = _np(result.E)[0, j]
    return (
        result,
        j,
        np.linalg.norm(e_t) / np.linalg.norm(e_in),
        power[j],
        _np(result.k)[0, j],
        e_t,
    )


def test_glan_taylor_case10(set_test_backend):
    """Oracle case 10 (Chipman 2019 ch. 22): calcite 1.65835 / 1.48640, axis ŷ in
    both prisms, air gap normal (0, -sin 40°, cos 40°), normal incidence.

    The y (e) path: field ratios 0.8043758, 1.8901704, 0.4900140, 1.1956242 and
    powers 0.9617312, 0.9262100, 0.9262100, 0.9617312 (to half the last printed
    digit, 5E-8); the x (o) path is totally reflected at the gap.
    """
    n_o, n_e = CALCITE_HANDBOOK
    crystal = _uniaxial(n_o, n_e, [0.0, 1.0, 0.0])
    air = _iso(1.0)
    gap = np.array([0.0, -math.sin(math.radians(40.0)), math.cos(math.radians(40.0))])
    z = np.array([0.0, 0.0, 1.0])
    ratios, powers = [], []
    k, e = z, np.array([0.0, 1.0, 0.0])
    for ma, mb, normal in (
        (air, crystal, z),
        (crystal, air, gap),
        (air, crystal, gap),
        (crystal, air, z),
    ):
        _, _, ratio, power, k, e = _glan_taylor_step(ma, mb, normal, k, e)
        ratios.append(ratio)
        powers.append(power)
    assert _max_error(ratios, [0.8043758, 1.8901704, 0.4900140, 1.1956242]) < 5e-8
    assert _max_error(powers, [0.9617312, 0.9262100, 0.9262100, 0.9617312]) < 5e-8
    assert abs(np.prod(ratios) - 0.8907650) < 5e-8
    assert abs(np.prod(powers) - 0.7934623) < 5e-8

    # The x (o) path: TIR at the gap.
    result, *_ = _glan_taylor_step(air, crystal, z, z, np.array([1.0, 0.0, 0.0]))
    j = 2 + int(np.argmax(_np(result.power)[0, 2:]))
    k_o, e_o = _np(result.k)[0, j], _np(result.E)[0, j]
    result = solve_interface(gap, crystal, air, k_o, e_o)
    assert np.all(_np(result.evanescent)[0, 2:])
    assert abs(_np(result.reflectance)[0] - 1.0) < 1e-12


# --------------------------------------------------------------------- consistency


def test_same_medium_transmits_unchanged(set_test_backend):
    """A = B (a rotated biaxial crystal): no reflection, the incident mode
    passes with amplitude 1 into the same mode."""
    rng = np.random.default_rng(31)
    for _ in range(10):
        m = _biaxial([1.62, 1.70, 1.85], _random_rotation(rng))
        normal = _random_unit(rng)
        d = (_random_unit(rng) + 2 * normal) / 3.0
        modes = plane_wave_modes(d, m)
        k_in, e_in = _np(modes.k)[0, 0], _np(modes.E)[0, 0]
        result = solve_interface(normal, m, m, k_in, e_in)
        e = _np(result.E)[0]
        assert _max_error(e[:2], 0.0) < 1e-12
        assert _max_error(e[2:].sum(axis=0), e_in) < 1e-12


def test_residual_of_a_field_outside_the_modes(set_test_backend):
    """A field along k̂_in in an isotropic A is not a mode: ``residual`` shows it."""
    result = solve_interface(
        [0.0, 0.0, 1.0], _iso(1.0), _iso(1.5), [0.0, 0.0, 1.0], [0.0, 0.0, 1.0]
    )
    assert abs(_np(result.residual)[0] - 1.0) < 1e-15


def test_constitutive_matrix_of_materials(set_test_backend):
    """A scalar material gives (n + ik)² I ⊕ I; a tensor material gives
    ``constitutive_6x6``."""
    m = _np(constitutive_matrix(IdealMaterial(n=1.5, k=0.1), [0.5, 0.6]))
    assert m.shape == (2, 6, 6)
    assert _max_error(m[0], _six((1.5 + 0.1j) ** 2 * np.eye(3))) < 1e-15
    crystal = UniaxialMaterial(
        IdealMaterial(n=1.6), IdealMaterial(n=1.4), optic_axis=(0.0, 1.0, 1.0)
    )
    expected = _np(crystal.constitutive_6x6(0.5))
    assert _max_error(_np(constitutive_matrix(crystal, 0.5)), expected) < 1e-15


def test_torch_gradient_of_transmitted_power():
    """Torch: the derivative of a weighted sum of the transmitted powers with
    respect to ε_e equals a central finite difference to 1E-6 relative, away
    from the degeneracy."""
    torch = pytest.importorskip("torch")
    be.set_backend("torch")
    be.set_precision("float64")
    try:
        c = torch.tensor([0.3, 0.2, 0.9], dtype=torch.float64)
        c = (c / torch.linalg.norm(c)).to(torch.complex128)
        eye = torch.eye(3, dtype=torch.complex128)
        theta = math.radians(35.0)
        k_in = [math.sin(theta), 0.0, math.cos(theta)]
        e_in = [math.cos(theta), 0.3, -math.sin(theta)]

        def power(eps_e):
            eps = 2.25 * eye + (eps_e.to(torch.complex128) - 2.25) * torch.outer(c, c)
            result = solve_interface(
                [0.0, 0.0, 1.0], _iso(1.0), torch.block_diag(eps, eye), k_in, e_in
            )
            return result.power[0, 2] + 2.0 * result.power[0, 3]

        x = torch.tensor(2.89, dtype=torch.float64, requires_grad=True)
        (grad,) = torch.autograd.grad(power(x), x)
        h = 1e-4  # the FD rounding noise is 1E-15 / h relative to a derivative of 3E-4
        up = power(torch.tensor(2.89 + h, dtype=torch.float64))
        down = power(torch.tensor(2.89 - h, dtype=torch.float64))
        fd = ((up - down) / (2 * h)).item()
        assert abs(grad.item() - fd) < 1e-6 * abs(fd)
    finally:
        be.set_backend("numpy")


def test_wollaston_case2(set_test_backend):
    """Oracle case 2 (register E-19): a calcite Wollaston prism, wedge 20°, at
    normal incidence. Prism 1 axis x̂, prism 2 axis ŷ, internal face normal
    (sin 20°, 0, cos 20°), exit face ẑ into air. An anisotropic A and an
    anisotropic B at a tilted face.

    y-pol (o → e) leaves at -3.6217511833°, x-pol (e → o) at +3.5692606184°
    (printed to 10 decimals from indices printed to 10 decimals: 1E-8). The
    Snell chain n1 sin a = n2 sin b, exit angle arcsin(n2 sin(a - b)), with the
    same indices: 1E-12.
    """
    n_o, n_e = CALCITE
    prism1 = _uniaxial(n_o, n_e, [1.0, 0.0, 0.0])
    prism2 = _uniaxial(n_o, n_e, [0.0, 1.0, 0.0])
    a = math.radians(20.0)
    face = np.array([math.sin(a), 0.0, math.cos(a)])
    z = np.array([0.0, 0.0, 1.0])
    expected = {"y": -3.6217511833, "x": 3.5692606184}
    for pol, (n1, n2) in (("y", (n_o, n_e)), ("x", (n_e, n_o))):
        e_in = np.array([0.0, 1.0, 0.0]) if pol == "y" else np.array([1.0, 0.0, 0.0])
        modes = plane_wave_modes(z, prism1)
        j = int(np.argmax(np.abs(_np(modes.E)[0] @ e_in)))
        k, e = _np(modes.k)[0, j], _np(modes.E)[0, j]
        for ma, mb, normal in ((prism1, prism2, face), (prism2, _iso(1.0), z)):
            result = solve_interface(normal, ma, mb, k, e)
            power = _np(result.power)[0]
            child = 2 + int(np.argmax(power[2:]))
            k, e = _np(result.k)[0, child], _np(result.E)[0, child]
        angle = math.degrees(math.atan2(k[0].real, k[2].real))
        b = math.asin(n1 * math.sin(a) / n2)
        snell = math.degrees(math.asin(n2 * math.sin(a - b)))
        assert abs(angle - snell) < 1e-12
        assert abs(angle - expected[pol]) < 1e-8
