"""Tests of ray splitting by branch enumeration (both backends).

Closed forms: the calcite beam displacer (two branches, powers cos² and sin² of the
input azimuth times the face transmittances), the Wollaston prism (exit
angles; the branch powers add to the traced power), the Savart plate
(shear and the fringe phase of the coherent sum), the calcite
Glan-Taylor (one branch survives, P_yy and its power), and a glass
plate with ghosts (the geometric series of R and T). The power ledger of a
lossless system closes to 1E-12.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

import optiland.backend as be
from optiland.materials import IdealMaterial
from optiland.optic import Optic
from optiland.rays import AnisotropicRays, PolarizationState
from optiland.raytrace.branches import BranchTracer, PowerLedger, ray_power

from .test_tracing import (
    CALCITE,
    CALCITE_HANDBOOK,
    WL,
    _calcite,
    _crystal_optic,
    _np,
    _rays,
    _state,
)

S45 = math.sin(math.pi / 4)


def _closed(ledger: PowerLedger, tol: float = 1e-12) -> None:
    """A lossless ledger: nothing absorbed and the fields add to 1."""
    assert abs(ledger.absorbed) < tol
    assert abs(ledger.total() - 1.0) < tol


def _face(n: float) -> float:
    """The power transmittance 4n / (1 + n)² of a face at normal incidence."""
    return 4 * n / (1 + n) ** 2


def _fresnel_r(n: float, theta: float) -> tuple[float, float]:
    """R_s, R_p from air into index n at the angle theta."""
    c = math.cos(theta)
    root = math.sqrt(n**2 - math.sin(theta) ** 2)
    rs = (c - root) / (c + root)
    rp = (n**2 * c - root) / (n**2 * c + root)
    return rs**2, rp**2


def _exit(branch):
    """Exit point, direction and OPL of ray 0 of a branch."""
    r = branch.rays
    x = np.array([_np(r.x)[0], _np(r.y)[0], _np(r.z)[0]])
    d = np.array([_np(r.L)[0], _np(r.M)[0], _np(r.N)[0]])
    return x, d, _np(r.opd)[0]


# ------------------------------------------------------------- displacer


@pytest.mark.parametrize("azimuth", [0.3, 1.1])
def test_displacer_two_branches(set_test_backend, azimuth):
    """The e branch carries cos²a, the o branch sin²a of the input
    (x is the e field of the 45° cut), each times the transmittance of its two
    faces; the e ray walks off by -0.1092064213 d; the reflections that are
    not followed close the ledger."""
    axis = (S45, 0.0, S45)
    optic = _crystal_optic([_calcite(axis)], [1.0])
    optic.updater.set_polarization(_state(math.cos(azimuth), math.sin(azimuth)))
    result = BranchTracer(optic).trace_all(rays=_rays())
    o_key = (("s1", "T", "o"), ("s2", "T"))
    e_key = (("s1", "T", "e"), ("s2", "T"))
    assert result.keys() == [o_key, e_key]
    n_o, n_e = CALCITE
    n_45 = n_o * n_e / math.sqrt((n_o**2 + n_e**2) / 2)
    c2, s2 = math.cos(azimuth) ** 2, math.sin(azimuth) ** 2
    assert abs(result[e_key].power - c2 * _face(n_45) ** 2) < 1e-12
    assert abs(result[o_key].power - s2 * _face(n_o) ** 2) < 1e-12
    assert abs(_np(result[e_key].rays.x)[0] - (-0.1092064213)) < 1e-9
    assert abs(_np(result[o_key].rays.x)[0]) < 1e-15
    # The reflected power at the two faces (not followed without ghosts).
    r_o, r_e = 1 - _face(n_o), 1 - _face(n_45)
    unfollowed = s2 * (r_o + _face(n_o) * r_o) + c2 * (r_e + _face(n_45) * r_e)
    assert abs(result.ledger.unfollowed - unfollowed) < 1e-12
    assert result.ledger.kept == pytest.approx(
        result[e_key].power + result[o_key].power
    )
    _closed(result.ledger)
    # Each branch equals the fixed-mode trace of that path.
    for key in (o_key, e_key):
        rays = BranchTracer(optic).sequence(key).trace_generic(0.0, 0.0, 0.0, 0.0, WL)
        rays.update_intensity(optic.polarization_state)
        assert abs(_np(rays.i)[0] - result[key].power) < 1e-12
        assert abs(_np(rays.x)[0] - _np(result[key].rays.x)[0]) < 1e-12


def test_displacer_with_ghosts_is_the_plate_series(set_test_backend):
    """With every reflection followed (threshold 0), the e and o powers of
    each round trip are the incoherent plate series T² R^(2m) of each mode (the
    faces of a normal-incidence plate do not couple o and e: the e-to-o
    branches carry no power and are pruned)."""
    axis = (S45, 0.0, S45)
    optic = _crystal_optic([_calcite(axis)], [1.0])
    optic.updater.set_polarization(_state(math.cos(0.3), math.sin(0.3)))
    result = BranchTracer(
        optic, ghosts="all", threshold=0.0, max_reflections=4
    ).trace_all(rays=_rays())
    n_o, n_e = CALCITE
    n_45 = n_o * n_e / math.sqrt((n_o**2 + n_e**2) / 2)
    for mode, n, weight in (
        ("o", n_o, math.sin(0.3) ** 2),
        ("e", n_45, math.cos(0.3) ** 2),
    ):
        t, r = _face(n), 1 - _face(n)
        for m in range(3):
            key = (("s1", "T", mode),)
            for _ in range(m):
                key += (("s2", "R", mode), ("s1", "R", mode))
            key += (("s2", "T"),)
            assert abs(result[key].power - weight * t**2 * r ** (2 * m)) < 1e-12
    assert result.ledger.returned > 0.0 and result.ledger.pruned >= 0.0
    _closed(result.ledger)


# ------------------------------------------------------------- Wollaston


@pytest.mark.parametrize(
    "theta, plane, x_exit, y_exit",
    [
        (0.0, "xz", 3.5692606184, -3.6217511833),  # 10 digits
        (3.0, "yz", 3.5761302, -3.6249192),  # 7 digits
        (3.0, "xz", 6.5379644, -0.5761934),
    ],
)
def test_wollaston_branches(set_test_backend, theta, plane, x_exit, y_exit):
    """The x field (e -> o) and the y field (o -> e) leave at
    the Snell-chain angles (measured in the x-z plane); the powers of the two main
    branches and the unfollowed reflections close the ledger."""
    media = [_calcite((1.0, 0.0, 0.0)), _calcite((0.0, 1.0, 0.0))]
    tilts = [0.0, math.radians(20.0), 0.0]
    optic = _crystal_optic(media, [2.0, 2.0], tilts=tilts)
    optic.updater.set_polarization(_state(S45, S45))
    t = math.radians(theta)
    if plane == "xz":
        rays = _rays(t)
    else:  # a ray in the y-z plane that reaches the origin
        rays = AnisotropicRays(
            *(be.array([v]) for v in (0.0, -math.tan(t), -1.0)),
            *(be.array([v]) for v in (0.0, math.sin(t), math.cos(t))),
            be.array([1.0]),
            be.array([WL]),
        )
    result = BranchTracer(optic).trace_all(rays=rays)
    x_key = (("s1", "T", "e"), ("s2", "T", "o"), ("s3", "T"))
    y_key = (("s1", "T", "o"), ("s2", "T", "e"), ("s3", "T"))
    tol = 1e-8 if theta == 0.0 else 5e-8
    for key, angle in ((x_key, x_exit), (y_key, y_exit)):
        _, d, _ = _exit(result[key])
        assert abs(math.degrees(math.atan2(d[0], d[2])) - angle) < tol
    # At normal incidence the o -> o and e -> e branches carry no power.
    if theta == 0.0:
        assert set(result.keys()) == {x_key, y_key}
    _closed(result.ledger)


# ------------------------------------------------------------- Savart


def test_savart_normal_incidence(set_test_backend):
    """The beams e1 -> o2 and o1 -> e2 are displaced by
    (-0.1092064213, 0) and (0, -0.1092064213) mm, separation 0.1544412022 mm,
    zero OPD; the o -> o and e -> e branches carry no power."""
    media = [_calcite((S45, 0.0, S45)), _calcite((0.0, S45, S45))]
    optic = _crystal_optic(media, [1.0, 1.0])
    optic.updater.set_polarization(_state(S45, S45))
    result = BranchTracer(optic).trace_all(rays=_rays())
    eo = (("s1", "T", "e"), ("s2", "T", "o"), ("s3", "T"))
    oe = (("s1", "T", "o"), ("s2", "T", "e"), ("s3", "T"))
    assert set(result.keys()) == {eo, oe}
    (xa, da, oa), (xb, _, ob) = _exit(result[eo]), _exit(result[oe])
    assert np.max(np.abs(xa[:2] - [-0.1092064213, 0.0])) < 1e-9
    assert np.max(np.abs(xb[:2] - [0.0, -0.1092064213])) < 1e-9
    assert abs(np.linalg.norm(xa - xb) - 0.1544412022) < 1e-9
    assert abs(oa - ob - float(np.dot(da, xa - xb))) < 1e-12
    # Orthogonal fields: the coherent power is the incoherent one.
    coherent = _np(result.coherent_intensity([eo, oe]))
    incoherent = _np(result.incoherent_intensity([eo, oe]))
    assert abs(coherent[0] - incoherent[0]) < 1e-12
    _closed(result.ledger)


def test_savart_fringe(set_test_backend):
    """Zhang, Ren and Mu 2010 at 3°, t = 6 mm: OPD 0.0348725 mm and
    shear 0.9418175 mm (7 digits). Behind an analyzer at 45° the coherent sum
    of the two branches is P_a + P_b + 2 √(P_a P_b) cos φ, φ = k0 OPD."""
    media = [
        _calcite((S45, 0.0, S45), CALCITE_HANDBOOK),
        _calcite((0.0, S45, S45), CALCITE_HANDBOOK),
    ]
    optic = _crystal_optic(media, [6.0, 6.0])
    optic.updater.set_polarization(_state(S45, S45))
    result = BranchTracer(optic).trace_all(rays=_rays(math.radians(3.0)))
    eo = (("s1", "T", "e"), ("s2", "T", "o"), ("s3", "T"))
    oe = (("s1", "T", "o"), ("s2", "T", "e"), ("s3", "T"))
    (xa, da, oa), (xb, _, ob) = _exit(result[eo]), _exit(result[oe])
    opd = oa - ob - float(np.dot(da, xa - xb))
    assert abs(opd - 0.0348725) < 5e-8
    shear = np.linalg.norm((xa - xb) - np.dot(xa - xb, da) * da)
    assert abs(shear - 0.9418175) < 5e-8

    analyzer = np.array([S45, S45, 0.0])
    k0 = 2 * math.pi / (WL * 1e-3)
    amplitudes = []
    for key, opl in ((eo, oa), (oe, ob)):
        field = _np(result.exit_fields(key)[0])[0] * np.exp(-1j * k0 * opl)
        assert np.max(np.abs(field.imag)) < 1e-12  # the path phase is k0 OPL
        amplitudes.append(float(analyzer @ field.real))
    a, b = amplitudes
    expected = a**2 + b**2 + 2 * a * b * math.cos(k0 * opd)
    coherent = _np(result.coherent_intensity([eo, oe], analyzer=analyzer))[0]
    assert abs(coherent - expected) < 1e-9
    # The fringe phase from the case table (7 digits: 5E-4 rad).
    closed = a**2 + b**2 + 2 * a * b * math.cos(k0 * 0.0348725)
    assert abs(coherent - closed) < 2 * abs(a * b) * 6e-4
    _closed(result.ledger)


# ------------------------------------------------------------- Glan-Taylor


def _glan_taylor(ghosts=None):
    """Calcite (1.65835, 1.48640), axis y in both prisms, an air gap
    of normal (0, -sin 40°, cos 40°), normal incidence."""
    calcite = _calcite((0.0, 1.0, 0.0), CALCITE_HANDBOOK)
    cut = math.radians(40.0)
    optic = Optic()
    optic.surfaces.add(index=0, radius=be.inf, thickness=be.inf)
    optic.surfaces.add(
        index=1, thickness=3.0, material=calcite, is_stop=True,
        interaction_type="anisotropic", comment="in",
    )  # fmt: skip
    optic.surfaces.add(
        index=2, thickness=0.01, rx=cut, interaction_type="anisotropic", comment="gap1"
    )
    optic.surfaces.add(
        index=3, thickness=3.0, material=calcite, rx=cut,
        interaction_type="anisotropic", comment="gap2",
    )  # fmt: skip
    optic.surfaces.add(
        index=4, thickness=5.0, interaction_type="anisotropic", comment="out"
    )
    optic.surfaces.add(index=5)
    optic.set_aperture(aperture_type="EPD", value=2.0)
    optic.fields.set_type(field_type="angle")
    optic.fields.add(y=0.0)
    optic.wavelengths.add(value=WL, is_primary=True)
    optic.updater.set_polarization(_state(S45, S45))
    return BranchTracer(optic, ghosts=ghosts)


@pytest.mark.parametrize("ghosts", [None, [2]])
def test_glan_taylor(set_test_backend, ghosts):
    """The o branch is totally reflected at the gap; the e branch
    has P_yy = 0.8907650 and the power 0.7934623 of the y input (7 digits),
    and it is the only branch with power; the ledger closes."""
    result = _glan_taylor(ghosts).trace_all(rays=_rays())
    e_key = (("in", "T", "e"), ("gap1", "T"), ("gap2", "T", "e"), ("out", "T"))
    kept = [k for k, b in result.branches.items() if b.power > 1e-12]
    assert kept == [e_key]
    branch = result[e_key]
    assert abs(branch.power - 0.5 * 0.7934623) < 5e-8
    p = _np(branch.rays.p)[0]
    assert abs(abs(p[1, 1]) - 0.8907650) < 5e-8
    assert np.max(np.abs(p[0, :2])) < 1e-12 and np.max(np.abs(p[:2, 0])) < 1e-12
    # No o branch reaches the image; its child at the gap is lost.
    assert all(k[0][2] == "e" for k in result.branches)
    _closed(result.ledger)


# ------------------------------------------------------------- ghosts


@pytest.mark.parametrize("theta_deg, state", [(0.0, (1.0, 0.0)), (30.0, (0.0, 1.0))])
def test_glass_plate_ghost_series(set_test_backend, theta_deg, state):
    """A glass plate (n = 1.5) with ghosts at both faces: the image branches
    carry T² R^(2m), the returned ones R and T² R^(2m+1) (the incoherent
    series); max_reflections = 2 prunes T R³ after the third reflection. At 30°
    in the x-z plane the y input is s-polarized (R_s)."""
    glass = IdealMaterial(1.5)
    optic = Optic()
    optic.surfaces.add(index=0, radius=be.inf, thickness=be.inf)
    optic.surfaces.add(index=1, thickness=2.0, material=glass, is_stop=True)
    optic.surfaces.add(index=2, thickness=5.0)
    optic.surfaces.add(index=3)
    optic.set_aperture(aperture_type="EPD", value=2.0)
    optic.fields.set_type(field_type="angle")
    optic.fields.add(y=0.0)
    optic.wavelengths.add(value=WL, is_primary=True)
    optic.updater.set_polarization(_state(*state))
    theta = math.radians(theta_deg)
    r_s, r_p = _fresnel_r(1.5, theta)
    r = r_p if state == (1.0, 0.0) else r_s
    t = 1 - r
    result = BranchTracer(optic, ghosts="all", threshold=0.0).trace_all(
        rays=_rays(theta)
    )
    T1, T2, R1, R2 = ("s1", "T"), ("s2", "T"), ("s1", "R"), ("s2", "R")
    expected = {
        (T1, T2): (t**2, "image"),
        (T1, R2, R1, T2): (t**2 * r**2, "image"),
        (T1, R2, T1): (t**2 * r, "returned"),
        (R1,): (r, "returned"),
    }
    assert set(result.keys()) == set(expected)
    for key, (power, terminal) in expected.items():
        assert abs(result[key].power - power) < 1e-12
        assert result[key].terminal == terminal
    assert result.pruned == {(T1, R2, R1, R2): pytest.approx((t * r**3, "reflections"))}
    assert abs(result.ledger.pruned - t * r**3) < 1e-12
    _closed(result.ledger)


def test_threshold_prunes_and_reports(set_test_backend):
    """The default threshold 1E-6 drops the fourth-order ghost of a glass
    plate (T² R⁴ = 2.4E-6 · 0.0016 < 1E-6 of the launch power is not reached
    with max_reflections 2; with 4 it is) and reports its power."""
    glass = IdealMaterial(1.5)
    optic = Optic()
    optic.surfaces.add(index=0, radius=be.inf, thickness=be.inf)
    optic.surfaces.add(index=1, thickness=2.0, material=glass, is_stop=True)
    optic.surfaces.add(index=2, thickness=5.0)
    optic.surfaces.add(index=3)
    optic.set_aperture(aperture_type="EPD", value=2.0)
    optic.fields.set_type(field_type="angle")
    optic.fields.add(y=0.0)
    optic.wavelengths.add(value=WL, is_primary=True)
    optic.updater.set_polarization(_state(1.0, 0.0))
    r, t = 0.04, 0.96
    result = BranchTracer(optic, ghosts="all", max_reflections=6).trace_all(
        rays=_rays()
    )
    powers = sorted(b.power for b in result.branches.values())
    assert min(powers) > 1e-6
    assert all(
        p <= 1e-6 for p, reason in result.pruned.values() if reason == "threshold"
    )
    assert result.ledger.pruned > 0.0
    # The kept image power is T² (1 + R² + R⁴) (T² R⁶ = 3.8E-9 is pruned).
    assert abs(result.ledger.kept - t**2 * (1 + r**2 + r**4)) < 1e-12
    _closed(result.ledger)


# ------------------------------------------------------------- API


def test_power_helper_and_unpolarized_launch(set_test_backend):
    """``ray_power`` does not change the rays; an unpolarized launch averages
    the x and y inputs (displacer: each branch carries 1/2 of its faces)."""
    axis = (S45, 0.0, S45)
    optic = _crystal_optic([_calcite(axis)], [1.0])
    optic.updater.set_polarization(PolarizationState(is_polarized=False))
    rays = _rays()
    before = _np(rays.i).copy()
    power = ray_power(rays, None)
    assert np.array_equal(_np(rays.i), before) and abs(_np(power)[0] - 1.0) < 1e-15
    result = BranchTracer(optic).trace_all(rays=rays)
    n_o, n_e = CALCITE
    n_45 = n_o * n_e / math.sqrt((n_o**2 + n_e**2) / 2)
    assert (
        abs(result[(("s1", "T", "e"), ("s2", "T"))].power - 0.5 * _face(n_45) ** 2)
        < 1e-12
    )
    assert (
        abs(result[(("s1", "T", "o"), ("s2", "T"))].power - 0.5 * _face(n_o) ** 2)
        < 1e-12
    )
    # The launch rays are not changed by the trace.
    assert abs(_np(rays.z)[0] - (-1.0)) < 1e-15
    _closed(result.ledger)


def test_generated_rays_and_labels(set_test_backend):
    """Rays from the optic's own distribution; the model label names the
    surface in the key; ``enumerate`` gives the keys; a branch sequence
    traces the same rays."""
    axis = (S45, 0.0, S45)
    optic = _crystal_optic([_calcite(axis)], [1.0])
    optic.surfaces[1].interaction_model.label = "X1"
    optic.updater.set_polarization(_state(S45, S45))
    tracer = BranchTracer(optic)
    keys = tracer.enumerate(num_rays=6, distribution="hexapolar")
    assert keys == [(("X1", "T", "o"), ("s2", "T")), (("X1", "T", "e"), ("s2", "T"))]
    result = tracer.trace_all(num_rays=6)
    for key in keys:
        seq = tracer.sequence(key)
        rays = seq.trace(0.0, 0.0, WL, num_rays=6)
        assert np.max(np.abs(_np(rays.x) - _np(result[key].rays.x))) < 1e-12
        assert np.max(np.abs(_np(rays.i) - _np(result[key].rays.i))) < 1e-12
        # The views of the branch keep the records of each step.
        assert len(result[key].views) == len(result[key].steps) == 4


def test_errors(set_test_backend):
    axis = (S45, 0.0, S45)
    optic = _crystal_optic([_calcite(axis)], [1.0])
    tracer = BranchTracer(optic)
    with pytest.raises(ValueError, match="does not match"):
        tracer.sequence((("s9", "T", "o"), ("s2", "T")))
    with pytest.raises(ValueError, match="not a child"):
        tracer.sequence((("s1", "T", "x"), ("s2", "T")))
    with pytest.raises(ValueError, match="ends before"):
        tracer.sequence((("s1", "T", "o"),))
    with pytest.raises(ValueError, match="between the object"):
        BranchTracer(optic, ghosts=[3])
    with pytest.raises(ValueError, match="indices or 'all'"):
        BranchTracer(optic, ghosts="some")
    coated = Optic()
    coated.surfaces.add(index=0, radius=be.inf, thickness=be.inf)
    coated.surfaces.add(
        index=1, thickness=1.0, material=IdealMaterial(1.5), is_stop=True,
        coating=None,
    )  # fmt: skip
    coated.surfaces.add(index=2, thickness=1.0, material="mirror")
    coated.surfaces.add(index=3)
    coated.updater.set_polarization(_state(1.0, 0.0))
    with pytest.raises(ValueError, match="mirror"):
        BranchTracer(coated, ghosts=[2])
    plain = Optic()
    plain.surfaces.add(index=0, radius=be.inf, thickness=be.inf)
    plain.surfaces.add(index=1, thickness=1.0, is_stop=True)
    plain.surfaces.add(index=2)
    with pytest.raises(ValueError, match="polarization state"):
        BranchTracer(plain)


def test_folded_system_with_ghosts(set_test_backend):
    """A lens, a fold mirror and a second lens: the main branch equals the
    nominal trace, the ghosts of the four lens faces pass the mirror in both
    directions, and the ledger closes (the mirror reflects all)."""
    optic = Optic()
    optic.surfaces.add(index=0, radius=be.inf, thickness=be.inf)
    optic.surfaces.add(
        index=1, radius=50.0, thickness=3.0, material=IdealMaterial(1.5), is_stop=True
    )
    optic.surfaces.add(index=2, radius=-50.0, thickness=20.0)
    optic.surfaces.add(index=3, radius=-200.0, thickness=-10.0, material="mirror")
    optic.surfaces.add(
        index=4, radius=30.0, thickness=-2.0, material=IdealMaterial(1.6)
    )
    optic.surfaces.add(index=5, radius=40.0, thickness=-15.0)
    optic.surfaces.add(index=6)
    optic.set_aperture(aperture_type="EPD", value=5.0)
    optic.fields.set_type(field_type="angle")
    optic.fields.add(y=0.0)
    optic.wavelengths.add(value=0.55, is_primary=True)
    optic.updater.set_polarization(_state(1.0, 0.0))
    nominal = optic.trace(0, 0, 0.55, num_rays=3)
    result = BranchTracer(optic, ghosts=[1, 2, 4, 5]).trace_all(num_rays=3)
    main = result[(("s1", "T"), ("s2", "T"), ("s4", "T"), ("s5", "T"))]
    for name in ("x", "y", "z", "L", "M", "N", "opd"):
        diff = _np(getattr(nominal, name)) - _np(getattr(main.rays, name))
        assert np.max(np.abs(diff)) < 1e-12
    # A ghost from the back of the second lens returns through the mirror.
    back = (("s1", "T"), ("s2", "T"), ("s4", "T"), ("s5", "R"), ("s4", "T"))
    assert back + (("s2", "T"), ("s1", "T")) in result.branches
    assert result[(("s1", "R"),)].terminal == "returned"
    _closed(result.ledger)
    # The sequence of a branch through the mirror reproduces the branch.
    tracer = BranchTracer(optic, ghosts=[1, 2, 4, 5])
    for key in (tuple(main.key), back + (("s2", "T"), ("s1", "T"))):
        rays = tracer.sequence(key).trace(0.0, 0.0, 0.55, num_rays=3)
        for name in ("x", "y", "z", "L", "M", "N", "i"):
            diff = _np(getattr(rays, name)) - _np(getattr(result[key].rays, name))
            assert np.nanmax(np.abs(diff)) < 1e-12
