"""Tests of folds inside crystals (both backends).

A fold is a mirror surface of the optic (``material="mirror"``) with an
``AnisotropicInteractionModel`` next to a tensor material: the hypotenuse of a
right-angle prism (total internal reflection), a Porro face, a metal or a
perfect mirror on a crystal face. It reflects into the modes of the crystal
and keeps the listed order of the surfaces.

Closed forms:

* (a) A right-angle calcite prism with the optic axis normal to the plane of
  incidence: s (E along the axis) sees n_e, p sees n_o, and both reflect
  totally at 45°. The phases of r_s, r_p are -φ_s, -φ_p of Chipman, Lam and
  Young, *Polarized Light and Optical Systems* (CRC, 2019), §8.3.7,
  Eq. (8.28), p. 307, with sin θ_C = 1/n (exp(-iωt): the transmitted wave
  decays as exp(-k0 γ z), γ = √(n² sin²θ - 1)).
* (b) Each branch of that prism equals an isotropic prism of index n_o or n_e
  with a fold mirror; the paraxial image shift is the tunnel-diagram plate
  L (1 - 1/n).
* (c) An optic axis in the plane of incidence: no o-e conversion, and the e
  child reflects at its own angle (tangential k matched on the e sheet of the
  index surface). A general axis: o-e conversion, the powers add to 1.
* (d) A quartz fold at 30° (below the critical angle): the reflected powers
  plus the ``escaped`` power are 1; ``escaped`` is the Fresnel T of the face.
* (e) A perfect mirror on a crystal face: r = -1 for each mode at normal
  incidence; at oblique incidence the tangential E of the summed field is 0.
  A metal fold: the reflected power is the Fresnel R; the rest is absorbed.
* (f) Two folds (a Porro-type retroreflector): branch count, ledger, and
  ``BranchTracer.sequence(key)`` reproduces each branch.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

import optiland.backend as be
from optiland.anisotropic import constitutive_matrix, plane_wave_modes, solve_interface
from optiland.interactions import AnisotropicInteractionModel
from optiland.materials import IdealMaterial
from optiland.materials.anisotropic import UniaxialMaterial
from optiland.optic import Optic
from optiland.raytrace.branches import BranchTracer

from .test_branches import _closed, _face
from .test_tracing import CALCITE, WL, _calcite, _np, _state

QUARTZ = (1.5442464, 1.5533580)  # n_o, n_e at 589.3 nm (Ghosh 1999)
METAL = (1.2, 7.26)  # an aluminium-like n, k at 589 nm (the value is not fitted)
S45 = math.sin(math.pi / 4)
HYPOTENUSE = np.array([0.0, -S45, S45])  # the fold normal of rx = 45°


def _reflected(theta):
    """The axis after a mirror at rx = theta (z reflects to (0, sin 2θ,
    -cos 2θ)) and the rx of a surface normal to it."""
    turn = 2 * theta
    return np.array([0.0, math.sin(turn), -math.cos(turn)]), turn - math.pi


def _prism(medium, anisotropic=True, fold_angle=math.pi / 4, exit_face=True):
    """A right-angle prism: entrance face at z = 0, a fold at z = 5 (rx =
    ``fold_angle``), then the exit face 5 mm along the reflected axis and the
    image 5 mm behind it; without ``exit_face`` the image lies in the crystal
    2 mm after the fold (an anisotropic image surface "img" of the same
    crystal: no interface)."""
    kw = {"interaction_type": "anisotropic"} if anisotropic else {}
    optic = Optic()
    optic.surfaces.add(index=0, radius=be.inf, thickness=be.inf)
    optic.surfaces.add(
        index=1, thickness=5.0, material=medium, is_stop=True, comment="in", **kw
    )
    optic.surfaces.add(
        index=2, x=0.0, y=0.0, z=5.0, rx=fold_angle, material="mirror",
        comment="hyp", **kw,
    )  # fmt: skip
    axis, tilt = _reflected(fold_angle)
    if exit_face:
        p = np.array([0.0, 0.0, 5.0]) + 5.0 * axis
        optic.surfaces.add(
            index=3, x=0.0, y=float(p[1]), z=float(p[2]), rx=tilt, comment="out",
            **kw,
        )  # fmt: skip
        q = p + 5.0 * axis
        optic.surfaces.add(index=4, x=0.0, y=float(q[1]), z=float(q[2]), rx=tilt)
    else:
        p = np.array([0.0, 0.0, 5.0]) + 2.0 * axis
        optic.surfaces.add(
            index=3, x=0.0, y=float(p[1]), z=float(p[2]), rx=tilt,
            material=medium, comment="img", **kw,
        )  # fmt: skip
    optic.set_aperture(aperture_type="EPD", value=2.0)
    optic.fields.set_type(field_type="angle")
    optic.fields.add(y=0.0)
    optic.fields.add(y=3.0)
    optic.wavelengths.add(value=WL, is_primary=True)
    optic.updater.set_polarization(_state(1.0, 0.0))
    return optic


def _power(result, prefix):
    """The summed power of the branches whose key starts with ``prefix``."""
    return sum(
        b.power for k, b in result.branches.items() if k[: len(prefix)] == prefix
    )


def _branch(result, prefix):
    """The branch ``prefix`` + the image entry in the same mode (the image
    surface "img" in the crystal is not an interface)."""
    return result[(*prefix, ("img", "T", prefix[-1][2]))]


def _unit(v):
    v = np.asarray(v, dtype=float)
    return v / np.linalg.norm(v)


def _tir_phases(n, theta):
    """φ_s, φ_p of Chipman, Lam and Young (2019), Eq. (8.28), sin θ_C = 1/n."""
    sc2 = 1.0 / n**2
    root = math.sqrt(math.sin(theta) ** 2 - sc2)
    phi_s = 2 * math.atan(root / math.cos(theta))
    phi_p = 2 * math.atan(root / (math.cos(theta) * sc2))
    return phi_s, phi_p


# ------------------------------------------------------------- (a), (b)


def test_right_angle_calcite_prism_case_a(set_test_backend):
    """(a) Axis along x, fold in the y-z plane: the x input (s at the fold, e,
    n_e) and the y input (p, o, n_o) each give one branch with the power of
    the two faces (TIR loses nothing), Arg r = -φ of Eq. (8.28) with
    p̂ = k̂ × ŝ on both beams, no s-p coupling, and a closed ledger."""
    n_o, n_e = CALCITE
    phi_s, _ = _tir_phases(n_e, math.pi / 4)
    _, phi_p = _tir_phases(n_o, math.pi / 4)
    for state, mode, n in (((1.0, 0.0), "e", n_e), ((0.0, 1.0), "o", n_o)):
        optic = _prism(_calcite((1.0, 0.0, 0.0)))
        optic.updater.set_polarization(_state(*state))
        result = BranchTracer(optic).trace_all(num_rays=3)
        key = (("in", "T", mode), ("hyp", "F", mode), ("out", "T"))
        kept = [k for k, b in result.branches.items() if b.power > 1e-20]
        assert kept == [key]
        assert abs(result[key].power - _face(n) ** 2) < 1e-12
        assert result.ledger.escaped == 0.0
        _closed(result.ledger)
        p = _np(result[key].rays.p)[0]
        # k_in = z, s = x, p_in = z × x = y; k_out = y, p_out = y × x = -z.
        if mode == "e":
            r, leak = p[0, 0], abs(p[2, 0])
            assert abs(np.angle(r) - (-phi_s)) < 1e-12
        else:
            r, leak = -p[2, 1], abs(p[0, 1])
            assert abs(np.angle(r) - (-phi_p)) < 1e-12
        assert leak < 1e-12
        # |r| = 1: the amplitude is the product of the two face factors.
        assert abs(abs(r) - 4 * n / (1 + n) ** 2) < 1e-12


@pytest.mark.parametrize("mode, index", [("o", 0), ("e", 1)])
def test_prism_equals_isotropic_prism_case_b(set_test_backend, mode, index):
    """(b) Each mode of the axis-x prism traces as an isotropic prism of its
    index with a fold mirror (x, y, z, L, M, N, OPD to 1E-12 at two fields),
    and the paraxial image of a thin lens moves by L (1 - 1/n) (L = 10 mm of
    unfolded glass path). The nominal trace equals the branch of its mode."""
    n = CALCITE[index]
    crystal = _prism(_calcite((1.0, 0.0, 0.0)))
    for s in (1, 2):
        crystal.surfaces[s].interaction_model.mode = mode
    crystal.updater.set_polarization(_state(*((1.0, 0.0) if index else (0.0, 1.0))))
    glass = _prism(IdealMaterial(n), anisotropic=False)
    glass.updater.set_polarization("ignore")
    for hy in (0.0, 1.0):
        a = crystal.trace(0.0, hy, WL, num_rays=5)
        b = glass.trace(0.0, hy, WL, num_rays=5)
        for name in ("x", "y", "z", "L", "M", "N", "opd"):
            diff = _np(getattr(a, name)) - _np(getattr(b, name))
            assert np.max(np.abs(diff)) < 1e-12, name
    # The nominal trace equals its branch (acceptance: finite rays to 1E-12).
    nominal = crystal.trace(0.0, 1.0, WL, num_rays=5)
    assert np.all(np.isfinite(_np(nominal.x)))
    result = BranchTracer(crystal).trace_all(0.0, 1.0, WL, num_rays=5)
    branch = result[(("in", "T", mode), ("hyp", "F", mode), ("out", "T"))]
    for name in ("x", "y", "z", "L", "M", "N", "opd", "i"):
        diff = _np(getattr(nominal, name)) - _np(getattr(branch.rays, name))
        assert np.max(np.abs(diff)) < 1e-12, name
    # Paraxial: a thin lens (f = 50) 10 mm before the prism.
    lens = Optic()
    lens.surfaces.add(index=0, radius=be.inf, thickness=be.inf)
    lens.surfaces.add(
        index=1, surface_type="paraxial", f=50.0, thickness=10.0, is_stop=True
    )
    medium = _calcite((1.0, 0.0, 0.0))
    kw = {"interaction_type": "anisotropic"}
    lens.surfaces.add(index=2, thickness=5.0, material=medium, **kw)
    lens.surfaces.add(
        index=3, x=0.0, y=0.0, z=15.0, rx=math.pi / 4, material="mirror", **kw
    )
    lens.surfaces.add(index=4, x=0.0, y=5.0, z=15.0, rx=-math.pi / 2, **kw)
    lens.surfaces.add(index=5, x=0.0, y=40.0, z=15.0, rx=-math.pi / 2)
    lens.set_aperture(aperture_type="EPD", value=2.0)
    lens.fields.set_type(field_type="angle")
    lens.fields.add(y=0.0)
    lens.wavelengths.add(value=WL, is_primary=True)
    lens.updater.set_polarization(_state(1.0, 0.0))
    for s in (2, 3):
        lens.surfaces[s].interaction_model.mode = mode
    y, u = lens.paraxial.marginal_ray()
    y, u = _np(y).ravel(), _np(u).ravel()
    image = 10.0 + 10.0 + abs(y[4] / u[4])
    assert abs(image - (50.0 + 10.0 * (1 - 1 / n))) < 1e-12


# ------------------------------------------------------------- (c)


def _e_index(k_hat, axis, n_o, n_e):
    """The e index of a wave normal: 1/n² = cos²ψ/n_o² + sin²ψ/n_e²."""
    c = float(np.dot(k_hat, axis))
    return 1.0 / math.sqrt(c**2 / n_o**2 + (1 - c**2) / n_e**2)


def test_axis_in_the_plane_of_incidence_case_c(set_test_backend):
    """(c) Axis in the y-z plane at 30° from z: the o (x) input reflects as o at
    45°; the e (y) input reflects as e only, at the angle of the tangential-k
    match on the e sheet (solved here by bisection), not at 45°."""
    n_o, n_e = CALCITE
    phi = math.radians(30.0)
    axis = np.array([0.0, math.sin(phi), math.cos(phi)])
    normal = _unit(HYPOTENUSE)
    tangent = _unit(np.cross(np.cross(normal, [0.0, 0.0, 1.0]), normal))
    for state, mode in (((1.0, 0.0), "o"), ((0.0, 1.0), "e")):
        optic = _prism(_calcite(tuple(axis)), exit_face=False)
        optic.updater.set_polarization(_state(*state))
        result = BranchTracer(optic, threshold=0.0).trace_all(num_rays=1)
        other = "e" if mode == "o" else "o"
        assert _power(result, (("in", "T", mode), ("hyp", "F", other))) < 1e-24
        branch = _branch(result, (("in", "T", mode), ("hyp", "F", mode)))
        k_in = (n_o if mode == "o" else _e_index([0, 0, 1], axis, n_o, n_e)) * np.array(
            [0.0, 0.0, 1.0]
        )
        k_t = float(np.dot(k_in, tangent))
        k_r = _np(branch.rays.k)[0].real
        # Tangential k is kept; the child leaves the face (k · n̂ < 0 here).
        assert abs(float(np.dot(k_r, tangent)) - k_t) < 1e-12
        if mode == "o":
            assert abs(np.linalg.norm(k_r) - n_o) < 1e-12
            assert np.max(np.abs(_unit(k_r) - [0.0, 1.0, 0.0])) < 1e-12
        else:
            # The normal component q > 0 along -n̂ with |k| = n_e(k̂).
            def f(q):
                k = k_t * tangent - q * normal
                return np.linalg.norm(k) - _e_index(_unit(k), axis, n_o, n_e)

            lo, hi = 0.0, 3.0
            for _ in range(200):
                mid = 0.5 * (lo + hi)
                lo, hi = (mid, hi) if f(mid) < 0 else (lo, mid)
            expected = k_t * tangent - 0.5 * (lo + hi) * normal
            assert np.max(np.abs(k_r - expected)) < 1e-12
            angle = math.degrees(math.atan2(k_t, float(np.dot(k_r, -normal))))
            assert abs(angle - 45.0) > 1.0  # its own angle
        _closed(result.ledger)


def test_general_axis_converts_case_c(set_test_backend):
    """(c) A general axis: each input mode splits into both reflected modes
    (o-e conversion), each child keeps the tangential k, and the reflected
    powers add to the power that reached the fold (TIR: 1)."""
    n_o, n_e = CALCITE
    axis = _unit([0.5, 0.6, 0.62])
    normal = _unit(HYPOTENUSE)
    optic = _prism(_calcite(tuple(axis)), exit_face=False)
    optic.updater.set_polarization(_state(S45, S45))
    result = BranchTracer(optic, threshold=0.0).trace_all(num_rays=1)
    assert len({key[:2] for key in result.branches}) == 4
    for mode in ("o", "e"):
        prefixes = [(("in", "T", mode), ("hyp", "F", m)) for m in ("o", "e")]
        powers = [_power(result, k) for k in prefixes]
        assert min(powers) > 1e-4  # conversion
        keys = [_branch(result, k).key for k in prefixes]
        tangential = [
            _np(result[k].rays.k)[0].real
            - np.dot(_np(result[k].rays.k)[0].real, normal) * normal
            for k in keys
        ]
        assert np.max(np.abs(tangential[0] - tangential[1])) < 1e-12
        # The two reflected modes leave at different angles.
        d0, d1 = (_unit(_np(result[k].rays.k)[0].real) for k in keys)
        assert np.linalg.norm(d0 - d1) > 1e-3
    # The powers of the fold children add to the entrance power (TIR).
    entrance = {m: 0.0 for m in ("o", "e")}
    for key, branch in result.branches.items():
        entrance[key[0][2]] += branch.power
    one = _single_face_powers(axis)
    for mode in ("o", "e"):
        assert abs(entrance[mode] - one[mode]) < 1e-12
    _closed(result.ledger)


def _single_face_powers(axis):
    """The powers of the o and e children of the entrance face (45° input)."""
    optic = Optic()
    optic.surfaces.add(index=0, radius=be.inf, thickness=be.inf)
    optic.surfaces.add(
        index=1, thickness=1.0, material=_calcite(tuple(axis)), is_stop=True,
        interaction_type="anisotropic", comment="in",
    )  # fmt: skip
    optic.surfaces.add(
        index=2, material=_calcite(tuple(axis)), interaction_type="anisotropic"
    )
    optic.set_aperture(aperture_type="EPD", value=2.0)
    optic.fields.set_type(field_type="angle")
    optic.fields.add(y=0.0)
    optic.wavelengths.add(value=WL, is_primary=True)
    optic.updater.set_polarization(_state(S45, S45))
    result = BranchTracer(optic, threshold=0.0).trace_all(num_rays=1)
    return {m: _power(result, (("in", "T", m),)) for m in ("o", "e")}


# ------------------------------------------------------------- (d)


def _fresnel_internal(n, theta):
    """R_s, R_p from index n into air at the angle theta (below critical)."""
    c = math.cos(theta)
    root = math.sqrt(1 - n**2 * math.sin(theta) ** 2)
    rs = (n * c - root) / (n * c + root)
    rp = (c - n * root) / (c + n * root)
    return rs**2, rp**2


def test_quartz_fold_below_the_critical_angle_case_d(set_test_backend):
    """(d) Quartz, axis along x, fold at 30° (critical angle 40.4°): the x input
    (s, n_e) reflects R_s and the y input (p, n_o) R_p of the internal Fresnel
    equations; the rest leaves through the fold face (``escaped``, the Fresnel
    T); nothing is absorbed. A general axis: reflected + escaped = the power
    at the fold, and escaped = the solver's T of the bare face."""
    n_o, n_e = QUARTZ
    theta = math.radians(30.0)
    for state, mode, n, which in (((1.0, 0.0), "e", n_e, 0), ((0.0, 1.0), "o", n_o, 1)):
        optic = _prism(_calcite((1.0, 0.0, 0.0), QUARTZ), fold_angle=theta)
        optic.updater.set_polarization(_state(*state))
        result = BranchTracer(optic).trace_all(num_rays=1)
        r = _fresnel_internal(n, theta)[which]
        key = (("in", "T", mode), ("hyp", "F", mode), ("out", "T"))
        assert abs(result[key].power - _face(n) ** 2 * r) < 1e-12
        assert abs(result.ledger.escaped - _face(n) * (1 - r)) < 1e-12
        _closed(result.ledger)
    # A general axis.
    axis = _unit([0.3, -0.5, 0.8])
    optic = _prism(_calcite(tuple(axis), QUARTZ), fold_angle=theta, exit_face=False)
    optic.updater.set_polarization(_state(S45, S45))
    result = BranchTracer(optic, threshold=0.0).trace_all(num_rays=1)
    reflected = sum(b.power for b in result.branches.values())
    entrance = _single_face_powers_quartz(axis)
    assert abs(reflected + result.ledger.escaped - sum(entrance.values())) < 1e-12
    # The solver's T of each incident mode at the fold.
    medium = _calcite(tuple(axis), QUARTZ)
    m = constitutive_matrix(medium, WL)
    normal = _unit(np.array([0.0, -math.sin(theta), math.cos(theta)]))
    modes = plane_wave_modes(np.array([0.0, 0.0, 1.0]), m)
    expected = 0.0
    for j in (0, 1):
        k_j = _np(modes.k)[0, j]
        is_o = abs(np.dot(k_j, k_j) - QUARTZ[0] ** 2) < 1e-9
        weight = entrance["o" if is_o else "e"]
        solved = solve_interface(
            normal, m, constitutive_matrix(IdealMaterial(1.0), WL), k_j,
            _np(modes.E)[0, j],
        )  # fmt: skip
        expected += weight * float(_np(solved.transmittance)[0])
    assert abs(result.ledger.escaped - expected) < 1e-12
    _closed(result.ledger)


def _single_face_powers_quartz(axis):
    optic = Optic()
    optic.surfaces.add(index=0, radius=be.inf, thickness=be.inf)
    optic.surfaces.add(
        index=1, thickness=1.0, material=_calcite(tuple(axis), QUARTZ),
        is_stop=True, interaction_type="anisotropic", comment="in",
    )  # fmt: skip
    optic.surfaces.add(
        index=2, material=_calcite(tuple(axis), QUARTZ), interaction_type="anisotropic"
    )
    optic.set_aperture(aperture_type="EPD", value=2.0)
    optic.fields.set_type(field_type="angle")
    optic.fields.add(y=0.0)
    optic.wavelengths.add(value=WL, is_primary=True)
    optic.updater.set_polarization(_state(S45, S45))
    result = BranchTracer(optic, threshold=0.0).trace_all(num_rays=1)
    return {m: _power(result, (("in", "T", m),)) for m in ("o", "e")}


# ------------------------------------------------------------- (e)


def _mirror_optic(medium, far, tilt=0.0):
    """Entrance face at z = 0, a mirror at z = 5 (rx = tilt) with the far
    medium ``far``, the image in the crystal 2 mm back along the reflected
    axis."""
    optic = Optic()
    optic.surfaces.add(index=0, radius=be.inf, thickness=be.inf)
    optic.surfaces.add(
        index=1, thickness=5.0, material=medium, is_stop=True, comment="in",
        interaction_type="anisotropic",
    )  # fmt: skip
    optic.surfaces.add(
        index=2, x=0.0, y=0.0, z=5.0, rx=tilt, material="mirror", comment="m",
        interaction_type="anisotropic",
    )  # fmt: skip
    axis, rx = _reflected(tilt)
    p = np.array([0.0, 0.0, 5.0]) + 2.0 * axis
    optic.surfaces.add(
        index=3, x=0.0, y=float(p[1]), z=float(p[2]), rx=rx, material=medium,
        comment="img", interaction_type="anisotropic",
    )  # fmt: skip
    optic.surfaces[2].interaction_model.far_material = far
    optic.set_aperture(aperture_type="EPD", value=2.0)
    optic.fields.set_type(field_type="angle")
    optic.fields.add(y=0.0)
    optic.wavelengths.add(value=WL, is_primary=True)
    optic.updater.set_polarization(_state(1.0, 0.0))
    return optic


def test_perfect_mirror_on_a_crystal_face_case_e(set_test_backend):
    """(e) A perfect mirror at normal incidence, axis in the face at 30° from x:
    each mode returns with r = -1 (P = -t on its own field), no conversion,
    and the ledger closes with nothing absorbed."""
    n_o, n_e = CALCITE
    a = math.radians(30.0)
    axis = (math.cos(a), math.sin(a), 0.0)
    e_dir = np.array(axis)
    o_dir = np.array([-math.sin(a), math.cos(a), 0.0])
    for field, mode, n in ((e_dir, "e", n_e), (o_dir, "o", n_o)):
        optic = _mirror_optic(_calcite(axis), "perfect_conductor")
        optic.updater.set_polarization(_state(*field[:2]))
        result = BranchTracer(optic, threshold=0.0).trace_all(num_rays=1)
        other = "e" if mode == "o" else "o"
        assert _power(result, (("in", "T", mode), ("m", "F", other))) < 1e-28
        branch = _branch(result, (("in", "T", mode), ("m", "F", mode)))
        p = _np(branch.rays.p)[0]
        t = 2.0 / (1.0 + n)  # the entrance field factor
        assert np.max(np.abs(p @ field - (-t) * field)) < 1e-12
        assert abs(branch.power - _face(n)) < 1e-12
        _closed(result.ledger)


def test_perfect_conductor_tangential_e_is_zero_case_e(set_test_backend):
    """(e) Oblique incidence from calcite (a general axis) on a perfect
    conductor: the tangential E of incident + reflected fields is 0 to 1E-14,
    R = 1, and the transmitted children are zero."""
    rng = np.random.default_rng(85)
    for _ in range(5):
        axis = _unit(rng.normal(size=3))
        m = constitutive_matrix(_calcite(tuple(axis)), WL)
        theta = rng.uniform(0.1, 1.2)
        d = np.array([math.sin(theta), 0.0, math.cos(theta)])
        modes = plane_wave_modes(d, m)
        for j in (0, 1):
            e_in = _np(modes.E)[0, j]
            result = solve_interface([0.0, 0.0, 1.0], m, None, _np(modes.k)[0, j], e_in)
            e = _np(result.E)[0]
            total = e_in + e[0] + e[1]
            assert np.max(np.abs(total[:2])) < 1e-14 * max(1.0, np.abs(e_in).max())
            assert abs(_np(result.reflectance)[0] - 1.0) < 1e-12
            assert np.all(_np(result.power)[0, 2:] == 0.0)
            assert np.all(_np(result.evanescent)[0, 2:])
            assert np.all(np.abs(_np(result.prt)[0, 2:]) == 0.0)


def test_metal_fold_case_e(set_test_backend):
    """A metal behind a 45° fold (axis along x): the x input (s, n_e) reflects
    R_s = |(n_e c - q)/(n_e c + q)|², q = √(ε_m - n_e² s²); the rest is the
    metal's absorption (``absorbed``), nothing escapes."""
    n_o, n_e = CALCITE
    metal = IdealMaterial(*METAL)
    theta = math.pi / 4
    optic = _mirror_optic(_calcite((1.0, 0.0, 0.0)), metal, tilt=theta)
    result = BranchTracer(optic).trace_all(num_rays=1)
    eps = complex(*METAL) ** 2
    c, s = math.cos(theta), math.sin(theta)
    q = np.sqrt(eps - n_e**2 * s**2)
    r_s = abs((n_e * c - q) / (n_e * c + q)) ** 2
    branch = _branch(result, (("in", "T", "e"), ("m", "F", "e")))
    assert abs(branch.power - _face(n_e) * r_s) < 1e-12
    assert result.ledger.escaped == 0.0
    assert abs(result.ledger.absorbed - _face(n_e) * (1 - r_s)) < 1e-12
    assert abs(result.ledger.total() - 1.0) < 1e-12


# ------------------------------------------------------------- (f)


def _porro(axis):
    """Entrance z = 0, folds at z = 5 (rx 45°, the beam turns to +y) and at
    y = 6 (rx -45°, the beam returns along -z), exit face at z = 0, image at
    z = -5."""
    medium = _calcite(axis)
    kw = {"interaction_type": "anisotropic"}
    optic = Optic()
    optic.surfaces.add(index=0, radius=be.inf, thickness=be.inf)
    optic.surfaces.add(
        index=1, thickness=5.0, material=medium, is_stop=True, comment="in", **kw
    )
    optic.surfaces.add(
        index=2, x=0.0, y=0.0, z=5.0, rx=math.pi / 4, material="mirror",
        comment="f1", **kw,
    )  # fmt: skip
    optic.surfaces.add(
        index=3, x=0.0, y=6.0, z=5.0, rx=-math.pi / 4, material="mirror",
        comment="f2", **kw,
    )  # fmt: skip
    optic.surfaces.add(index=4, x=0.0, y=6.0, z=0.0, rx=math.pi, comment="out", **kw)
    optic.surfaces.add(index=5, x=0.0, y=6.0, z=-5.0, rx=math.pi)
    optic.set_aperture(aperture_type="EPD", value=2.0)
    optic.fields.set_type(field_type="angle")
    optic.fields.add(y=0.0)
    optic.wavelengths.add(value=WL, is_primary=True)
    optic.updater.set_polarization(_state(S45, S45))
    return optic


def test_porro_two_folds_case_f(set_test_backend):
    """(f) Two folds in one calcite block (a general axis): 8 branches (2 modes
    at each of 3 splitting steps), all returned toward -z at the image, the
    ledger closes (with the power that escapes at the second fold), and
    ``sequence(key)`` reproduces each branch; with the exit face as a ghost
    surface, a ghost goes back through both folds (folding on the reverse leg)
    and leaves through the entrance."""
    axis = tuple(_unit([0.4, 0.3, 0.87]))
    optic = _porro(axis)
    tracer = BranchTracer(optic, threshold=0.0)
    result = tracer.trace_all(num_rays=3)
    assert len(result) == 8
    for key, branch in result.branches.items():
        assert [e[1] for e in key] == ["T", "F", "F", "T"]
        assert np.all(_np(branch.rays.N) < -0.9)
        rays = tracer.sequence(key).trace(0.0, 0.0, WL, num_rays=3)
        for name in ("x", "y", "z", "L", "M", "N", "i"):
            diff = _np(getattr(rays, name)) - _np(getattr(branch.rays, name))
            assert np.nanmax(np.abs(diff)) < 1e-12, (key, name)
    # The e children meet the second fold at their own angle, partly below the
    # critical angle: that power leaves through the face (``escaped``).
    assert result.ledger.escaped > 0.0
    _closed(result.ledger)
    # The ghost of the exit face folds twice on its way back.
    ghosts = BranchTracer(optic, ghosts=[4], threshold=0.0, max_reflections=1)
    result = ghosts.trace_all(num_rays=1)
    back = [k for k, b in result.branches.items() if b.terminal == "returned"]
    assert back
    for key in back:
        assert [e[1] for e in key] == ["T", "F", "F", "R", "F", "F", "T"]
        assert [e[0] for e in key] == ["in", "f1", "f2", "out", "f2", "f1", "in"]
    _closed(result.ledger)
    key = back[0]
    rays = ghosts.sequence(key).trace(0.0, 0.0, WL, num_rays=1)
    for name in ("x", "y", "z", "L", "M", "N", "i"):
        diff = _np(getattr(rays, name)) - _np(getattr(result[key].rays, name))
        assert np.nanmax(np.abs(diff)) < 1e-12, name


# ------------------------------------------------------------- API


def test_fold_model_api(set_test_backend):
    """The fold flag, the far medium, the key entry, serialization, and the
    errors."""
    optic = _prism(_calcite((1.0, 0.0, 0.0)))
    model = optic.surfaces[2].interaction_model
    assert isinstance(model, AnisotropicInteractionModel)
    assert model.fold and model.is_reflective
    assert isinstance(model.far_medium(), IdealMaterial)
    model.far_material = "perfect_conductor"
    assert model.far_medium() is None
    data = model.to_dict()
    assert data["far_material"] == "perfect_conductor"
    copy = AnisotropicInteractionModel.from_dict(data, None)
    assert copy.fold and copy.far_material == "perfect_conductor"
    model.far_material = IdealMaterial(*METAL)
    copy = AnisotropicInteractionModel.from_dict(model.to_dict(), None)
    assert isinstance(copy.far_material, IdealMaterial)
    assert abs(float(_np(copy.far_material.k(WL))) - METAL[1]) < 1e-12
    with pytest.raises(ValueError, match="far_material"):
        AnisotropicInteractionModel(None, far_material="air")
    # The nominal key entry of a fold is "F".
    model.far_material = None
    model.mode = "o"
    optic.surfaces[1].interaction_model.mode = "o"
    rays = optic.trace(0.0, 0.0, WL, num_rays=3)
    assert rays.branch_key == (("in", "T", "o"), ("hyp", "F", "o"), ("out", "T"))
    # A perfect conductor has no transmitted child.
    model.far_material = "perfect_conductor"
    model.is_reflective = False
    with pytest.raises(ValueError, match="perfect conductor"):
        optic.trace(0.0, 0.0, WL, num_rays=3)
    model.is_reflective = True
    # A key entry that is not a child of the fold.
    tracer = BranchTracer(optic)
    with pytest.raises(ValueError, match="not a child"):
        tracer.sequence((("in", "T", "o"), ("hyp", "R", "o"), ("out", "T")))
    # The isotropic prism of equal tensors keeps the summed mode "R".
    glass = _prism(_iso_tensor(1.5))
    rays = glass.trace(0.0, 0.0, WL, num_rays=3)
    assert rays.branch_key == (("in", "T"), ("hyp", "F"), ("out", "T"))
    assert np.all(np.isfinite(_np(rays.x)))


def _iso_tensor(n):
    return UniaxialMaterial(IdealMaterial(n), IdealMaterial(n), optic_axis=(0, 0, 1))
