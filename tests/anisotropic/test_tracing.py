"""Tests of sequential tracing through anisotropic media (both backends).

Closed forms (the project's oracle cases; register E-04, E-05, E-13, E-19,
E-20): the calcite beam displacer (case 1), the Wollaston prism (case 2), the
Savart plate at normal incidence (case 3) and at 3° (case 8, Zhang, Ren and Mu
2010), the tilted calcite plate OPD (case 7, Avendaño-Alejo and Rosete-Aguilar
2006, Eq. (33)), the Fresnel powers of a plate at normal incidence, and the
attenuation of an absorbing plate. The isotropic limit: the golden systems of
``tests/regression`` with their glasses replaced by equal isotropic tensors
trace like the isotropic path with Fresnel coatings.
"""

from __future__ import annotations

import math
import warnings

import numpy as np
import pytest

import optiland.backend as be
from optiland.interactions import AnisotropicInteractionModel
from optiland.materials import IdealMaterial
from optiland.materials.anisotropic import (
    BiaxialMaterial,
    TensorMaterial,
    UniaxialMaterial,
)
from optiland.optic import Optic
from optiland.propagation import AnisotropicPropagation
from optiland.rays import (
    AnisotropicRays,
    PolarizationState,
    PolarizedRays,
    RealRays,
    create_polarization,
)

from ..regression.systems import GOLDEN_SYSTEMS

CALCITE = (1.6583434042, 1.4861300612)  # Ghosh 1999, 589.3 nm (oracle case 1)
CALCITE_632 = (1.6556901060, 1.4849090302)  # Ghosh 1999, 632.8 nm
CALCITE_HANDBOOK = (1.65835, 1.48640)  # oracle case 8
WL = 0.5893


def _np(x):
    return np.asarray(be.to_numpy(x))


def _calcite(axis, indices=CALCITE):
    n_o, n_e = indices
    return UniaxialMaterial(IdealMaterial(n_o), IdealMaterial(n_e), optic_axis=axis)


def _state(ex, ey, phase_y=0.0):
    return PolarizationState(
        is_polarized=True, Ex=ex, Ey=ey, phase_x=0.0, phase_y=phase_y
    )


def _crystal_optic(media, thicknesses, tilts=None, image_gap=5.0):
    """An optic in air: one surface per crystal face, the last into air.

    Args:
        media: The media after surfaces 1 .. len(media) (air after the last).
        thicknesses: The thicknesses after surfaces 1 .. len(media).
        tilts: Optional ry rotations (radians) of surfaces 1 .. len(media) + 1.
    """
    tilts = tilts or [0.0] * (len(media) + 1)
    optic = Optic()
    optic.surfaces.add(index=0, radius=be.inf, thickness=be.inf)
    for i, (medium, t) in enumerate(zip(media, thicknesses, strict=True)):
        optic.surfaces.add(
            index=i + 1,
            thickness=t,
            material=medium,
            ry=tilts[i],
            is_stop=i == 0,
            interaction_type="anisotropic",
        )
    optic.surfaces.add(
        index=len(media) + 1,
        thickness=image_gap,
        ry=tilts[len(media)],
        interaction_type="anisotropic",
    )
    optic.surfaces.add(index=len(media) + 2)
    optic.set_aperture(aperture_type="EPD", value=2.0)
    optic.fields.set_type(field_type="angle")
    optic.fields.add(y=0.0)
    optic.wavelengths.add(value=WL, is_primary=True)
    optic.updater.set_polarization(_state(1.0, 0.0))
    return optic


def _set_modes(optic, modes):
    for i, mode in enumerate(modes):
        optic.surfaces[i + 1].interaction_model.mode = mode


def _rays(theta_x=0.0, wavelength=WL, n=1, z0=-1.0):
    """Rays in the x-z plane at angle theta_x that reach the origin."""
    t = math.tan(theta_x)
    ones = np.ones(n)
    return AnisotropicRays(
        be.array(-t * abs(z0) * ones),
        be.array(0 * ones),
        be.array(z0 * ones),
        be.array(math.sin(theta_x) * ones),
        be.array(0 * ones),
        be.array(math.cos(theta_x) * ones),
        be.array(ones),
        be.array(wavelength * ones),
    )


def _trace(optic, modes, rays):
    _set_modes(optic, modes)
    optic.surfaces.trace(rays, skip=1)
    return rays


def _exit_state(optic, surface):
    """Position, direction and OPD recorded on a surface (ray 0)."""
    s = optic.surfaces[surface]
    x = np.array([_np(s.x)[0], _np(s.y)[0], _np(s.z)[0]])
    d = np.array([_np(s.L)[0], _np(s.M)[0], _np(s.N)[0]])
    return x, d, _np(s.opd)[0]


def _wavefront_opd(a, b, n_ambient=1.0):
    """OPD a - b on a common exit wavefront of two parallel rays (x, d, opd)."""
    (xa, da, oa), (xb, _, ob) = a, b
    return oa - ob - n_ambient * float(np.dot(da, xa - xb))


# ------------------------------------------------------------ closed forms


@pytest.mark.parametrize(
    "indices, shift",
    [(CALCITE, -0.1092064213), (CALCITE_632, -0.1084363550)],
)
def test_calcite_displacer_case1(set_test_backend, indices, shift):
    """Oracle case 1: the e ray walks off by -0.1092064213 mm per mm (589.3 nm),
    the o ray does not move; OPL = n d along k (Re(k) · Δr)."""
    axis = (math.sin(math.pi / 4), 0.0, math.cos(math.pi / 4))
    for d in (1.0, 2.5):
        optic = _crystal_optic([_calcite(axis, indices)], [d])
        e = _trace(optic, ["e", "T"], _rays())
        o = _trace(optic, ["o", "T"], _rays())
        assert abs(_np(e.x)[0] - shift * d) < 1e-9 * d
        assert abs(_np(e.y)[0]) < 1e-15
        assert abs(_np(o.x)[0]) < 1e-15
        # Both leave along z.
        assert abs(_np(e.L)[0]) < 1e-15 and abs(_np(o.L)[0]) < 1e-15
        n_o, n_e = indices
        n_45 = n_o * n_e / math.sqrt((n_o**2 + n_e**2) / 2)  # E-04 at 45°
        path_air = 1.0 + 5.0  # before and after the plate
        assert abs(_np(o.opd)[0] - (path_air + n_o * d)) < 1e-12
        assert abs(_np(e.opd)[0] - (path_air + n_45 * d)) < 1e-12
        assert e.branch_key == (("", "e"), ("", "T"))


def test_calcite_walkoff_inside_the_plate(set_test_backend):
    """Inside the plate the ray is along S, the wave vector along z, and the
    angle between them is the case 1 walk-off (E-05; -6.2323695075097° at
    30 digits; the case table prints -6.2323695095, a misprint)."""
    axis = (math.sin(math.pi / 4), 0.0, math.cos(math.pi / 4))
    optic = _crystal_optic([_calcite(axis)], [1.0])
    rays = _rays()
    _set_modes(optic, ["e", "T"])
    optic.surfaces[1].trace(rays)
    k = _np(rays.k)[0]
    s = np.array([_np(rays.L)[0], _np(rays.M)[0], _np(rays.N)[0]])
    assert np.max(np.abs(k.imag)) == 0.0
    assert abs(k[0]) < 1e-15 and abs(k[1]) < 1e-15
    rho = math.degrees(math.atan2(s[0], s[2]))
    assert abs(rho - (-6.2323695075097)) < 1e-11
    assert rays.mode == "e"


@pytest.mark.parametrize(
    "modes, angle",
    [(["o", "e", "T"], -3.6217511833), (["e", "o", "T"], 3.5692606184)],
)
def test_wollaston_case2(set_test_backend, modes, angle):
    """Oracle case 2: calcite, prism axes x and y, cut at 20° (a tilted
    surface: the tensors turn into the surface frame): the y-pol (o -> e) and
    the x-pol (e -> o) exit angles equal the E-19 Snell chain to 1E-12°; the
    case table (10 digits) carries 1.2E-9° of rounding (1E-8)."""
    media = [_calcite((1.0, 0.0, 0.0)), _calcite((0.0, 1.0, 0.0))]
    tilts = [0.0, math.radians(20.0), 0.0]
    optic = _crystal_optic(media, [2.0, 2.0], tilts=tilts)
    rays = _trace(optic, modes, _rays())
    exit_angle = math.degrees(math.atan2(_np(rays.L)[0], _np(rays.N)[0]))
    n_o, n_e = CALCITE
    n1, n2 = (n_o, n_e) if modes[0] == "o" else (n_e, n_o)
    a = math.radians(20.0)
    b = math.asin(n1 * math.sin(a) / n2)
    snell = math.degrees(math.asin(n2 * math.sin(a - b)))
    assert abs(exit_angle - snell) < 1e-12
    assert abs(exit_angle - angle) < 1e-8
    assert abs(_np(rays.M)[0]) < 1e-15


def test_savart_plate_case3(set_test_backend):
    """Oracle case 3: two 1 mm plates, axes (1, 0, 1)/√2 and (0, 1, 1)/√2: the
    beams e1 -> o2 and o1 -> e2 shift by 0.1092064213 mm along -x and -y."""
    c = math.sqrt(0.5)
    media = [_calcite((c, 0.0, c)), _calcite((0.0, c, c))]
    optic = _crystal_optic(media, [1.0, 1.0])
    eo = _trace(optic, ["e", "o", "T"], _rays())
    oe = _trace(optic, ["o", "e", "T"], _rays())
    shift = 0.1092064213
    assert abs(_np(eo.x)[0] + shift) < 1e-9 and abs(_np(eo.y)[0]) < 1e-15
    assert abs(_np(oe.y)[0] + shift) < 1e-9 and abs(_np(oe.x)[0]) < 1e-15
    separation = math.hypot(_np(eo.x)[0] - _np(oe.x)[0], _np(eo.y)[0] - _np(oe.y)[0])
    assert abs(separation - 0.1544412022) < 1e-9
    # Equal optical paths at normal incidence (one o and one e segment each).
    assert abs(_np(eo.opd)[0] - _np(oe.opd)[0]) < 1e-12


@pytest.mark.parametrize(
    "incidence, opd, shear",
    [(0.0, 0.0, 0.9251578), (3.0, 0.0348725, 0.9418175)],
)
def test_savart_plate_oblique_case8(set_test_backend, incidence, opd, shear):
    """Oracle case 8 (Zhang, Ren and Mu 2010): 6 mm plates, handbook calcite,
    incidence in the x-z plane: the OPD eo - oe on a common exit wavefront and
    the shear (the distance between the two exit beams) to the printed 7
    digits."""
    c = math.sqrt(0.5)
    media = [
        _calcite((c, 0.0, c), CALCITE_HANDBOOK),
        _calcite((0.0, c, c), CALCITE_HANDBOOK),
    ]
    optic = _crystal_optic(media, [6.0, 6.0])
    theta = math.radians(incidence)
    _trace(optic, ["e", "o", "T"], _rays(theta))
    eo = _exit_state(optic, 3)
    _trace(optic, ["o", "e", "T"], _rays(theta))
    oe = _exit_state(optic, 3)
    assert np.max(np.abs(eo[1] - oe[1])) < 1e-15  # parallel exit beams
    assert abs(_wavefront_opd(eo, oe) - opd) < 5e-8
    # The shear is the distance between the two parallel exit beams.
    delta = eo[0] - oe[0]
    delta = delta - float(np.dot(delta, eo[1])) * eo[1]
    assert abs(float(np.linalg.norm(delta)) - shear) < 5e-8


@pytest.mark.parametrize(
    "incidence, opd_cm",
    [
        (-60.0, 0.1975725287),
        (-30.0, 0.1438486464),
        (0.0, 0.0836493402),
        (30.0, 0.0341054380),
        (60.0, 0.0074917159),
    ],
)
def test_tilted_plate_opd_case7(set_test_backend, incidence, opd_cm):
    """Oracle case 7 (Avendaño-Alejo 2006, Eq. (33)): a 1 cm calcite plate
    (1.658 / 1.486), axis in the plane of incidence at arctan(n_e / n_o) from
    the normal: the o - e OPD on a common exit wavefront = (q_o - q_e) d."""
    n_o, n_e = 1.658, 1.486
    phi = math.atan(n_e / n_o)
    medium = _calcite((math.sin(phi), 0.0, math.cos(phi)), (n_o, n_e))
    optic = _crystal_optic([medium], [10.0])
    theta = math.radians(incidence)
    _trace(optic, ["o", "T"], _rays(theta))
    o = _exit_state(optic, 2)
    _trace(optic, ["e", "T"], _rays(theta))
    e = _exit_state(optic, 2)
    assert abs(_wavefront_opd(o, e) - 10.0 * opd_cm) < 1e-9


def test_plate_powers_at_normal_incidence(set_test_backend):
    """Calcite at 45°: the x field goes into the e mode, the y field into the o
    mode; each passes two faces of index n (the admittance H'_y / E_x = n):
    power 0.5 (4 n / (1 + n)²)² for a 45° input."""
    axis = (math.sin(math.pi / 4), 0.0, math.cos(math.pi / 4))
    optic = _crystal_optic([_calcite(axis)], [1.0])
    state = _state(math.sqrt(0.5), math.sqrt(0.5), 0.3)
    n_o, n_e = CALCITE
    n_45 = n_o * n_e / math.sqrt((n_o**2 + n_e**2) / 2)
    for mode, n in (("o", n_o), ("e", n_45)):
        rays = _trace(optic, [mode, "T"], _rays())
        rays.update_intensity(state)
        expected = 0.5 * (4 * n / (1 + n) ** 2) ** 2
        assert abs(_np(rays.i)[0] - expected) < 1e-12


def test_absorbing_plate(set_test_backend):
    """An a-cut plate (axis x) with n_o + iκ: the o ray (y field) loses
    exp(-4π κ d / λ) in the bulk; the faces give 16 |ñ|² / |1 + ñ|⁴; the OPL
    is n_o d."""
    n_o = 1.6 + 2e-4j
    n_e = 1.5 + 1e-4j
    medium = UniaxialMaterial(
        IdealMaterial(n_o.real, n_o.imag),
        IdealMaterial(n_e.real, n_e.imag),
        optic_axis=(1.0, 0.0, 0.0),
    )
    d = 2.0
    optic = _crystal_optic([medium], [d])
    rays = _trace(optic, ["o", "T"], _rays(wavelength=0.6))
    rays.update_intensity(_state(0.0, 1.0))
    bulk = math.exp(-4 * math.pi * n_o.imag * d * 1e3 / 0.6)
    faces = 16 * abs(n_o) ** 2 / abs(1 + n_o) ** 4
    assert abs(_np(rays.i)[0] - bulk * faces) < 1e-12
    assert abs(_np(rays.opd)[0] - (6.0 + n_o.real * d)) < 1e-12


# ------------------------------------------------------------ mode labels


def test_mode_labels_calcite(set_test_backend):
    """Calcite is negative: the slow mode is o, the fast mode is e; at normal
    incidence t1 (s-like, E along y) is o."""
    axis = (math.sin(math.pi / 4), 0.0, math.cos(math.pi / 4))
    optic = _crystal_optic([_calcite(axis)], [1.0])
    x = {m: _np(_trace(optic, [m, "T"], _rays()).x)[0] for m in MODES_ONE}
    assert x["slow"] == x["o"] == x["t1"]
    assert x["fast"] == x["e"] == x["t2"]
    assert x["o"] != x["e"]
    # The default into a crystal is the slow mode.
    _set_modes(optic, [None, None])
    rays = _rays()
    optic.surfaces.trace(rays, skip=1)
    assert _np(rays.x)[0] == x["slow"]
    assert rays.branch_key == (("", "slow"), ("", "T"))


MODES_ONE = ("o", "e", "slow", "fast", "t1", "t2")


def test_mode_errors(set_test_backend):
    axis = (0.0, 0.0, 1.0)
    optic = _crystal_optic([_calcite(axis)], [1.0])
    with pytest.raises(ValueError, match="isotropic medium after"):
        _trace(optic, ["T", "T"], _rays())
    with pytest.raises(ValueError, match="after the surface is isotropic"):
        _trace(optic, ["o", "e"], _rays())
    with pytest.raises(ValueError, match="Unknown mode"):
        AnisotropicInteractionModel(parent_surface=None, mode="x")
    with pytest.raises(NotImplementedError):
        AnisotropicInteractionModel(parent_surface=None, is_reflective=True)
    biaxial = BiaxialMaterial(
        IdealMaterial(1.5), IdealMaterial(1.6), IdealMaterial(1.7)
    )
    optic = _crystal_optic([biaxial], [1.0])
    with pytest.raises(ValueError, match="UniaxialMaterial"):
        _trace(optic, ["o", "T"], _rays())


def test_biaxial_slow_and_fast(set_test_backend):
    """A biaxial plate with the principal axes on the global axes at normal
    incidence: the slow mode has index n_y (E along y), the fast mode n_x."""
    biaxial = BiaxialMaterial(
        IdealMaterial(1.5), IdealMaterial(1.6), IdealMaterial(1.7)
    )
    optic = _crystal_optic([biaxial], [2.0])
    slow = _trace(optic, ["slow", "T"], _rays())
    fast = _trace(optic, ["fast", "T"], _rays())
    assert abs(_np(slow.opd)[0] - (6.0 + 1.6 * 2.0)) < 1e-12
    assert abs(_np(fast.opd)[0] - (6.0 + 1.5 * 2.0)) < 1e-12


def test_requires_anisotropic_rays(set_test_backend):
    optic = _crystal_optic([_calcite((0.0, 0.0, 1.0))], [1.0])
    rays = PolarizedRays(*[be.array([v]) for v in (0, 0, -1, 0, 0, 1, 1, WL)])
    with pytest.raises(TypeError, match="AnisotropicRays"):
        optic.surfaces.trace(rays, skip=1)
    optic.updater.set_polarization("ignore")
    with pytest.raises(ValueError, match="tensor materials"):
        optic.trace(Hx=0, Hy=0, wavelength=WL, num_rays=1, distribution="line_y")
    rays = RealRays(*[be.array([v]) for v in (0, 0, 0, 0, 0, 1, 1, WL)])
    with pytest.raises(TypeError, match="wave vector"):
        AnisotropicPropagation().propagate(rays, 1.0)


# ------------------------------------------------------------ the optic


def test_optic_trace_through_a_displacer(set_test_backend):
    """``Optic.trace`` (paraxial aiming, ray generation, update_intensity)
    through a calcite displacer: every ray of a collimated fan walks off by
    the case 1 shift."""
    axis = (math.sin(math.pi / 4), 0.0, math.cos(math.pi / 4))
    optic = _crystal_optic([_calcite(axis)], [2.0])
    optic.updater.set_polarization(create_polarization("H"))
    assert optic.surfaces.uses_tensor_materials
    _set_modes(optic, ["e", "T"])
    rays = optic.trace(Hx=0, Hy=0, wavelength=WL, num_rays=5, distribution="line_y")
    assert isinstance(rays, AnisotropicRays)
    y_in = np.linspace(-1.0, 1.0, 5)
    assert np.max(np.abs(_np(rays.x) - (-0.1092064213 * 2.0))) < 1e-9
    assert np.max(np.abs(_np(rays.y) - y_in)) < 1e-12
    n_o, n_e = CALCITE
    n_45 = n_o * n_e / math.sqrt((n_o**2 + n_e**2) / 2)
    assert np.max(np.abs(_np(rays.i) - (4 * n_45 / (1 + n_45) ** 2) ** 2)) < 1e-12


def test_paraxial_trace_uses_the_mode_index(set_test_backend):
    """A calcite plano-convex lens (axis z, the paraxial modes are o): the
    paraxial focal length is R / (n_o - 1)."""
    optic = Optic()
    optic.surfaces.add(index=0, radius=be.inf, thickness=be.inf)
    optic.surfaces.add(
        index=1,
        radius=50.0,
        thickness=2.0,
        material=_calcite((0.0, 0.0, 1.0)),
        is_stop=True,
        interaction_type="anisotropic",
    )
    optic.surfaces.add(index=2, thickness=80.0, interaction_type="anisotropic")
    optic.surfaces.add(index=3)
    optic.set_aperture(aperture_type="EPD", value=5.0)
    optic.fields.add(y=0.0)
    optic.wavelengths.add(value=WL, is_primary=True)
    optic.updater.set_polarization(create_polarization("H"))
    optic.surfaces[1].interaction_model.mode = "o"
    f2 = float(_np(optic.paraxial.f2()))
    assert abs(f2 - 50.0 / (CALCITE[0] - 1.0)) < 1e-9


# ------------------------------------------------------------ isotropic limit


def _golden(name, tensor, lossless, state):
    """A golden system with Fresnel coatings or with isotropic tensors."""
    optic = GOLDEN_SYSTEMS[name]()
    optic.updater.set_polarization(state)

    def is_air(m):
        return type(m) is IdealMaterial

    for s in optic.surfaces:
        m = s.material_post
        if is_air(m):
            continue
        m = _Lossless(m) if lossless else m
        s.material_post = TensorMaterial(m) if tensor else m
    for s in optic.surfaces[1:]:
        if is_air(s.material_pre) and is_air(s.material_post):
            continue
        if tensor:
            s.interaction_model = AnisotropicInteractionModel(parent_surface=s)
        else:
            s.interaction_model.coating = None
            s.set_fresnel_coating()
    return optic


class _Lossless(IdealMaterial):
    """A catalog glass without its extinction coefficient."""

    def __init__(self, material):
        super().__init__(1.0)
        self.glass = material

    def _calculate_n(self, wavelength, **kwargs):
        return self.glass.n(wavelength)

    def _calculate_k(self, wavelength, **kwargs):
        return 0 * be.atleast_1d(be.asarray(wavelength))


def _compare(a, b, keys, rtol, atol):
    for key in keys:
        x, y = _np(getattr(a, key)), _np(getattr(b, key))
        assert np.array_equal(np.isnan(x), np.isnan(y)), key
        good = ~np.isnan(x)
        err = np.abs(x[good] - y[good])
        assert np.all(err <= atol + rtol * np.abs(x[good])), (key, err.max())


@pytest.mark.parametrize("name", sorted(GOLDEN_SYSTEMS))
def test_isotropic_tensors_trace_like_the_isotropic_path(set_test_backend, name):
    """Every golden system with its glasses replaced by equal isotropic tensors
    traces like the original with Fresnel coatings: positions, directions,
    OPD, power and flux factor to 1E-12 (relative, 1E-12 absolute) with the
    catalog glasses. The PRT agrees to 1E-12 when the glass extinction
    coefficient is 0. With it (κ ≈ 1E-8), the field of an absorbing medium is
    not transverse to Re k (exact plane wave) while ``JonesFresnel`` uses the
    real (s, p) frame; the PRT then differs by less than 20 κ_max."""
    state = _state(0.6, 0.8, 0.7)
    fields = ((0.0, 0.0), (0.0, 0.7), (0.3, 0.6), (0.0, 1.0))
    px = be.linspace(-0.7, 0.7, 5)
    py = be.linspace(-0.6, 0.65, 5)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        # One optic aims every ray set (its aimer caches the aiming solution).
        source = GOLDEN_SYSTEMS[name]()
        source.updater.set_polarization(state)
        generator = source.ray_tracer.ray_generator
        systems = {
            lossless: (
                _golden(name, False, lossless, state),
                _golden(name, True, lossless, state),
            )
            for lossless in (False, True)
        }
        kappa = max(
            float(np.max(_np(s.material_post.k(source.primary_wavelength))))
            for s in source.surfaces
        )
        for hx, hy in fields:
            for wl in source.wavelengths.get_wavelengths():
                for lossless, (iso, ten) in systems.items():
                    r1 = generator.generate_rays(hx, hy, px, py, wl)
                    r2 = AnisotropicRays.from_rays(r1)
                    iso.surfaces.trace(r1)
                    ten.surfaces.trace(r2)
                    r1.update_intensity(state)
                    r2.update_intensity(state)
                    keys = ("x", "y", "z", "L", "M", "N", "opd", "i")
                    _compare(r1, r2, keys, 1e-12, 1e-12)
                    finite = ~np.isnan(_np(r1.x))
                    ff1, ff2 = _np(r1.flux_factor), _np(r2.flux_factor)
                    assert np.max(np.abs(ff1 - ff2)[finite]) < 1e-12
                    p_err = np.max(np.abs(_np(r1.p) - _np(r2.p))[finite])
                    if lossless:
                        assert p_err < 1e-12
                    else:
                        assert p_err < 20 * kappa


# ------------------------------------------------------------ ray state


def test_rotation_of_the_wave_vector(set_test_backend):
    k = np.array([[0.3 + 1e-3j, -0.2, 1.4 + 2e-3j]])
    rays = AnisotropicRays(*[be.array([v]) for v in (0, 0, 0, 0, 0, 1, 1, WL)])
    rays.set_k(be.to_complex(be.array(k.real)) + 1j * be.array(k.imag))
    a, b, c = 0.3, -0.7, 1.1
    rays.rotate_x(a)
    rays.rotate_y(b)
    rays.rotate_z(c)

    def rx(t):
        return np.array(
            [[1, 0, 0], [0, math.cos(t), -math.sin(t)], [0, math.sin(t), math.cos(t)]]
        )

    def ry(t):
        return np.array(
            [[math.cos(t), 0, math.sin(t)], [0, 1, 0], [-math.sin(t), 0, math.cos(t)]]
        )

    def rz(t):
        return np.array(
            [[math.cos(t), -math.sin(t), 0], [math.sin(t), math.cos(t), 0], [0, 0, 1]]
        )

    expected = rz(c) @ ry(b) @ rx(a) @ k[0]
    assert np.max(np.abs(_np(rays.k)[0] - expected)) < 1e-15
    d = np.array([_np(rays.L)[0], _np(rays.M)[0], _np(rays.N)[0]])
    assert np.max(np.abs(d - rz(c) @ ry(b) @ rx(a) @ np.array([0, 0, 1.0]))) < 1e-15


def test_isotropic_refraction_and_reflection_update_k(set_test_backend):
    t = math.radians(30.0)
    rays = AnisotropicRays(
        *[be.array([v]) for v in (0, 0, 0, 0, math.sin(t), math.cos(t), 1, WL)]
    )
    rays.refract(be.array([0.0]), be.array([0.0]), be.array([1.0]), 1.0, 1.5)
    d = np.array([_np(rays.L)[0], _np(rays.M)[0], _np(rays.N)[0]])
    assert np.max(np.abs(_np(rays.k)[0] - 1.5 * d)) < 1e-15
    assert abs(1.5 * d[1] - math.sin(t)) < 1e-15
    rays.reflect(be.array([0.0]), be.array([0.0]), be.array([1.0]))
    assert np.max(np.abs(_np(rays.k)[0] - 1.5 * d * np.array([1, 1, -1]))) < 1e-15


def test_optical_path_and_attenuation(set_test_backend):
    """Re(k) · Δr and exp(-2 k0 Im(k) · Δr) along the ray direction S."""
    s = np.array([0.1, -0.2, math.sqrt(1 - 0.05)])
    k = np.array([0.2 + 1e-5j, 0.1, 1.5 + 2e-5j])
    rays = AnisotropicRays(
        *[be.array([v]) for v in (0, 0, 0, s[0], s[1], s[2], 1, 0.5)]
    )
    rays.set_k(be.to_complex(be.array([k.real])) + 1j * be.array([k.imag]))
    t = 3.0
    assert abs(_np(rays.optical_path(t))[0] - t * float(np.dot(k.real, s))) < 1e-15
    expected = math.exp(-2 * (2 * math.pi / 0.5) * float(np.dot(k.imag, s)) * t * 1e3)
    assert abs(_np(rays.attenuation(t))[0] - expected) < 1e-15


def test_from_rays_copies_the_state(set_test_backend):
    rays = PolarizedRays(
        *[be.array([v, v]) for v in (0.1, 0.2, 0.3, 0, 0.6, 0.8, 1, WL)]
    )
    rays.opd = be.array([1.0, 2.0])
    rays.flux_factor = be.array([0.5, 0.25])
    new = AnisotropicRays.from_rays(rays)
    for key in ("x", "y", "z", "L", "M", "N", "i", "w", "opd", "p", "flux_factor"):
        assert np.array_equal(_np(getattr(new, key)), _np(getattr(rays, key)))
    assert np.max(np.abs(_np(new.k) - np.array([[0, 0.6, 0.8]] * 2))) == 0.0
    assert new.mode is None and new.branch_key == ()


def test_model_round_trip():
    model = AnisotropicInteractionModel(parent_surface=None, mode="e", label="s1")
    data = model.to_dict()
    assert data["type"] == "AnisotropicInteractionModel"
    new = AnisotropicInteractionModel.from_dict(data, None)
    assert isinstance(new, AnisotropicInteractionModel)
    assert new.mode == "e" and new.label == "s1"
    medium = _calcite((0.0, 0.0, 1.0))
    assert isinstance(medium.propagation_model, AnisotropicPropagation)
