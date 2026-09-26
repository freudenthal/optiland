"""Tests of optically active and bianisotropic media (both backends).

Held data and closed forms:

* Quartz rotatory power along the optic axis, traced through a z-cut plate,
  against the measurements of Lowry and Coode-Adams, Phil. Trans. R. Soc. A
  226, 391-466 (1927): Table III (p. 399, the sodium D lines, "Mean"), Table
  VIII (p. 460, the cadmium line 3403.6529 Å of Meggers and Burns, ultraviolet)
  and Table V (p. 412, 2.0000 µm, infrared); wavelengths in air, 20 °C. The
  stated uncertainties: formula (vi) ±0.002 °/mm (p. 395), the infrared
  readings 0.004 °/mm (p. 402), the ultraviolet casual error 0.005 °/mm
  (p. 458). Also Lowry, Phil. Trans. A 212, 261-297 (1913), p. 293:
  21.7283 °/mm at the sodium mass centre (vacuum wavelength 0.58940502 µm).
* A linear polarization turns by k0 κ d; right quartz
  turns x̂ toward -ŷ along +ẑ.
* Quartz near the optic axis: circular modes on the axis (the slow mode is
  L = (1, i)); the index split √(Δ² + δ²) and the circularity δ / √(Δ² + δ²)
  of the first-order theory (Δ = n_e(θ) - n_o, δ = γ33 cos²θ + γ11 sin²θ);
  E is linear where γ11/γ33 = -(n_e/n_o)² cot²θ (Eimerl, JOSA B 5, 1453
  (1988), Eq. (70), exact for E), D is not linear there.
* A Pasteur slab (Bassiri, Papas and Engheta, JOSA A 5, 1450 (1988)): two
  refracted waves with sin θ± = sin θ / n±, n± = √ε ± κ; the faces transmit
  as a dielectric of index √ε at normal incidence.
* Reciprocity: for a reciprocal medium the Jones matrix of the reverse pass
  is Q Jᵀ Q (Q = diag(1, -1)); the plate turned by 180° about x is the
  reverse pass. A gyrotropic (Faraday) plate is not reciprocal: the rotation
  doubles on the round trip.
* Energy: the power ledger of a lossless plate closes to 1E-12.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
from scipy.optimize import brentq

import optiland.backend as be
from optiland.anisotropic import constitutive_matrix, plane_wave_modes, solve_interface
from optiland.materials import IdealMaterial, Material
from optiland.materials.anisotropic import (
    BianisotropicMaterial,
    QuartzMaterial,
    TensorMaterial,
    euler_zxz_matrix,
    kappa_from_rotatory_power,
    quartz_rotatory_power,
    standard_air_index,
)
from optiland.raytrace.branches import BranchTracer
from optiland.rays import create_polarization

from .test_tracing import _crystal_optic, _np, _rays

TURN = np.diag([1.0, -1.0, -1.0])  # Rx(180°): the reverse pass of a plate
Q = np.diag([1.0, -1.0])

# (label, wavelength in air in µm, observed ρ °/mm, uncertainty °/mm)
LOWRY_1927 = [
    ("D1", 0.5895932, 21.7010, 0.002),  # Table III, p. 399
    ("D2", 0.5889965, 21.7492, 0.002),  # Table III, p. 399
    ("Cd UV", 0.34036529, 72.455, 0.005),  # Table VIII, p. 460
    ("IR", 2.0000, 1.631, 0.004),  # Table V, p. 412
]


def _vacuum(air_um: float) -> float:
    """The vacuum wavelength of a line given in standard air."""
    w = air_um
    for _ in range(8):
        w = float(_np(standard_air_index(w))) * air_um
    return w


def _k0_mm(wavelength_um: float) -> float:
    return 2 * math.pi / (wavelength_um * 1e-3)


def _jones(optic, wavelength: float, theta: float = 0.0) -> np.ndarray:
    """The direct-pass Jones matrix of a plate in air at normal incidence.

    The coherent sum of the transmitted branches for an x and a y input, in the
    global (x, y) frame, shape (2, 2).
    """
    optic.updater.set_polarization(create_polarization("unpolarized"))
    result = BranchTracer(optic).trace_all(rays=_rays(theta, wavelength=wavelength))
    fields = result.coherent_fields()
    return np.stack([_np(f)[0, :2] for f in fields], axis=-1)


def _rotation_deg(jones_column: np.ndarray) -> tuple[float, float]:
    """The azimuth (degrees) of a linear state and |Im(E_y / E_x)|."""
    ratio = jones_column[1] / jones_column[0]
    return math.degrees(math.atan(ratio.real)), abs(ratio.imag)


# ------------------------------------------------------------ the preset


def test_standard_air_index(set_test_backend):
    """Na D1: 589.5924 nm in air, 589.7558 nm in vacuum (NIST Atomic Spectra
    Database); the index is their ratio to the rounding of the digits."""
    n = float(_np(standard_air_index(be.asarray(0.5897558))))
    assert abs(n - 589.7558 / 589.5924) < 2e-7


def test_quartz_preset(set_test_backend):
    """κ = ρ λ0 / (2π) with the sign of the hand; the preset is lossless,
    reciprocal and serializable; the hand is checked."""
    right, left = QuartzMaterial(), QuartzMaterial(hand="left")
    w = 0.58940502
    rho = float(_np(quartz_rotatory_power(w)))
    kappa = float(_np(kappa_from_rotatory_power(rho, w)))
    assert abs(float(_np(right.kappa(w))[0]) - kappa) < 1e-20
    assert abs(float(_np(left.kappa(w))[0]) + kappa) < 1e-20
    assert right.is_transparent(w) and right.is_reciprocal(w)
    copy = QuartzMaterial.from_dict(right.to_dict())
    assert copy == right
    assert (
        np.max(np.abs(_np(copy.constitutive_6x6(w)) - _np(right.constitutive_6x6(w))))
        == 0
    )
    with pytest.raises(ValueError, match="hand"):
        QuartzMaterial(hand="dextro")


# ------------------------------------------------------------ rotatory power


@pytest.mark.parametrize(
    "label, air_um, observed, tol", LOWRY_1927, ids=[x[0] for x in LOWRY_1927]
)
def test_quartz_rotatory_power_traced(set_test_backend, label, air_um, observed, tol):
    """A 1 mm z-cut plate of right quartz turns x̂ toward -ŷ by the measured
    ρ d within the stated uncertainty; left quartz turns it the other way; the
    exit state is linear and the angle is k0 κ d."""
    w = _vacuum(air_um)
    d = 1.0
    for hand, sign in (("right", 1.0), ("left", -1.0)):
        material = QuartzMaterial(hand=hand)
        jones = _jones(_crystal_optic([material], [d]), w)
        psi, ellipticity = _rotation_deg(jones[:, 0])
        assert ellipticity < 1e-9
        kappa = float(_np(material.kappa(w))[0])
        assert abs(math.radians(psi) + _k0_mm(w) * kappa * d) < 1e-9
        assert abs(-psi / d - sign * observed) < tol, (label, hand, -psi / d)


def test_quartz_rotatory_power_lowry_1913(set_test_backend):
    """Lowry 1913: 21.7283 °/mm at λ0 = 0.58940502 µm (±0.002 °/mm)."""
    jones = _jones(_crystal_optic([QuartzMaterial()], [1.0]), 0.58940502)
    psi, _ = _rotation_deg(jones[:, 0])
    assert abs(-psi - 21.7283) < 0.002


# ------------------------------------------------------------ near the axis


def _quartz_modes(theta: float, wavelength: float = 0.5893):
    matrix = constitutive_matrix(QuartzMaterial(), wavelength)
    s = np.array([[math.sin(theta), 0.0, math.cos(theta)]])
    modes = plane_wave_modes(be.asarray(s), matrix, x_ref=be.asarray([[0.0, 1.0, 0.0]]))
    return s[0], _np(matrix)[0], modes


def _circularity(theta: float, field: str) -> float:
    """S3/S0 of the E or the D field of the slow mode in the frame (ŷ, ŷ × ŝ, ŝ)."""
    s, m, modes = _quartz_modes(theta)
    index = np.real(_np(modes.index)[0])
    slow = int(np.argmax(index))
    e, h = _np(modes.E)[0, slow], _np(modes.H)[0, slow]
    v = e if field == "E" else m[:3, :3] @ e + m[:3, 3:] @ h
    e1 = np.array([0.0, 1.0, 0.0])
    a, b = v @ e1, v @ np.cross(e1, s)
    return float(2 * np.imag(a * np.conj(b)) / (abs(a) ** 2 + abs(b) ** 2))


def test_quartz_modes_near_the_axis(set_test_backend):
    """On the axis the slow mode of right quartz is L = (1, i) and
    n_{L,R} = √(n_o² + κ²) ± κ; off the axis the split and the circularity
    follow the first-order theory; E is linear at the Eimerl (70) angle, D is not."""
    w = 0.5893
    n_o = float(np.ravel(_np(Material("SiO2", reference="Ghosh-o").n(w)))[0])
    n_e = float(np.ravel(_np(Material("SiO2", reference="Ghosh-e").n(w)))[0])
    kappa = float(_np(QuartzMaterial().kappa(w))[0])
    g33, g11 = 2 * kappa, -0.525 * 2 * kappa

    _, _, modes = _quartz_modes(0.0)
    index = np.real(_np(modes.index)[0])
    slow = int(np.argmax(index))
    root = math.sqrt(n_o**2 + kappa**2)
    assert abs(index[slow] - (root + kappa)) < 1e-14
    assert abs(index[1 - slow] - (root - kappa)) < 1e-14
    e = _np(modes.E)[0, slow]
    # L: counter-clockwise facing the source. The eig vectors of a pair split by
    # 2κ carry about 1E-16 |Δ| / 2κ: 1E-9.
    assert abs(e[1] / e[0] - 1j) < 1e-9

    for degrees in (0.5, 1.0, 2.0, 5.0, 10.0, 30.0):
        t = math.radians(degrees)
        index = np.real(_np(_quartz_modes(t)[2].index)[0])
        n_theta = 1 / math.sqrt(math.cos(t) ** 2 / n_o**2 + math.sin(t) ** 2 / n_e**2)
        linear = n_theta - n_o
        circular = g33 * math.cos(t) ** 2 + g11 * math.sin(t) ** 2
        split = math.hypot(linear, circular)
        assert abs(abs(index[0] - index[1]) / split - 1) < 1e-4
        assert abs(_circularity(t, "D") - circular / split) < 2e-4

    t_eimerl = math.atan(math.sqrt((n_e / n_o) ** 2 / 0.525))
    t_e = brentq(lambda t: _circularity(t, "E"), 0.8, 1.1, xtol=1e-14)
    t_d = brentq(lambda t: _circularity(t, "D"), 0.8, 1.1, xtol=1e-14)
    assert abs(t_e - t_eimerl) < 1e-9
    assert abs(t_d - t_eimerl) > 1e-3


# ------------------------------------------------------------ reflection


def _dcr(material_b, theta: float, wavelength: float = 0.5893) -> float:
    """(R_L - R_R) / (R_L + R_R) from air, L = p + i s, incidence in x-z."""
    k_in = be.asarray([[math.sin(theta), 0.0, math.cos(theta)]])
    s = np.array([0.0, 1.0, 0.0])
    p = np.array([math.cos(theta), 0.0, -math.sin(theta)])
    reflectance = {}
    for name, sign in (("L", 1.0), ("R", -1.0)):
        e_in = be.to_complex(be.asarray(p[None])) + 1j * sign * be.to_complex(
            be.asarray(s[None])
        )
        res = solve_interface(
            be.asarray([0.0, 0.0, 1.0]),
            constitutive_matrix(IdealMaterial(1.0), wavelength),
            constitutive_matrix(material_b, wavelength),
            k_in,
            e_in / math.sqrt(2),
        )
        reflectance[name] = float(_np(res.reflectance)[0])
    return (reflectance["L"] - reflectance["R"]) / (reflectance["L"] + reflectance["R"])


def test_differential_circular_reflection(set_test_backend):
    """Z-cut quartz at normal incidence: DCR = 0 in the Tellegen
    form (the Landau matching would give 4κ/(n² - 1) = 1E-4). Silverman and
    Badoz, JOSA A 7, 1163 (1990): ε = η² diag(b, a, c), ξ = iκI, κ = f η,
    f = 1E-4: DCR = -0.0180641 f at 0° (η 1.5, a 1, b 1.2, c 1) and
    0.8914420 f at the Brewster angle 54.636° (c = 1.2)."""
    assert abs(_dcr(QuartzMaterial(), 0.0)) < 1e-12
    eta, f = 1.5, 1e-4

    def silverman(a: float, b: float, c: float) -> BianisotropicMaterial:
        return BianisotropicMaterial.from_optical_activity(
            eta**2 * np.diag([b, a, c]), f * eta * np.eye(3), "tellegen"
        )

    assert abs(_dcr(silverman(1.0, 1.2, 1.0), 0.0) / f + 0.0180641) < 5e-8
    brewster = math.acos(math.sqrt((eta**2 * 1.2 - 1) / (eta**4 * 1.2 - 1)))
    assert abs(math.degrees(brewster) - 54.636) < 5e-4
    assert abs(_dcr(silverman(1.0, 1.0, 1.2), brewster) / f - 0.8914420) < 5e-8


# ------------------------------------------------------------ Pasteur slab


def _pasteur(epsilon: float, kappa: float, rotation=None) -> BianisotropicMaterial:
    return BianisotropicMaterial.from_optical_activity(
        epsilon * np.eye(3), kappa * np.eye(3), "tellegen", rotation=rotation
    )


def test_pasteur_slab(set_test_backend):
    """A chiral slab: at normal incidence x̂ turns by k0 κ d and the power is
    (4n / (1 + n)²)², n = √ε; at 40° the two refracted waves follow
    sin θ± = sin θ / n± and leave parallel to the input, displaced by d tan θ±."""
    eps, kappa, d, w = 2.25, 1e-4, 0.5, 0.5893
    n = math.sqrt(eps)
    material = _pasteur(eps, kappa)
    jones = _jones(_crystal_optic([material], [d]), w)
    psi, ellipticity = _rotation_deg(jones[:, 0])
    assert ellipticity < 1e-9
    assert abs(math.radians(psi) + _k0_mm(w) * kappa * d) < 1e-9
    assert abs(np.sum(np.abs(jones[:, 0]) ** 2) - (4 * n / (1 + n) ** 2) ** 2) < 1e-12

    theta = math.radians(40.0)
    k_in = be.asarray([[math.sin(theta), 0.0, math.cos(theta)]])
    res = solve_interface(
        be.asarray([0.0, 0.0, 1.0]),
        constitutive_matrix(IdealMaterial(1.0), w),
        constitutive_matrix(material, w),
        k_in,
        be.asarray([[0.0, 1.0, 0.0]]),
    )
    k_t = np.real(_np(res.k)[0, 2:])
    angles = sorted(math.asin(k[0] / np.linalg.norm(k)) for k in k_t)
    expected = sorted(math.asin(math.sin(theta) / (n + s * kappa)) for s in (1, -1))
    assert np.max(np.abs(np.array(angles) - expected)) < 1e-12

    optic = _crystal_optic([material], [d], image_gap=0.0)
    optic.updater.set_polarization(create_polarization("unpolarized"))
    result = BranchTracer(optic).trace_all(rays=_rays(theta, z0=-1.0))
    assert len(result) == 2
    for key, branch in result.branches.items():
        n_mode = n + kappa if key[0][2] == "slow" else n - kappa
        t_mode = math.asin(math.sin(theta) / n_mode)
        direction = np.array(
            [_np(branch.rays.L)[0], _np(branch.rays.M)[0], _np(branch.rays.N)[0]]
        )
        assert (
            np.max(np.abs(direction - [math.sin(theta), 0.0, math.cos(theta)])) < 1e-12
        )
        exit_x = float(_np(branch.views[-2].x)[0])
        assert abs(exit_x - d * math.tan(t_mode)) < 1e-12


# ------------------------------------------------------------ reciprocity


def _faraday(epsilon: float, g: float, rotation=None) -> TensorMaterial:
    """A gyrotropic (magneto-optic) medium, ε = [[e, ig, 0], [-ig, e, 0], [0, 0, e]]."""
    tensor = np.array(
        [[epsilon, 1j * g, 0.0], [-1j * g, epsilon, 0.0], [0.0, 0.0, epsilon]]
    )
    return TensorMaterial(tensor, rotation=rotation)


RECIPROCAL = {
    "pasteur": lambda r: _pasteur(2.25, 1e-4, r),
    "quartz": lambda r: QuartzMaterial(rotation=r),
    "active biaxial": lambda r: BianisotropicMaterial.from_optical_activity(
        np.diag([2.2, 2.3, 2.45]),
        np.array([[3e-5, 1e-5, 0.0], [1e-5, -2e-5, 2e-5], [0.0, 2e-5, 4e-5]]),
        rotation=r,
    ),
}


@pytest.mark.parametrize("name", list(RECIPROCAL))
def test_reciprocity_of_reciprocal_media(set_test_backend, name):
    """The plate turned by 180° about x (the reverse pass) has the Jones
    matrix Q Jᵀ Q of the forward pass, in two orientations. Tolerance 1E-10:
    the plate phase k0 OPL is about 1E4 rad (measured 1.4E-11)."""
    w, d = 0.5893, 0.7
    for angles in ((0.0, 0.0, 0.0), (20.0, 35.0, -50.0)):
        r = euler_zxz_matrix(*angles)
        forward = _jones(_crystal_optic([RECIPROCAL[name](r)], [d]), w)
        reverse = _jones(_crystal_optic([RECIPROCAL[name](TURN @ r)], [d]), w)
        assert np.max(np.abs(reverse - Q @ forward.T @ Q)) < 1e-10, angles


def _faraday_ratio(n_l: float, n_r: float, k0d: float) -> complex:
    """E_y / E_x behind a Faraday plate at normal incidence for an x input.

    x̂ = ((1, i) + (1, -i)) / 2; each circular mode crosses two faces
    (T = 4n / (1 + n)², the impedance 1/n of the mode) and the plate
    (exp(i k0 n d)).
    """
    a_l = 4 * n_l / (1 + n_l) ** 2 * np.exp(1j * k0d * n_l)
    a_r = 4 * n_r / (1 + n_r) ** 2 * np.exp(1j * k0d * n_r)
    return complex(1j * (a_l - a_r) / (a_l + a_r))


def test_faraday_plate_is_not_reciprocal(set_test_backend):
    """g along z: (1, ±i) with n_{L,R} = √(e ∓ g); x̂ turns toward +ŷ by about
    k0 (n_R - n_L) d / 2, slightly elliptical (the modes have different face
    transmissions). The reverse pass has -g in its own frame: it turns the other
    way there, so the round trip doubles the rotation and Q Jᵀ Q ≠ J_reverse.
    A tilted g is not reciprocal either."""
    e, g, d, w = 2.25, 2e-5, 0.7, 0.5893
    n_l, n_r = math.sqrt(e - g), math.sqrt(e + g)
    k0d = _k0_mm(w) * d
    forward = _jones(_crystal_optic([_faraday(e, g)], [d]), w)
    reverse = _jones(_crystal_optic([_faraday(e, g, TURN)], [d]), w)
    assert abs(forward[1, 0] / forward[0, 0] - _faraday_ratio(n_l, n_r, k0d)) < 1e-10
    assert abs(reverse[1, 0] / reverse[0, 0] - _faraday_ratio(n_r, n_l, k0d)) < 1e-10
    psi_f, _ = _rotation_deg(forward[:, 0])
    psi_r, _ = _rotation_deg(reverse[:, 0])
    rotation = math.degrees(k0d * (n_r - n_l) / 2)
    assert abs(psi_f - rotation) < 1e-3 * rotation
    assert abs(psi_r + rotation) < 1e-3 * rotation
    assert np.max(np.abs(reverse - Q @ forward.T @ Q)) > 0.1 * math.radians(rotation)
    r = euler_zxz_matrix(20.0, 35.0, -50.0)
    forward = _jones(_crystal_optic([_faraday(e, g, r)], [d]), w)
    reverse = _jones(_crystal_optic([_faraday(e, g, TURN @ r)], [d]), w)
    assert np.max(np.abs(reverse - Q @ forward.T @ Q)) > 1e-3


# ------------------------------------------------------------ energy


@pytest.mark.parametrize("name", ["pasteur", "quartz", "faraday"])
def test_energy_balance(set_test_backend, name):
    """Every reflection followed (two at most): the ledger of a lossless
    tilted plate closes at 0° and 30° and nothing is absorbed."""
    r = euler_zxz_matrix(20.0, 35.0, -50.0)
    material = _faraday(2.25, 2e-5, r) if name == "faraday" else RECIPROCAL[name](r)
    for theta in (0.0, math.radians(30.0)):
        optic = _crystal_optic([material], [2.0])
        optic.updater.set_polarization(create_polarization("unpolarized"))
        tracer = BranchTracer(optic, ghosts="all", max_reflections=2, threshold=0.0)
        ledger = tracer.trace_all(rays=_rays(theta)).ledger
        assert abs(ledger.absorbed) < 1e-12
        assert abs(ledger.total() - 1.0) < 1e-12
