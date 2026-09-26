"""Tests of the tensor and bianisotropic materials (closed forms, both backends)."""

from __future__ import annotations

import math

import numpy as np
import pytest

import optiland.backend as be
from optiland.materials import IdealMaterial, Material
from optiland.materials.anisotropic import (
    BaseTensorMaterial,
    BianisotropicMaterial,
    BiaxialMaterial,
    TensorMaterial,
    UniaxialMaterial,
    alpha_from_gyration,
    euler_zxz_matrix,
    gyration_from_alpha,
    kappa_from_rotatory_power,
    quartz_alpha,
)

from ..utils import assert_allclose


def _np(x):
    return np.asarray(be.to_numpy(x))


def _cross_matrix(v):
    return np.array([[0.0, -v[2], v[1]], [v[2], 0.0, -v[0]], [-v[1], v[0], 0.0]])


def _calcite():
    return Material("CaCO3", reference="Ghosh-o"), Material(
        "CaCO3", reference="Ghosh-e"
    )


def _ktp():
    return (
        Material("KTiOPO4", reference="Kato-alpha"),
        Material("KTiOPO4", reference="Kato-beta"),
        Material("KTiOPO4", reference="Kato-gamma"),
    )


def _quartz():
    return Material("SiO2", reference="Ghosh-o"), Material("SiO2", reference="Ghosh-e")


# --------------------------------------------------------------------- uniaxial


def test_uniaxial_calcite_against_ghosh_table(set_test_backend):
    """Ghosh 1999, Tables 2 and 3, calcite at 0.5890 µm.

    The eigenvalues of ε are n_o² (twice) and n_e². Equation column: 1.658364,
    1.486139 (6 decimals, so 6E-7). Measured column: 1.65835, 1.48640; the
    paper gives the differences -0.000014 and 0.000261 (so 1.5E-5, 2.7E-4).
    """
    ordinary, extraordinary = _calcite()
    material = UniaxialMaterial(ordinary, extraordinary, optic_axis=(1.0, 0.0, 1.0))
    eps = _np(material.epsilon(0.5890))
    assert eps.shape == (1, 3, 3)
    values = np.sqrt(np.linalg.eigvalsh(eps[0]))  # ascending: n_e < n_o
    assert_allclose(values, [1.486139, 1.658364, 1.658364], rtol=0, atol=6e-7)
    assert_allclose(values, [1.48640, 1.65835, 1.65835], rtol=0, atol=2.7e-4)
    assert abs(values[2] - 1.65835) <= 1.5e-5
    # The eigenvector of n_e is the optic axis.
    _, vectors = np.linalg.eigh(eps[0])
    axis = np.array([1.0, 0.0, 1.0]) / math.sqrt(2)
    assert abs(abs(vectors[:, 0] @ axis) - 1.0) < 1e-12


def test_uniaxial_closed_form(set_test_backend):
    """ε = ε_o I + Δε ĉĉᵀ (Lekner 1991, Eq. 18): ĉᵀεĉ = ε_e, vᵀεv = ε_o for v ⊥ ĉ,
    and the extraordinary condition kᵀεk = ε_o ε_e for n(θ) (Lekner 1991, Eq. 22)."""
    ordinary, extraordinary = _calcite()
    wavelengths = np.array([0.4, 0.5893, 1.064])
    axis = np.array([0.3, -0.5, 0.8])
    axis = axis / np.linalg.norm(axis)
    material = UniaxialMaterial(ordinary, extraordinary, optic_axis=axis)
    eps = _np(material.epsilon(be.asarray(wavelengths)))
    assert eps.shape == (3, 3, 3)
    n_o = _np(ordinary.n(be.asarray(wavelengths)))
    n_e = _np(extraordinary.n(be.asarray(wavelengths)))
    perpendicular = np.cross(axis, [1.0, 0.0, 0.0])
    perpendicular /= np.linalg.norm(perpendicular)
    for i in range(3):
        assert abs(axis @ eps[i] @ axis - n_e[i] ** 2) < 1e-13
        assert abs(perpendicular @ eps[i] @ perpendicular - n_o[i] ** 2) < 1e-13
        theta = 0.7
        k_hat = math.cos(theta) * axis + math.sin(theta) * perpendicular
        n_theta = 1 / math.sqrt(
            math.cos(theta) ** 2 / n_o[i] ** 2 + math.sin(theta) ** 2 / n_e[i] ** 2
        )
        k = n_theta * k_hat
        assert abs(k @ eps[i] @ k - n_o[i] ** 2 * n_e[i] ** 2) < 1e-12


def test_uniaxial_with_loss_uses_complex_index(set_test_backend):
    ordinary = IdealMaterial(1.6, 0.01)
    extraordinary = IdealMaterial(1.5, 0.002)
    material = UniaxialMaterial(ordinary, extraordinary)
    eps = _np(material.epsilon(0.6))[0]
    assert_allclose(
        eps, np.diag([(1.6 + 0.01j) ** 2, (1.6 + 0.01j) ** 2, (1.5 + 0.002j) ** 2])
    )
    assert not material.is_transparent(0.6)
    assert material.is_reciprocal(0.6)


# --------------------------------------------------------------- rotation, biaxial


def test_euler_zxz_matrix():
    r = euler_zxz_matrix(30.0, 40.0, 50.0)
    assert_allclose(r @ r.T, np.eye(3), rtol=0, atol=1e-15)
    assert abs(np.linalg.det(r) - 1.0) < 1e-15
    assert_allclose(
        euler_zxz_matrix(-50.0, -40.0, -30.0) @ r, np.eye(3), rtol=0, atol=1e-15
    )
    # Rz(90°) turns x̂ into ŷ.
    assert_allclose(
        euler_zxz_matrix(90.0, 0.0, 0.0) @ [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], atol=1e-15
    )
    # Rx(90°) turns ẑ into -ŷ.
    assert_allclose(
        euler_zxz_matrix(0.0, 90.0, 0.0) @ [0.0, 0.0, 1.0], [0.0, -1.0, 0.0], atol=1e-15
    )


def test_rotation_invariants_biaxial(set_test_backend):
    """Eigenvalues and trace do not change with the orientation; ε = R ε_c Rᵀ."""
    x, y, z = _ktp()
    rotation = euler_zxz_matrix(17.0, 63.0, -121.0)
    crystal = BiaxialMaterial(x, y, z)
    rotated = BiaxialMaterial(x, y, z, rotation=rotation)
    wavelengths = be.asarray([0.6328, 1.064])
    eps_c = _np(crystal.epsilon(wavelengths))
    eps_g = _np(rotated.epsilon(wavelengths))
    for i in range(2):
        assert_allclose(
            np.linalg.eigvalsh(eps_g[i]),
            np.sort(np.diag(eps_c[i]).real),
            rtol=0,
            atol=1e-14,
        )
        assert abs(np.trace(eps_g[i]) - np.trace(eps_c[i])) < 1e-14
        assert_allclose(eps_g[i], rotation @ eps_c[i] @ rotation.T, rtol=0, atol=1e-14)
        assert_allclose(eps_g[i], eps_g[i].T, rtol=0, atol=1e-15)
    n_alpha = _np(x.n(wavelengths))
    assert_allclose(np.sqrt(eps_c[:, 0, 0].real), n_alpha, rtol=0, atol=1e-15)
    assert rotated.is_transparent(wavelengths)
    assert rotated.is_reciprocal(wavelengths)
    assert not rotated.is_isotropic(wavelengths)


def test_biaxial_optic_axes(set_test_backend):
    """Along a binormal (x-z plane, ±V from z) the section of the index
    ellipsoid is a circle of radius n_β: the plane ⊥ û has ε⁻¹ = I/n_β²."""
    x, y, z = _ktp()
    material = BiaxialMaterial(x, y, z)
    eps = _np(material.epsilon(0.6328))[0].real
    n_a, n_b, n_g = np.sqrt(np.diag(eps))
    assert n_a < n_b < n_g
    v = math.atan((n_g / n_a) * math.sqrt((n_b**2 - n_a**2) / (n_g**2 - n_b**2)))
    inverse = np.linalg.inv(eps)
    for sign in (1.0, -1.0):
        u = np.array([sign * math.sin(v), 0.0, math.cos(v)])
        e1 = np.array([0.0, 1.0, 0.0])
        e2 = np.cross(u, e1)
        section = np.array([[a @ inverse @ b for b in (e1, e2)] for a in (e1, e2)])
        assert_allclose(section, np.eye(2) / n_b**2, rtol=0, atol=1e-15)


def test_invalid_rotation_raises():
    x, y, z = _ktp()
    with pytest.raises(ValueError, match="orthogonal"):
        BiaxialMaterial(x, y, z, rotation=np.diag([1.0, 1.0, 2.0]))
    with pytest.raises(ValueError, match="proper"):
        BiaxialMaterial(x, y, z, rotation=np.diag([1.0, 1.0, -1.0]))
    with pytest.raises(ValueError, match="optic_axis"):
        UniaxialMaterial(x, y, optic_axis=(0.0, 0.0, 0.0))


# --------------------------------------------------------------- general tensors


def test_tensor_material_constant_and_callable(set_test_backend):
    constant = np.array([[2.25, 0.1j, 0.0], [-0.1j, 2.3, 0.0], [0.0, 0.0, 2.4 + 0.01j]])
    material = TensorMaterial(constant)
    eps = _np(material.epsilon(be.asarray([0.5, 0.6])))
    assert eps.shape == (2, 3, 3)
    assert_allclose(eps[1], constant, rtol=0, atol=0)
    assert not material.is_transparent(0.5)  # Im ε_zz > 0
    assert not material.is_reciprocal(0.5)  # ε ≠ εᵀ (gyrotropic)

    def dispersive(w):
        eps = be.to_complex(be.zeros((be.size(w), 3, 3)))
        eps[:, 0, 0] = 2.0 + 0.01 / w**2
        eps[:, 1, 1] = 2.1 + 0.01 / w**2
        eps[:, 2, 2] = 2.2 + 0.01 / w**2
        return eps

    rotation = euler_zxz_matrix(10.0, 20.0, 30.0)
    material = TensorMaterial(dispersive, mu=np.eye(3), rotation=rotation)
    eps = _np(material.epsilon(be.asarray([0.5, 1.0])))
    expected = rotation @ np.diag([2.04, 2.14, 2.24]) @ rotation.T
    assert_allclose(eps[0], expected, rtol=0, atol=1e-14)
    assert material.is_transparent([0.5, 1.0])
    with pytest.raises(TypeError, match="callable"):
        material.to_dict()


def test_tensor_material_bad_shape():
    material = TensorMaterial(lambda w: np.ones((2, 2)))
    with pytest.raises(ValueError, match="shape"):
        material.epsilon(0.5)
    with pytest.raises(ValueError, match="3x3"):
        TensorMaterial(np.ones(4))


def test_scalar_source_and_scalar_materials_unchanged(set_test_backend):
    glass = IdealMaterial(1.5)
    material = TensorMaterial(glass)
    assert_allclose(
        _np(material.epsilon(0.55))[0], 2.25 * np.eye(3), rtol=0, atol=1e-15
    )
    assert material.is_isotropic(0.55)
    assert_allclose(glass.n(0.55), 1.5)


def test_tensor_material_has_no_scalar_index():
    ordinary, extraordinary = _calcite()
    material = UniaxialMaterial(ordinary, extraordinary)
    with pytest.raises(TypeError, match="tensor material"):
        material.n(0.5893)
    with pytest.raises(TypeError, match="tensor material"):
        material.k(0.5893)
    assert isinstance(material, BaseTensorMaterial)


def test_lossless_media_are_hermitian(set_test_backend):
    """Lossless media: the 6x6 matrix [[ε, ξ], [ζ, μ]] is Hermitian."""
    x, y, z = _ktp()
    ordinary, extraordinary = _quartz()
    kappa = kappa_from_rotatory_power(21.7283, 0.5894050191)
    media = [
        UniaxialMaterial(*_calcite(), optic_axis=(0.2, 0.3, 0.9)),
        BiaxialMaterial(x, y, z, rotation=euler_zxz_matrix(5.0, 50.0, 95.0)),
        BianisotropicMaterial.from_optical_activity(
            UniaxialMaterial(ordinary, extraordinary),
            quartz_alpha(kappa),
            rotation=euler_zxz_matrix(0.0, 35.0, 0.0),
        ),
    ]
    for material in media:
        m = _np(material.constitutive_6x6(be.asarray([0.55, 0.6328])))
        assert m.shape == (2, 6, 6)
        assert np.max(np.abs(m - np.conj(np.swapaxes(m, -1, -2)))) < 1e-15
        assert material.is_transparent([0.55, 0.6328])
        assert material.is_reciprocal([0.55, 0.6328])


# ------------------------------------------------------------ optical activity


def test_kappa_from_rotatory_power(set_test_backend):
    """21.7283 °/mm (Lowry 1913) at λ0 = 589.4050191 nm (vacuum wavelength)."""
    kappa = kappa_from_rotatory_power(21.7283, 0.5894050191)
    assert abs(float(kappa) - 3.5574359e-05) < 5e-13


def test_gyration_alpha_map(set_test_backend):
    """α = (tr γ/2) I - γᵀ and γ = (tr α) I - αᵀ are inverse maps; quartz form."""
    rng = np.random.default_rng(50)
    gamma = rng.normal(size=(4, 3, 3)) * 1e-4
    alpha = _np(alpha_from_gyration(be.asarray(gamma)))
    for i in range(4):
        assert_allclose(
            alpha[i],
            np.trace(gamma[i]) / 2 * np.eye(3) - gamma[i].T,
            rtol=0,
            atol=1e-18,
        )
    assert_allclose(
        _np(gyration_from_alpha(be.asarray(alpha))), gamma, rtol=0, atol=1e-18
    )
    kappa = 3.5574359e-05
    quartz = _np(quartz_alpha(kappa))
    assert_allclose(
        quartz, np.diag([kappa, kappa, kappa * (2 * -0.525 - 1)]), rtol=0, atol=1e-20
    )
    gyration = _np(gyration_from_alpha(be.asarray(quartz)))
    assert abs(gyration[2, 2] - 2 * kappa) < 1e-20  # γ33 = 2κ
    assert abs(gyration[0, 0] / gyration[2, 2] + 0.525) < 1e-12


def test_tellegen_and_landau_wave_equations_agree(set_test_backend):
    """For ξ = iα, ζ = -iαᵀ and ε_T = ε_L + ααᵀ the
    Tellegen wave matrix K(K - ζ) + ε_T + ξ(K - ζ) equals the Landau matrix
    KK + ε_L + i[γk]x for every k (K = [k]x), so the bulk modes are the same."""
    rng = np.random.default_rng(17)
    for _ in range(10):
        a = rng.normal(size=(3, 3))
        eps_l = a @ a.T + 2 * np.eye(3)
        gamma = rng.normal(size=(3, 3)) * 1e-2
        material = BianisotropicMaterial.from_gyration(eps_l, gamma)
        eps_t, mu, xi, zeta = (_np(t)[0] for t in material.tensors(0.6))
        assert_allclose(mu, np.eye(3), rtol=0, atol=0)
        alpha = np.trace(gamma) / 2 * np.eye(3) - gamma.T
        assert_allclose(eps_t, eps_l + alpha @ alpha.T, rtol=0, atol=1e-15)
        assert_allclose(xi, 1j * alpha, rtol=0, atol=1e-18)
        assert_allclose(zeta, -1j * alpha.T, rtol=0, atol=1e-18)
        k = rng.normal(size=3) * 1.5
        big_k = _cross_matrix(k)
        tellegen = big_k @ (big_k - zeta) + eps_t + xi @ (big_k - zeta)
        landau = big_k @ big_k + eps_l + 1j * _cross_matrix(gamma @ k)
        assert np.max(np.abs(tellegen - landau)) < 1e-14


def test_quartz_on_axis_indices(set_test_backend):
    """Along the optic axis n_{L,R} = √(n_o² + κ²) ± κ exactly (the wave
    matrix is singular there)."""
    ordinary, extraordinary = _quartz()
    kappa = float(kappa_from_rotatory_power(21.7283, 0.5894050191))
    material = BianisotropicMaterial.from_optical_activity(
        UniaxialMaterial(ordinary, extraordinary), quartz_alpha(kappa)
    )
    eps, mu, xi, zeta = (_np(t)[0] for t in material.tensors(0.5894050191))
    n_o = float(_np(ordinary.n(0.5894050191)).ravel()[0])
    for sign in (1.0, -1.0):
        n = math.sqrt(n_o**2 + kappa**2) + sign * kappa
        big_k = _cross_matrix([0.0, 0.0, n])
        wave = big_k @ (big_k - zeta) + eps + xi @ (big_k - zeta)
        singular = np.linalg.svd(wave, compute_uv=False)
        assert singular[-1] < 1e-14
        # A small change of n makes the matrix regular (a simple root).
        big_k = _cross_matrix([0.0, 0.0, n + 1e-6])
        wave = big_k @ (big_k - zeta) + eps + xi @ (big_k - zeta)
        assert np.linalg.svd(wave, compute_uv=False)[-1] > 1e-7


def test_pasteur_and_post_constraint(set_test_backend):
    """Pasteur ξ = iκI, ζ = -iκI obeys Post and reciprocity; a Tellegen medium
    ξ = ζ = χI violates both (tr(μ⁻¹(ξ + ζ)) = 6χ)."""
    glass = IdealMaterial(1.5)
    pasteur = BianisotropicMaterial.from_optical_activity(
        glass, 1e-3 * np.eye(3), epsilon_form="tellegen"
    )
    assert pasteur.is_isotropic(0.6)
    assert pasteur.is_transparent(0.6)
    assert pasteur.is_reciprocal(0.6)
    assert abs(complex(_np(pasteur.post_constraint(0.6))[0])) < 1e-18
    assert_allclose(_np(pasteur.epsilon(0.6))[0], 2.25 * np.eye(3), rtol=0, atol=1e-15)
    landau = BianisotropicMaterial.from_optical_activity(glass, 1e-3 * np.eye(3))
    assert_allclose(
        _np(landau.epsilon(0.6))[0], (2.25 + 1e-6) * np.eye(3), rtol=0, atol=1e-15
    )

    chi = 2e-3
    tellegen = BianisotropicMaterial(
        glass, xi=chi * np.eye(3), zeta=chi * np.eye(3), mu=2 * np.eye(3)
    )
    assert_allclose(
        _np(tellegen.post_constraint([0.5, 0.6])),
        [3 * chi, 3 * chi],
        rtol=0,
        atol=1e-18,
    )
    assert not tellegen.is_reciprocal(0.6)
    assert tellegen.is_transparent(0.6)
    with pytest.raises(ValueError, match="epsilon_form"):
        BianisotropicMaterial.from_optical_activity(
            glass, np.eye(3), epsilon_form="born"
        )


def test_bianisotropic_callable_alpha(set_test_backend):
    """A dispersive α (a callable of the wavelength) and a gyration callable."""
    ordinary, extraordinary = _quartz()

    def alpha(w):
        return quartz_alpha(
            kappa_from_rotatory_power(21.7283 * (0.5894050191 / w) ** 2, w)
        )

    material = BianisotropicMaterial.from_optical_activity(
        UniaxialMaterial(ordinary, extraordinary), alpha
    )
    xi = _np(material.xi(be.asarray([0.5894050191, 0.4])))
    assert abs(xi[0, 0, 0].imag - 3.5574359e-05) < 5e-13
    assert xi[1, 0, 0].imag > xi[0, 0, 0].imag
    from_gyration = BianisotropicMaterial.from_gyration(
        UniaxialMaterial(ordinary, extraordinary),
        lambda w: gyration_from_alpha(alpha(w)),
    )
    assert_allclose(_np(from_gyration.xi(0.4)), xi[1:], rtol=0, atol=1e-18)
    assert_allclose(
        _np(from_gyration.epsilon(0.4)), _np(material.epsilon(0.4)), rtol=0, atol=1e-15
    )


# --------------------------------------------------------------- serialization


def test_serialization_round_trip(set_test_backend):
    x, y, z = _ktp()
    ordinary, extraordinary = _quartz()
    media = [
        UniaxialMaterial(*_calcite(), optic_axis=(0.0, 1.0, 1.0)),
        BiaxialMaterial(x, y, z, rotation=euler_zxz_matrix(1.0, 2.0, 3.0)),
        TensorMaterial(np.diag([2.0, 2.1, 2.2 + 0.1j])),
        BianisotropicMaterial.from_optical_activity(
            UniaxialMaterial(ordinary, extraordinary), quartz_alpha(3.5e-5)
        ),
        BianisotropicMaterial(
            IdealMaterial(1.4), xi=1e-3j * np.eye(3), zeta=-1e-3j * np.eye(3)
        ),
    ]
    for material in media:
        data = material.to_dict()
        copy = BaseTensorMaterial.from_dict(data)
        assert type(copy) is type(material)
        assert copy == material
        assert_allclose(
            _np(copy.constitutive_6x6(0.6)),
            _np(material.constitutive_6x6(0.6)),
            rtol=0,
            atol=0,
        )
