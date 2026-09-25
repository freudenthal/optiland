"""Closed-form tests of the polarization conventions of the sequential ray trace.

Each test checks one defect of the issue "Polarization ray tracing: p-reflection
sign, complex index, power normalization and related defects" (D-1 to D-9) and D-10
against a closed form. References: Born and Wolf, Principles of Optics, 7th ed.,
sections 1.5.2, 1.6.4 and 14.2 (Fresnel and Airy formulas, absorbing media);
Macleod, Thin-Film Optical Filters, 5th ed., chapter 2 (the thin-film p
convention); Yun, Crabtree and Chipman, Appl. Opt. 50, 2855 (2011) (the PRT
matrix, P k_in = k_out).

Conventions: time dependence exp(-iwt), complex index n + ik with k >= 0.
"""

from __future__ import annotations

import numpy as np
import pytest

import optiland.backend as be
from optiland import optic
from optiland.analysis.jones_pupil import JonesPupil
from optiland.coatings import FresnelCoating, JonesThinFilm, ThinFilmCoating
from optiland.jones import JonesFresnel
from optiland.materials import IdealMaterial
from optiland.rays import PolarizationState, PolarizedRays, create_polarization
from optiland.thin_film import ThinFilmStack

from .utils import assert_allclose

WL = 0.633  # µm


def _ray(theta_deg: float) -> PolarizedRays:
    """One ray at the origin, direction (0, sin, cos) in the y-z plane."""
    t = np.deg2rad(theta_deg)

    def a(v):
        return be.array([float(v)])

    return PolarizedRays(
        a(0), a(0), a(0), a(0), a(np.sin(t)), a(np.cos(t)), a(1), a(WL)
    )


def _reflect(rays: PolarizedRays, coating) -> PolarizedRays:
    """Reflect on the plane z = 0 and apply the coating: the two calls of the
    refractive-reflective interaction model."""
    rays.reflect(0.0, 0.0, 1.0)
    zero = be.zeros_like(rays.x)
    coating.interact(rays, reflect=True, nx=zero, ny=zero, nz=be.ones_like(rays.x))
    return rays


def _field_out(rays: PolarizedRays, state: str) -> be.ndarray:
    E0 = rays._get_3d_electric_field(create_polarization(state))
    return rays.get_output_field(E0)[0], E0[0]


def _fresnel(N: complex, theta: float):
    """Fresnel r_s, r_pp, t_s, t_pp from a medium of index 1 into index N."""
    c = np.cos(theta)
    root = np.sqrt(N**2 - np.sin(theta) ** 2 + 0j)  # N cos(theta_t), Im >= 0
    rs = (c - root) / (c + root)
    rp = (N**2 * c - root) / (N**2 * c + root)
    ts = 2 * c / (c + root)
    tp = 2 * N * c / (N**2 * c + root)
    return rs, rp, ts, tp, root


class TestReflectionSign:
    """D-1: the p entry of a Fresnel reflection."""

    @pytest.mark.parametrize("state", ["H", "V", "L+45", "RCP"])
    def test_normal_incidence_lab_field(self, set_test_backend, state):
        """At normal incidence the reflected lab field is r E for every
        transverse component, r = (1 - n) / (1 + n) = -0.2."""
        rays = _reflect(
            _ray(0.0), FresnelCoating(IdealMaterial(1.0), IdealMaterial(1.5))
        )
        E1, E0 = _field_out(rays, state)
        assert_allclose(E1, -0.2 * E0, rtol=0, atol=1e-14)

    @pytest.mark.parametrize("theta_deg", [30.0, 60.0, 85.0])
    def test_oblique_p_entry(self, set_test_backend, theta_deg):
        """In the local (s, p, k) frame with p = k x s, the p entry is +r_pp."""
        theta = np.deg2rad(theta_deg)
        rays = _ray(theta_deg)
        rays.reflect(0.0, 0.0, 1.0)
        jones = JonesFresnel(IdealMaterial(1.0), IdealMaterial(1.5))
        J = jones.calculate_matrix(rays, reflect=True, aoi=be.array([theta]))
        rs, rp, _, _, _ = _fresnel(1.5, theta)
        assert_allclose(J[0, 0, 0], rs, rtol=0, atol=1e-14)
        assert_allclose(J[0, 1, 1], rp, rtol=0, atol=1e-14)

    def test_circular_handedness_changes(self, set_test_backend):
        """A circular state changes its handedness on one reflection at
        normal incidence: S3 = Im((E x E*) . k) / |E|^2 changes sign."""
        rays = _reflect(
            _ray(0.0), FresnelCoating(IdealMaterial(1.0), IdealMaterial(1.5))
        )
        E1, E0 = _field_out(rays, "RCP")
        E0 = be.to_numpy(E0)
        E1 = be.to_numpy(E1)
        s3_in = (
            np.imag(np.cross(E0, np.conj(E0)) @ [0.0, 0.0, 1.0]) / np.vdot(E0, E0).real
        )
        s3_out = (
            np.imag(np.cross(E1, np.conj(E1)) @ [0.0, 0.0, -1.0]) / np.vdot(E1, E1).real
        )
        assert_allclose(s3_out, -s3_in, rtol=0, atol=1e-12)
        assert abs(s3_in) > 0.99


class TestComplexIndex:
    """D-5: the extinction coefficient of the exit material."""

    def test_metal_reflectance(self, set_test_backend):
        """Aluminium at 633 nm, N = 1.4495 + 7.5387i: R = |(1 - N)/(1 + N)|^2."""
        N = 1.4495 + 7.5387j
        rays = _reflect(
            _ray(0.0), FresnelCoating(IdealMaterial(1.0), IdealMaterial(N.real, N.imag))
        )
        E1, E0 = _field_out(rays, "H")
        r = (1 - N) / (1 + N)
        assert_allclose(E1, r * E0, rtol=0, atol=1e-14)
        assert abs(abs(r) ** 2 - 0.9077) < 1e-4

    @pytest.mark.parametrize("reflect", [True, False])
    def test_oblique_absorbing(self, set_test_backend, reflect):
        theta = 0.2
        N = 1.5 + 0.1j
        rays = _ray(np.rad2deg(theta))
        jones = JonesFresnel(IdealMaterial(1.0), IdealMaterial(N.real, N.imag))
        J = jones.calculate_matrix(rays, reflect=reflect, aoi=be.array([theta]))
        rs, rp, ts, tp, _ = _fresnel(N, theta)
        expected = (rs, rp) if reflect else (ts, tp)
        assert_allclose(J[0, 0, 0], expected[0], rtol=0, atol=1e-14)
        assert_allclose(J[0, 1, 1], expected[1], rtol=0, atol=1e-14)


class TestReflectionPRT:
    """D-6: the PRT matrix of a reflection maps k_in to k_out."""

    @pytest.mark.parametrize(
        "coating",
        [
            FresnelCoating(IdealMaterial(1.0), IdealMaterial(1.5)),
            ThinFilmCoating(IdealMaterial(1.0), IdealMaterial(1.5)),
        ],
        ids=["fresnel", "thin_film"],
    )
    def test_k_column(self, set_test_backend, coating):
        t = np.deg2rad(30.0)
        rays = _reflect(_ray(30.0), coating)
        k_in = be.array([0.0, np.sin(t), np.cos(t)])
        k_out = be.array([0.0, np.sin(t), -np.cos(t)])
        assert_allclose(
            be.matmul(rays.p[0], be.to_complex(k_in)), k_out, rtol=0, atol=1e-14
        )


class TestThinFilmTimeConvention:
    """D-2: thin-film r in exp(-iwt), as t."""

    @pytest.mark.parametrize("pol", ["s", "p"])
    def test_single_absorbing_layer(self, set_test_backend, pol):
        """Airy formula for one layer (Born and Wolf 1.6.4), exp(-iwt):
        r = (r01 + r12 e^{2i b}) / (1 + r01 r12 e^{2i b}), b = 2 pi d N1 cos1 / wl.
        thin_film states r_p with the Macleod sign: r_p = -r_pp."""
        theta = np.deg2rad(45.0)
        n0, n1, n2, d = 1.0, 0.2 + 3.0j, 1.5 + 0.0j, 0.030
        stack = ThinFilmStack(IdealMaterial(n0), IdealMaterial(n2.real))
        stack.add_layer_nm(IdealMaterial(n1.real, n1.imag), d * 1e3)
        out = stack.compute_rtRTA_elementwise(be.array([WL]), be.array([theta]), pol)

        sin2 = (n0 * np.sin(theta)) ** 2
        nc = [np.sqrt(n**2 - sin2 + 0j) for n in (n0, n1, n2)]  # N cos
        if pol == "s":
            r01 = (nc[0] - nc[1]) / (nc[0] + nc[1])
            r12 = (nc[1] - nc[2]) / (nc[1] + nc[2])
        else:
            n = (n0, n1, n2)
            r01 = (n[1] ** 2 * nc[0] - n[0] ** 2 * nc[1]) / (
                n[1] ** 2 * nc[0] + n[0] ** 2 * nc[1]
            )
            r12 = (n[2] ** 2 * nc[1] - n[1] ** 2 * nc[2]) / (
                n[2] ** 2 * nc[1] + n[1] ** 2 * nc[2]
            )
        phase = np.exp(2j * 2 * np.pi * d * nc[1] / WL)
        r = (r01 + r12 * phase) / (1 + r01 * r12 * phase)
        expected = r if pol == "s" else -r
        assert_allclose(out["r"][0], expected, rtol=0, atol=1e-12)
        assert_allclose(out["R"][0], abs(r) ** 2, rtol=0, atol=1e-12)

    @pytest.mark.parametrize("reflect", [True, False])
    def test_jones_keeps_complex_coefficients(self, set_test_backend, reflect):
        """The JonesThinFilm matrix of an absorbing stack keeps the imaginary
        parts of r and t on every backend (D-10)."""
        theta = np.deg2rad(45.0)
        stack = ThinFilmStack(IdealMaterial(1.0), IdealMaterial(1.5))
        stack.add_layer_nm(IdealMaterial(0.2, 3.0), 30.0)
        rays = _ray(45.0)
        J = JonesThinFilm(stack).calculate_matrix(
            rays, reflect=reflect, aoi=be.array([theta])
        )
        key = "r" if reflect else "t"
        s = stack.compute_rtRTA_elementwise(be.array([WL]), be.array([theta]), "s")[key]
        assert_allclose(J[0, 0, 0], s[0], rtol=0, atol=1e-14)
        assert abs(float(be.to_numpy(be.imag(J[0, 0, 0])))) > 0.1


class TestThinFilmFieldAmplitude:
    """D-4: JonesThinFilm uses the full-field t_p."""

    @pytest.mark.parametrize("theta_deg", [0.0, 45.0, 70.0])
    def test_zero_layers_equals_fresnel(self, set_test_backend, theta_deg):
        theta = np.deg2rad(theta_deg)
        rays = _ray(theta_deg)
        stack = ThinFilmStack(IdealMaterial(1.0), IdealMaterial(1.5))
        J = JonesThinFilm(stack).calculate_matrix(
            rays, reflect=False, aoi=be.array([theta])
        )
        _, _, ts, tp, _ = _fresnel(1.5, theta)
        assert_allclose(J[0, 0, 0], ts, rtol=0, atol=1e-14)
        assert_allclose(J[0, 1, 1], tp, rtol=0, atol=1e-14)
        if theta_deg == 45.0:
            assert abs(tp - 0.7280) < 1e-4


class TestPower:
    """D-3: the ray intensity is the power transmittance."""

    @pytest.mark.parametrize("jones_type", ["fresnel", "thin_film"])
    @pytest.mark.parametrize("theta_deg", [0.0, 30.0, 60.0, 85.0])
    def test_energy_balance(self, set_test_backend, jones_type, theta_deg):
        """|r|^2 + |t|^2 * flux_factor = 1 for s and p at a lossless interface."""
        theta = be.array([np.deg2rad(theta_deg)])
        rays = _ray(theta_deg)
        pre, post = IdealMaterial(1.0), IdealMaterial(1.5)
        if jones_type == "fresnel":
            jones = JonesFresnel(pre, post)
        else:
            jones = JonesThinFilm(ThinFilmStack(pre, post))
        Jr = jones.calculate_matrix(rays, reflect=True, aoi=theta)
        Jt = jones.calculate_matrix(rays, reflect=False, aoi=theta)
        flux = jones.calculate_flux_factor(rays, reflect=False, aoi=theta)
        assert_allclose(jones.calculate_flux_factor(rays, reflect=True, aoi=theta), 1.0)
        for j in (0, 1):
            total = be.abs(Jr[0, j, j]) ** 2 + be.abs(Jt[0, j, j]) ** 2 * flux[0]
            assert_allclose(total, 1.0, rtol=0, atol=1e-14)

    @staticmethod
    def _single_surface(theta_deg, state):
        lens = optic.Optic()
        lens.surfaces.add(index=0, radius=np.inf, thickness=np.inf)
        lens.surfaces.add(
            index=1,
            radius=np.inf,
            thickness=1.0,
            material=IdealMaterial(1.5),
            is_stop=True,
            coating="fresnel",
        )
        lens.surfaces.add(index=2, material=IdealMaterial(1.5))
        lens.set_aperture(aperture_type="EPD", value=2.0)
        lens.fields.set_type(field_type="angle")
        lens.fields.add(y=0.0)
        lens.fields.add(y=theta_deg)
        lens.wavelengths.add(value=WL, is_primary=True)
        lens.updater.set_polarization(state)
        return lens

    def test_single_surface_trace(self, set_test_backend):
        """s-polarized ("H" in the y-z plane of incidence) at 30 deg into
        n = 1.5: T_s = |t_s|^2 n cos(theta_t) / cos(theta_i) = 0.9422."""
        theta = np.deg2rad(30.0)
        lens = self._single_surface(30.0, create_polarization("H"))
        rays = lens.trace(Hx=0, Hy=1, wavelength=WL, num_rays=1, distribution="line_y")
        rs, _, ts, _, root = _fresnel(1.5, theta)
        T_s = abs(ts) ** 2 * root.real / np.cos(theta)
        assert_allclose(rays.i, T_s, rtol=0, atol=1e-12)
        assert_allclose(T_s, 1 - abs(rs) ** 2, rtol=0, atol=1e-14)
        assert abs(T_s - 0.9422) < 1e-4

    def test_single_surface_unpolarized(self, set_test_backend):
        theta = np.deg2rad(60.0)
        lens = self._single_surface(60.0, PolarizationState(is_polarized=False))
        rays = lens.trace(Hx=0, Hy=1, wavelength=WL, num_rays=1, distribution="line_y")
        rs, rp, _, _, _ = _fresnel(1.5, theta)
        assert_allclose(
            rays.i, 1 - (abs(rs) ** 2 + abs(rp) ** 2) / 2, rtol=0, atol=1e-12
        )


class TestBulkAbsorption:
    """D-9: a polarized trace keeps the running intensity."""

    @staticmethod
    def _block(polarized):
        lens = optic.Optic()
        lens.surfaces.add(index=0, radius=np.inf, thickness=np.inf)
        lens.surfaces.add(
            index=1,
            radius=np.inf,
            thickness=100.0,
            material=IdealMaterial(1.5, 1e-6),
            is_stop=True,
        )
        lens.surfaces.add(index=2, radius=np.inf, thickness=10.0)
        lens.surfaces.add(index=3)
        lens.set_aperture(aperture_type="EPD", value=2.0)
        lens.fields.set_type(field_type="angle")
        lens.fields.add(y=0.0)
        lens.wavelengths.add(value=WL, is_primary=True)
        if polarized:
            lens.updater.set_polarization(create_polarization("H"))
        return lens

    def test_uncoated_block(self, set_test_backend):
        """No coating: every PRT is a rotation, so the polarized intensity is
        the scalar intensity, exp(-4 pi k d / wl) for 100 mm, k = 1E-6."""
        kw = {
            "Hx": 0.0,
            "Hy": 0.0,
            "wavelength": WL,
            "num_rays": 3,
            "distribution": "line_y",
        }
        plain = self._block(False).trace(**kw).i
        polar = self._block(True).trace(**kw).i
        expected = np.exp(-4 * np.pi * 1e-6 * 100.0e3 / WL)
        assert_allclose(plain, expected, rtol=1e-10, atol=0)
        assert_allclose(polar, plain, rtol=0, atol=1e-14)


class TestJonesPupilBasis:
    """D-7: the Jones pupil input basis is transverse to the input ray."""

    def test_tilted_field_identity(self, set_test_backend):
        """A 10 deg field through flat uncoated surfaces in air: P = I, and
        with a transverse input basis the pupil Jones matrix is the identity
        (with the global x, y basis J_yy would be cos 10 deg)."""
        lens = optic.Optic()
        lens.surfaces.add(index=0, radius=np.inf, thickness=np.inf)
        lens.surfaces.add(index=1, radius=np.inf, thickness=5.0, is_stop=True)
        lens.surfaces.add(index=2)
        lens.set_aperture(aperture_type="EPD", value=2.0)
        lens.fields.set_type(field_type="angle")
        lens.fields.add(y=0.0)
        lens.fields.add(y=10.0)
        lens.wavelengths.add(value=WL, is_primary=True)
        lens.updater.set_polarization(PolarizationState())
        J = JonesPupil(lens, field=(0, 1), grid_size=3).data[0]["J"]
        expected = be.to_numpy(be.tile(be.eye(2), (9, 1, 1)))
        assert_allclose(J, expected, rtol=0, atol=1e-12)


class TestMirrorCoating:
    """D-8: a mirror surface keeps the coating that the user gives."""

    def test_mirror_keeps_fresnel_coating(self, set_test_backend):
        theta = np.deg2rad(30.0)
        coating = FresnelCoating(IdealMaterial(1.0), IdealMaterial(1.5))
        lens = optic.Optic()
        lens.surfaces.add(index=0, radius=np.inf, thickness=np.inf)
        lens.surfaces.add(
            index=1,
            radius=np.inf,
            thickness=-1.0,
            material="mirror",
            is_stop=True,
            coating=coating,
        )
        lens.surfaces.add(index=2)
        lens.set_aperture(aperture_type="EPD", value=2.0)
        lens.fields.set_type(field_type="angle")
        lens.fields.add(y=0.0)
        lens.fields.add(y=30.0)
        lens.wavelengths.add(value=WL, is_primary=True)
        lens.updater.set_polarization(create_polarization("H"))

        mirror_coating = lens.surfaces[1].interaction_model.coating
        assert mirror_coating.material_post.index == 1.5

        rays = lens.trace(Hx=0, Hy=1, wavelength=WL, num_rays=1, distribution="line_y")
        rs, _, _, _, _ = _fresnel(1.5, theta)
        assert_allclose(rays.i, abs(rs) ** 2, rtol=0, atol=1e-12)

    def test_fresnel_string_follows_surface(self, set_test_backend):
        """coating="fresnel" on a refracting surface follows the surface
        materials, as before."""
        lens = optic.Optic()
        lens.surfaces.add(index=0, radius=np.inf, thickness=np.inf)
        lens.surfaces.add(
            index=1,
            radius=np.inf,
            thickness=1.0,
            material=IdealMaterial(1.5),
            coating="fresnel",
        )
        lens.surfaces.add(index=2)
        lens.surfaces[1].material_post = IdealMaterial(1.7)
        coating = lens.surfaces[1].interaction_model.coating
        assert coating.follows_surface
        assert coating.material_post.index == 1.7


class TestRayAimingWithCoatings:
    """The ray aimers trace real rays through polarized coatings."""

    @pytest.mark.parametrize("mode", ["iterative", "robust"])
    def test_aimed_trace_with_fresnel_coatings(self, set_test_backend, mode):
        lens = optic.Optic()
        lens.surfaces.add(index=0, radius=np.inf, thickness=np.inf)
        lens.surfaces.add(
            index=1,
            radius=np.inf,
            thickness=2.0,
            material=IdealMaterial(1.5),
            is_stop=True,
            coating="fresnel",
        )
        lens.surfaces.add(index=2, thickness=5.0, coating="fresnel")
        lens.surfaces.add(index=3)
        lens.set_aperture(aperture_type="EPD", value=2.0)
        lens.fields.set_type(field_type="angle")
        lens.fields.add(y=0.0)
        lens.fields.add(y=10.0)
        lens.wavelengths.add(value=WL, is_primary=True)
        lens.updater.set_polarization(create_polarization("H"))
        lens.ray_tracer.set_aiming(mode)

        rays = lens.trace(Hx=0, Hy=0, wavelength=WL, num_rays=1, distribution="line_y")

        # Normal incidence on two faces of n = 1.5: T = (1 - 0.04)**2.
        assert_allclose(rays.i, (1 - 0.04) ** 2, rtol=0, atol=1e-12)
