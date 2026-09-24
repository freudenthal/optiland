"""Tests of the Stokes and Mueller outputs of polarized rays.

Expected values are closed forms: the Mueller matrices of an ideal polarizer, a
quarter-wave retarder and a mirror at normal incidence (Chipman, Lam and Young,
Polarized Light and Optical Systems, 2018, ch. 6), the Fresnel equations (Born
and Wolf, Principles of Optics, 7th ed., 1.5.2), and the retardance of total
internal reflection (Born and Wolf 1.5.4).
"""

from __future__ import annotations

import numpy as np
import pytest

import optiland.backend as be
from optiland import optic
from optiland.analysis import mueller
from optiland.coatings import FresnelCoating, PolarizerCoating, RetarderCoating
from optiland.jones import JonesFresnel
from optiland.materials import IdealMaterial
from optiland.rays import PolarizationState, PolarizedRays, create_polarization

from .utils import assert_allclose

WL = 0.633  # µm


def _c(z):
    """Complex backend array from a complex array-like."""
    z = np.asarray(z, dtype=complex)
    return be.to_complex(be.array(z.real)) + 1j * be.to_complex(be.array(z.imag))


QWP_MUELLER = np.array(
    [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1], [0, 0, -1, 0]], dtype=float
)


def _flat(coating, *, mirror=False, field_deg=0.0, state=None):
    """One flat surface with ``coating`` at the stop; air everywhere."""
    lens = optic.Optic()
    lens.surfaces.add(index=0, radius=np.inf, thickness=np.inf)
    if mirror:
        lens.surfaces.add(
            index=1,
            radius=np.inf,
            thickness=-1.0,
            material="mirror",
            is_stop=True,
            coating=coating,
        )
    else:
        lens.surfaces.add(
            index=1, radius=np.inf, thickness=1.0, is_stop=True, coating=coating
        )
    lens.surfaces.add(index=2)
    lens.set_aperture(aperture_type="EPD", value=2.0)
    lens.fields.set_type(field_type="angle")
    lens.fields.add(y=0.0)
    if field_deg:
        lens.fields.add(y=field_deg)
    lens.wavelengths.add(value=WL, is_primary=True)
    lens.updater.set_polarization(state or PolarizationState())
    return lens


def _chief(lens, field=0.0):
    return lens.trace_generic(Hx=0.0, Hy=field, Px=0.0, Py=0.0, wavelength=WL)


def _singlet(state):
    lens = optic.Optic()
    lens.surfaces.add(index=0, radius=np.inf, thickness=np.inf)
    lens.surfaces.add(
        index=1,
        radius=50.0,
        thickness=5.0,
        material=IdealMaterial(1.6),
        is_stop=True,
        coating="fresnel",
    )
    lens.surfaces.add(index=2, radius=-60.0, thickness=90.0, coating="fresnel")
    lens.surfaces.add(index=3)
    lens.set_aperture(aperture_type="EPD", value=20.0)
    lens.fields.set_type(field_type="angle")
    lens.fields.add(y=0.0)
    lens.fields.add(y=10.0)
    lens.wavelengths.add(value=WL, is_primary=True)
    lens.updater.set_polarization(state)
    return lens


def _k_in(rays):
    return be.stack([rays._L0, rays._M0, rays._N0], axis=1)


def _k_out(rays):
    return be.stack([rays.L, rays.M, rays.N], axis=1)


class TestJonesToMueller:
    def test_identity(self, set_test_backend):
        M = mueller.jones_to_mueller(_c([[[1.0, 0.0], [0.0, 1.0]]]))
        assert_allclose(M[0], np.eye(4), rtol=0, atol=1e-15)

    def test_polarizer_x(self, set_test_backend):
        M = mueller.jones_to_mueller(_c([[[1.0, 0.0], [0.0, 0.0]]]))
        expected = 0.5 * np.array(
            [[1, 1, 0, 0], [1, 1, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]
        )
        assert_allclose(M[0], expected, rtol=0, atol=1e-15)

    def test_quarter_wave_y_slow(self, set_test_backend):
        """J = diag(1, i): +45 deg linear becomes left circular (S3 = -1)."""
        M = mueller.jones_to_mueller(_c([[[1.0, 0.0], [0.0, 1.0j]]]))
        assert_allclose(M[0], QWP_MUELLER, rtol=0, atol=1e-15)

    def test_stokes_of_output(self, set_test_backend):
        """S(J E) = M(J) S(E) for random J and E."""
        rng = np.random.default_rng(3)
        J = rng.normal(size=(5, 2, 2)) + 1j * rng.normal(size=(5, 2, 2))
        E = rng.normal(size=(5, 2)) + 1j * rng.normal(size=(5, 2))
        M = mueller.jones_to_mueller(_c(J))
        s_in = mueller.stokes_from_jones_vector(_c(E))
        s_out = mueller.stokes_from_jones_vector(_c(np.einsum("nij,nj->ni", J, E)))
        assert_allclose(be.sum(M * s_in[:, None, :], axis=2), s_out, rtol=0, atol=1e-12)

    def test_s3_sign_of_rcp(self, set_test_backend):
        """The "RCP" input state has S3 = +1 (right circular, looking toward
        the source), equal to Im((E x E*) . k) of its 3D field."""
        s = mueller.stokes_from_state(create_polarization("RCP"))
        assert_allclose(s, [1.0, 0.0, 0.0, 1.0], rtol=0, atol=1e-15)


class TestTracedElements:
    def test_polarizer(self, set_test_backend):
        lens = _flat(PolarizerCoating(axis=(1.0, 0.0, 0.0)))
        M = mueller.mueller_matrix(_chief(lens))
        expected = 0.5 * np.array(
            [[1, 1, 0, 0], [1, 1, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]
        )
        assert_allclose(M[0], expected, rtol=0, atol=1e-14)
        rays = _chief(lens)
        d = mueller.diattenuation(rays.p, _k_in(rays), _k_out(rays))
        assert_allclose(d, 1.0, rtol=0, atol=1e-12)

    def test_quarter_wave_retarder(self, set_test_backend):
        lens = _flat(RetarderCoating(np.pi / 2, axis=(1.0, 0.0, 0.0)))
        rays = _chief(lens)
        assert_allclose(
            mueller.mueller_matrix(rays)[0], QWP_MUELLER, rtol=0, atol=1e-14
        )
        assert_allclose(
            mueller.retardance(rays.p, rays.q, _k_in(rays)),
            np.pi / 2,
            rtol=0,
            atol=1e-12,
        )
        assert_allclose(
            mueller.diattenuation(rays.p, _k_in(rays), _k_out(rays)),
            0.0,
            rtol=0,
            atol=1e-7,
        )

    def test_mirror_normal_incidence(self, set_test_backend):
        """A mirror of n = 1.5 at normal incidence: M = R diag(1, 1, -1, -1)
        with R = 0.04 in the detector frame x_det = x, y_det = k_out x x = -y."""
        lens = _flat(
            FresnelCoating(IdealMaterial(1.0), IdealMaterial(1.5)), mirror=True
        )
        M = mueller.mueller_matrix(_chief(lens))
        assert_allclose(
            M[0], 0.04 * np.diag([1.0, 1.0, -1.0, -1.0]), rtol=0, atol=1e-14
        )

    def test_stokes_at_detector_equals_intensity(self, set_test_backend):
        state = create_polarization("L+45")
        lens = _flat(RetarderCoating(np.pi / 2, axis=(1.0, 0.0, 0.0)), state=state)
        rays = lens.trace(Hx=0, Hy=0, wavelength=WL, num_rays=3, distribution="line_y")
        s = mueller.stokes_at_detector(rays, state)
        assert_allclose(s[:, 0], rays.i, rtol=0, atol=1e-14)
        assert_allclose(s[:, 3], -rays.i, rtol=0, atol=1e-14)  # left circular


class TestInterface:
    @pytest.mark.parametrize("theta_deg", [30.0, 60.0])
    def test_transmission_diattenuation(self, set_test_backend, theta_deg):
        """D = ||t_s|^2 - |t_p|^2| / (|t_s|^2 + |t_p|^2); retardance 0. The uncoated
        image surface refracts back to air and adds only a rotation."""
        theta = np.deg2rad(theta_deg)
        lens = _flat(None, field_deg=theta_deg)
        lens.surfaces[1].material_post = IdealMaterial(1.5)
        lens.surfaces[1].set_fresnel_coating()
        rays = _chief(lens, 1.0)
        c = np.cos(theta)
        root = np.sqrt(1.5**2 - np.sin(theta) ** 2)
        ts = 2 * c / (c + root)
        tp = 2 * 1.5 * c / (1.5**2 * c + root)
        expected = abs(ts**2 - tp**2) / (ts**2 + tp**2)  # D >= 0
        d = mueller.diattenuation(rays.p, _k_in(rays), _k_out(rays))
        assert_allclose(d, expected, rtol=0, atol=1e-12)
        M = mueller.mueller_matrix(rays)
        assert_allclose(
            mueller.diattenuation_from_mueller(M), expected, rtol=0, atol=1e-12
        )
        assert_allclose(
            mueller.retardance(rays.p, rays.q, _k_in(rays)), 0.0, rtol=0, atol=1e-12
        )
        # M00 is the unpolarized transmittance, 1 - (R_s + R_p) / 2
        rs = (c - root) / (c + root)
        rp = (1.5**2 * c - root) / (1.5**2 * c + root)
        assert_allclose(M[0, 0, 0], 1 - (rs**2 + rp**2) / 2, rtol=0, atol=1e-12)

    def test_total_internal_reflection_retardance(self, set_test_backend):
        """Glass (1.5) to air at 60 deg: tan(Δ/2) = cos θ sqrt(sin²θ - n²) / sin²θ
        with n = 1 / 1.5 (Born and Wolf 1.5.4); no diattenuation."""
        theta = np.deg2rad(60.0)
        one = be.ones(1)
        rays = PolarizedRays(
            0 * one,
            0 * one,
            0 * one,
            0 * one,
            np.sin(theta) * one,
            np.cos(theta) * one,
            one,
            WL * one,
        )
        rays.reflect(0.0, 0.0, 1.0)
        coating = FresnelCoating(IdealMaterial(1.5), IdealMaterial(1.0))
        coating.interact(rays, reflect=True, nx=0 * one, ny=0 * one, nz=one)
        n = 1.0 / 1.5
        s2 = np.sin(theta) ** 2
        expected = 2 * np.arctan(np.cos(theta) * np.sqrt(s2 - n**2) / s2)
        assert_allclose(
            mueller.retardance(rays.p, rays.q, _k_in(rays)),
            expected,
            rtol=0,
            atol=1e-12,
        )
        assert_allclose(
            mueller.diattenuation(rays.p, _k_in(rays), _k_out(rays)),
            0.0,
            rtol=0,
            atol=1e-7,
        )

    def test_jones_fresnel_absorbing_diattenuation(self, set_test_backend):
        """Metal reflection at 45 deg: D = (R_s - R_p) / (R_s + R_p)."""
        theta = np.deg2rad(45.0)
        N = 1.4495 + 7.5387j
        one = be.ones(1)
        rays = PolarizedRays(
            0 * one,
            0 * one,
            0 * one,
            0 * one,
            np.sin(theta) * one,
            np.cos(theta) * one,
            one,
            WL * one,
        )
        rays.reflect(0.0, 0.0, 1.0)
        J = JonesFresnel(IdealMaterial(1.0), IdealMaterial(N.real, N.imag))
        rays.update(J.calculate_matrix(rays, reflect=True, aoi=be.array([theta])))
        c = np.cos(theta)
        root = np.sqrt(N**2 - np.sin(theta) ** 2)
        Rs = abs((c - root) / (c + root)) ** 2
        Rp = abs((N**2 * c - root) / (N**2 * c + root)) ** 2
        d = mueller.diattenuation(rays.p, _k_in(rays), _k_out(rays))
        assert_allclose(d, (Rs - Rp) / (Rs + Rp), rtol=0, atol=1e-12)


class TestSystem:
    def test_singlet_non_depolarizing(self, set_test_backend):
        """A coated singlet: every ray Mueller matrix is non-depolarizing, and
        M00 equals the unpolarized trace intensity."""
        lens = _singlet(PolarizationState(is_polarized=False))
        rays = lens.trace(
            Hx=0, Hy=1, wavelength=WL, num_rays=5, distribution="hexapolar"
        )
        M = mueller.mueller_matrix(rays)
        assert_allclose(mueller.depolarization_index(M), 1.0, rtol=0, atol=1e-12)
        assert_allclose(M[:, 0, 0], rays.i, rtol=0, atol=1e-12)
        d_prt = mueller.diattenuation(rays.p, _k_in(rays), _k_out(rays))
        assert_allclose(
            mueller.diattenuation_from_mueller(M), d_prt, rtol=0, atol=1e-12
        )
        assert be.all(d_prt > 0.0)

    def test_uncoated_path_q_equals_p(self, set_test_backend):
        lens = _flat(None, field_deg=20.0)
        lens.surfaces[1].material_post = IdealMaterial(1.5)
        rays = _chief(lens, 1.0)
        assert_allclose(rays.p, rays.q, rtol=0, atol=1e-15)

    def test_detector_frame_rejects_parallel_reference(self, set_test_backend):
        with pytest.raises(ValueError, match="parallel"):
            mueller.detector_frame(be.array([[1.0, 0.0, 0.0]]))
