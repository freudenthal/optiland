"""The local period of a curved grating with ruled grooves.

The grooves of StandardGratingGeometry are the surface cut by the planes
h . r = j d, h = (-sin(alpha), cos(alpha), 0). Along the surface the groove
phase 2 pi (h . r) / d has the gradient (2 pi / d) h_t, h_t = h - (h . n) n,
so the local period is d / |h_t| and order m adds m (lambda / d) h_t to the
tangential part of n k (the vector grating equation with this local period).
"""

from __future__ import annotations

import numpy as np
import pytest

import optiland.backend as be
from optiland.materials import IdealMaterial
from optiland.optic import Optic
from optiland.rays import RealRays

from .utils import assert_allclose

RADIUS = 70.0
WAVELENGTH = 0.633  # um
PERIOD = 1.0  # um
POINTS = ((0.0, 5.0), (5.0, 0.0), (5.0, 5.0), (-8.0, 6.0))


def _grating(alpha, order=1):
    """One StandardGratingGeometry surface (R = 70 mm), air to n = 1.5."""
    optic = Optic()
    optic.surfaces.add(index=0, radius=be.inf, thickness=be.inf)
    optic.surfaces.add(
        index=1,
        radius=RADIUS,
        thickness=10.0,
        surface_type="grating",
        grating_order=order,
        grating_period=PERIOD,
        groove_orientation_angle=alpha,
        material=IdealMaterial(n=1.5),
        is_stop=True,
    )
    optic.surfaces.add(index=2)
    return optic.surfaces[1]


def _rays_at(geometry, x, y, direction=(0.0, 0.0, 1.0)):
    x, y = be.array([x]), be.array([y])
    d = np.asarray(direction, dtype=float)
    d = d / np.linalg.norm(d)
    return RealRays(
        x,
        y,
        geometry.sag(x, y),
        be.array([d[0]]),
        be.array([d[1]]),
        be.array([d[2]]),
        be.ones(1),
        be.array([WAVELENGTH]),
    )


def _sphere_normal(x, y):
    n = np.array([x, y, -np.sqrt(RADIUS**2 - x**2 - y**2)])
    return n / np.linalg.norm(n)


def _projected_h(x, y, alpha):
    h = np.array([-np.sin(alpha), np.cos(alpha), 0.0])
    n = _sphere_normal(x, y)
    return h - (h @ n) * n


@pytest.mark.parametrize("alpha", [0.0, 0.4])
def test_period_factor_is_projected_groove_normal(set_test_backend, alpha):
    surface = _grating(alpha)
    geometry = surface.geometry
    for x, y in POINTS:
        rays = _rays_at(geometry, x, y)
        fx, fy, fz = geometry.grating_vector(rays)
        factor = surface.interaction_model._period_factor(fx, fy)
        expected = np.linalg.norm(_projected_h(x, y, alpha))
        assert_allclose(factor, expected, 0, 1e-14)


def test_period_factor_unchanged_on_principal_sections(set_test_backend):
    """For alpha = 0 the normal tilts along h or across it where x = 0 or
    y = 0; there the old factor sqrt(fx**2 + fy**2) is exact."""
    surface = _grating(0.0)
    geometry = surface.geometry
    for x, y in POINTS[:2]:
        rays = _rays_at(geometry, x, y)
        fx, fy, fz = geometry.grating_vector(rays)
        factor = surface.interaction_model._period_factor(fx, fy)
        assert_allclose(factor, be.sqrt(fx**2 + fy**2), 0, 1e-14)


def test_period_factor_differs_off_principal_sections(set_test_backend):
    surface = _grating(0.0)
    geometry = surface.geometry
    differences = []
    for x, y in POINTS[2:]:
        rays = _rays_at(geometry, x, y)
        fx, fy, fz = geometry.grating_vector(rays)
        factor = surface.interaction_model._period_factor(fx, fy)
        differences.append(float(be.to_numpy(be.sqrt(fx**2 + fy**2) - factor)[0]))
    assert_allclose(differences, [1.311e-5, 4.851e-5], 0, 2e-8)


@pytest.mark.parametrize("alpha", [0.0, 0.4])
@pytest.mark.parametrize("order", [-1, 1])
def test_traced_ray_obeys_ruled_grating_equation(set_test_backend, alpha, order):
    surface = _grating(alpha, order)
    geometry = surface.geometry
    direction = np.array([0.05, -0.08, 1.0])
    direction = direction / np.linalg.norm(direction)
    for x, y in POINTS:
        rays = _rays_at(geometry, x, y, direction)
        rays = surface.interaction_model.interact_real_rays(rays)

        n = _sphere_normal(x, y)
        tangential = direction - (direction @ n) * n
        tangential = tangential + order * (WAVELENGTH / PERIOD) * _projected_h(
            x, y, alpha
        )
        tangential = tangential / 1.5
        normal = np.sqrt(1.0 - tangential @ tangential) * np.sign(direction @ n)
        expected = tangential + normal * n

        out = [float(be.to_numpy(c)[0]) for c in (rays.L, rays.M, rays.N)]
        assert_allclose(out, expected, 0, 1e-12)
