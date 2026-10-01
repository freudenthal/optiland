"""The paraxial grating models against the first-order form of the real trace.

A flat grating adds m lambda / d to the reduced angle: n2 sin(theta2) =
n1 sin(theta1) + m lambda / d. The paraxial form is n2 u2 = n1 u1 + m lambda / d,
and on reflection the slope changes sign as for a mirror. The real and the
paraxial slopes then differ only by the third-order terms of
tan(theta) - sin(theta) = theta**3 / 2 + ..., so
|u2(paraxial) - M/N(real)| <= |u1|**3 + |u2|**3.
"""

from __future__ import annotations

import pytest

import optiland.backend as be
from optiland.materials import IdealMaterial
from optiland.optic import Optic
from optiland.phase import LinearGratingPhaseProfile
from optiland.rays import ParaxialRays, RealRays

from .utils import assert_allclose

WAVELENGTH = 0.633  # um
PERIOD = 5.0  # um
U_IN = 0.01


def _one_surface(reflect, order, period=PERIOD, kind="grating"):
    """One flat grating surface (air to n = 1.5, or a mirror in air)."""
    optic = Optic()
    optic.surfaces.add(index=0, radius=be.inf, thickness=be.inf)
    kwargs = {
        "index": 1,
        "radius": be.inf,
        "thickness": -5.0 if reflect else 5.0,
        "is_stop": True,
        "material": "mirror" if reflect else IdealMaterial(n=1.5),
    }
    if kind == "grating":
        kwargs.update(
            surface_type="grating",
            grating_order=order,
            grating_period=period,
            groove_orientation_angle=0.0,
        )
    else:
        kwargs.update(
            surface_type="plane",
            interaction_type="phase",
            phase_profile=LinearGratingPhaseProfile(
                period=period / 1000.0, angle=be.pi / 2, order=order
            ),
        )
    optic.surfaces.add(**kwargs)
    optic.surfaces.add(index=2)
    optic.set_aperture(aperture_type="EPD", value=2.0)
    optic.fields.set_type(field_type="angle")
    optic.fields.add(y=0.0)
    optic.wavelengths.add(value=WAVELENGTH, is_primary=True)
    return optic


def _slopes(surface, u_in):
    """The paraxial slope and the real slope M/N after the surface."""
    paraxial = ParaxialRays(
        be.zeros(1), be.array([u_in]), be.zeros(1), be.array([WAVELENGTH])
    )
    u_out = surface.interaction_model.interact_paraxial_rays(paraxial).u

    theta = be.arctan(be.array([u_in]))
    real = RealRays(
        be.zeros(1),
        be.zeros(1),
        be.zeros(1),
        be.zeros(1),
        be.sin(theta),
        be.cos(theta),
        be.ones(1),
        be.array([WAVELENGTH]),
    )
    real = surface.interaction_model.interact_real_rays(real)
    return u_out, real.M / real.N


@pytest.mark.parametrize("kind", ["grating", "phase"])
@pytest.mark.parametrize("reflect", [False, True])
@pytest.mark.parametrize("order", [-1, 0, 1])
def test_paraxial_slope_is_first_order_real_slope(
    set_test_backend, kind, reflect, order
):
    surface = _one_surface(reflect, order, kind=kind).surfaces[1]
    u_out, real_slope = _slopes(surface, U_IN)

    n2 = 1.0 if reflect else 1.5
    closed_form = (U_IN + order * WAVELENGTH / PERIOD) / n2
    if reflect:
        closed_form = -closed_form
    assert_allclose(u_out, closed_form, 0, 1e-12)

    bound = abs(U_IN) ** 3 + abs(closed_form) ** 3
    assert be.to_numpy(be.abs(u_out - real_slope))[0] < bound


def test_reflective_grating_order_zero_is_a_mirror_in_glass(set_test_backend):
    """The mirror term of a reflective grating does not depend on the medium."""
    slopes = []
    for surface_type in ("standard", "grating"):
        optic = Optic()
        optic.surfaces.add(index=0, radius=be.inf, thickness=be.inf)
        optic.surfaces.add(index=1, radius=be.inf, thickness=5.0, material="N-BK7")
        kwargs = {}
        if surface_type == "grating":
            kwargs = {
                "grating_order": 0,
                "grating_period": PERIOD,
                "groove_orientation_angle": 0.0,
            }
        optic.surfaces.add(
            index=2,
            radius=-40.0,
            thickness=-5.0,
            material="mirror",
            surface_type=surface_type,
            is_stop=True,
            **kwargs,
        )
        optic.surfaces.add(index=3)
        model = optic.surfaces[2].interaction_model
        rays = ParaxialRays(
            be.array([1.5]), be.array([0.02]), be.zeros(1), be.array([WAVELENGTH])
        )
        slopes.append(model.interact_paraxial_rays(rays).u)
    assert_allclose(slopes[1], slopes[0], 0, 1e-15)


def _grating_before_stop(order, reflect):
    """A flat grating (d = 50 um) 20 mm before the stop, the image 30 mm after."""
    sign = -1.0 if reflect else 1.0
    optic = Optic()
    optic.surfaces.add(index=0, radius=be.inf, thickness=be.inf)
    optic.surfaces.add(
        index=1,
        radius=be.inf,
        thickness=sign * 20.0,
        surface_type="grating",
        grating_order=order,
        grating_period=50.0,
        groove_orientation_angle=0.0,
        material="mirror" if reflect else "air",
    )
    optic.surfaces.add(index=2, radius=be.inf, thickness=sign * 30.0, is_stop=True)
    optic.surfaces.add(index=3)
    optic.set_aperture(aperture_type="EPD", value=2.0)
    optic.fields.set_type(field_type="angle")
    optic.fields.add(y=0.0)
    optic.fields.add(y=1.0)
    optic.wavelengths.add(value=WAVELENGTH, is_primary=True)
    return optic


@pytest.mark.parametrize("reflect", [False, True])
@pytest.mark.parametrize("order", [-1, 0, 1])
def test_paraxial_trace_after_grating_follows_real_trace(
    set_test_backend, reflect, order
):
    """Paraxial ray heights and slopes at the image equal the real ones to
    third order (path 50 mm, slopes below 0.031)."""
    optic = _grating_before_stop(order, reflect)
    for hy, py in ((0.0, 1.0), (1.0, 0.0), (1.0, 1.0)):
        optic.paraxial.trace(Hy=hy, Py=py, wavelength=WAVELENGTH)
        y_par = optic.surfaces.y[-1]
        u_par = optic.surfaces.u[-1]
        real = optic.trace_generic(Hx=0.0, Hy=hy, Px=0.0, Py=py, wavelength=WAVELENGTH)
        u_cube = 0.031**3
        assert be.to_numpy(be.abs(u_par - real.M / real.N))[0] < 2 * u_cube
        assert be.to_numpy(be.abs(y_par - real.y))[0] < 50.0 * 2 * u_cube
