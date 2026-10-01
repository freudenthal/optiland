"""Tests of OpticUpdater.set_radius on geometries without their own set_radius."""

from __future__ import annotations

import pytest

import optiland.backend as be
from optiland.coordinate_system import CoordinateSystem
from optiland.geometries import (
    Plane,
    PlaneGrating,
    StandardGeometry,
    StandardGratingGeometry,
)
from optiland.geometries.grid_sag import GridSagGeometry
from optiland.multiconfig.multi_configuration import MultiConfiguration
from optiland.optic import Optic

from .utils import assert_allclose

WAVELENGTH = 0.587


def _grating_optic(radius=be.inf, order=-1, period=5.0, angle=0.3):
    """A transmission grating behind a glass plate, at the stop."""
    optic = Optic()
    optic.surfaces.add(index=0, radius=be.inf, thickness=be.inf)
    optic.surfaces.add(index=1, radius=be.inf, thickness=5, material="N-BK7")
    optic.surfaces.add(
        index=2,
        radius=radius,
        thickness=30,
        surface_type="grating",
        grating_order=order,
        grating_period=period,
        groove_orientation_angle=angle,
        is_stop=True,
    )
    optic.surfaces.add(index=3)
    optic.set_aperture(aperture_type="EPD", value=10.0)
    optic.fields.set_type(field_type="angle")
    optic.fields.add(y=0)
    optic.fields.add(y=5)
    optic.wavelengths.add(value=WAVELENGTH, is_primary=True)
    return optic


def _trace(optic):
    return optic.trace(
        Hx=0.3, Hy=1.0, wavelength=WAVELENGTH, num_rays=4, distribution="hexapolar"
    )


def test_multiconfiguration_keeps_plane_grating(set_test_backend):
    """The radius link of a new configuration keeps a plane grating."""
    optic = _grating_optic()
    reference = _trace(_grating_optic())

    config = MultiConfiguration(optic).add_configuration()

    assert type(config.surfaces[2].geometry) is PlaneGrating
    rays = _trace(config)
    for name in ("x", "y", "z", "L", "M", "N"):
        assert_allclose(getattr(rays, name), getattr(reference, name), 0, 1e-12)


@pytest.mark.parametrize("geometry_type", [Plane, PlaneGrating])
def test_infinite_radius_on_flat_geometry_changes_nothing(
    set_test_backend, geometry_type
):
    optic = _grating_optic()
    if geometry_type is Plane:
        index = 1
    else:
        index = 2
    geometry = optic.surfaces[index].geometry
    assert type(geometry) is geometry_type

    optic.updater.set_radius(be.inf, index)

    assert optic.surfaces[index].geometry is geometry


def test_finite_radius_on_plane_gives_standard_geometry(set_test_backend):
    optic = _grating_optic()
    cs = optic.surfaces[1].geometry.cs

    optic.updater.set_radius(-40.0, 1)

    geometry = optic.surfaces[1].geometry
    assert type(geometry) is StandardGeometry
    assert geometry.cs is cs
    assert_allclose(geometry.radius, -40.0)
    assert_allclose(geometry.k, 0.0)


def test_finite_radius_on_plane_grating_gives_standard_grating(set_test_backend):
    """The converted grating traces as a curved grating built directly."""
    optic = _grating_optic()
    cs = optic.surfaces[2].geometry.cs

    optic.updater.set_radius(-60.0, 2)

    geometry = optic.surfaces[2].geometry
    assert type(geometry) is StandardGratingGeometry
    assert geometry.cs is cs
    assert_allclose(geometry.radius, -60.0)
    assert_allclose(geometry.grating_order, -1)
    assert_allclose(geometry.grating_period, 5.0)
    assert_allclose(geometry.groove_orientation_angle, 0.3)

    rays = _trace(optic)
    reference = _trace(_grating_optic(radius=-60.0))
    for name in ("x", "y", "z", "L", "M", "N"):
        assert_allclose(getattr(rays, name), getattr(reference, name), 0, 1e-12)


def test_geometry_without_radius_raises(set_test_backend):
    optic = _grating_optic()
    cs = CoordinateSystem()
    grid = GridSagGeometry(
        cs,
        [-1.0, 0.0, 1.0],
        [-1.0, 0.0, 1.0],
        [[0.1, 0.2, 0.1], [0.2, 0.4, 0.2], [0.1, 0.2, 0.1]],
    )
    optic.surfaces[1].geometry = grid

    with pytest.raises(ValueError, match="GridSagGeometry"):
        optic.updater.set_radius(25.0, 1)
    assert optic.surfaces[1].geometry is grid


def test_geometry_with_set_radius_takes_value(set_test_backend):
    optic = _grating_optic(radius=-60.0)
    geometry = optic.surfaces[2].geometry

    optic.updater.set_radius(-75.0, 2)

    assert optic.surfaces[2].geometry is geometry
    assert_allclose(geometry.radius, -75.0)
