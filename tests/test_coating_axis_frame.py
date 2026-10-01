"""The axis of a polarized coating is a global axis on a turned surface.

During the interaction the rays are in the local frame of the surface. The
interaction model gives the coating the rotation R from the global to the local
frame, and the coating uses R · axis. The checks launch a ray along the local
normal (normal incidence) and compare the polarization ray-tracing matrix with
the matrix of the ideal element in the same frame:

* linear polarizer: a a^T + k k^T,
* linear retarder: exp(-i d / 2) a a^T + exp(i d / 2) b b^T + k k^T,
* linear diattenuator: t_max a a^T + t_min b b^T + k k^T,

where a is the unit transverse part of R · axis, k the ray direction and
b = k x a.
"""

from __future__ import annotations

import numpy as np
import pytest

import optiland.backend as be
from optiland.coatings import (
    BaseCoatingPolarized,
    FresnelCoating,
    PolarizerCoating,
    RetarderCoating,
)
from optiland.coordinate_system import CoordinateSystem
from optiland.jones import (
    JonesLinearDiattenuator,
    JonesLinearPolarizer,
    JonesLinearRetarder,
)
from optiland.materials import IdealMaterial
from optiland.optic import Optic
from optiland.rays import PolarizedRays

from .utils import assert_allclose

RETARDANCE = 0.7
T_MIN, T_MAX = 0.3, 0.9
AXIS = np.array([1.0, 2.0, 0.5]) / np.linalg.norm([1.0, 2.0, 0.5])


class DiattenuatorTestCoating(BaseCoatingPolarized):
    """A coating with a linear diattenuator (the Jones class has no coating)."""

    def __init__(self, axis):
        self._jones = JonesLinearDiattenuator(T_MIN, T_MAX, axis)

    @property
    def jones(self):
        return self._jones


def _coating(kind, axis):
    axis = tuple(float(a) for a in axis)
    if kind == "polarizer":
        return PolarizerCoating(axis=axis)
    if kind == "retarder":
        return RetarderCoating(RETARDANCE, axis=axis)
    return DiattenuatorTestCoating(axis)


def _hand_matrix(kind, axis_local):
    """The ideal element (3, 3) for a ray along +z in the local frame."""
    k = np.array([0.0, 0.0, 1.0])
    a = axis_local - (axis_local @ k) * k
    a = a / np.linalg.norm(a)
    b = np.cross(k, a)
    kk = np.outer(k, k)
    if kind == "polarizer":
        return np.outer(a, a) + kk
    if kind == "retarder":
        return (
            np.exp(-0.5j * RETARDANCE) * np.outer(a, a)
            + np.exp(0.5j * RETARDANCE) * np.outer(b, b)
            + kk
        )
    return T_MAX * np.outer(a, a) + T_MIN * np.outer(b, b) + kk


def _rot(axis, angle):
    """The matrix of RealRays.rotate_x/y/z (axis 0, 1, 2) for the angle."""
    c, s = np.cos(angle), np.sin(angle)
    i, j = [(1, 2), (0, 2), (0, 1)][axis]
    m = np.eye(3)
    m[i, i], m[j, j] = c, c
    if axis == 1:
        m[i, j], m[j, i] = s, -s
    else:
        m[i, j], m[j, i] = -s, s
    return m


def _global_to_local(rx, ry, rz, reference=None):
    """R of CoordinateSystem.localize: reference first, then rz, ry, rx."""
    r = _rot(0, -rx) @ _rot(1, -ry) @ _rot(2, -rz)
    if reference is not None:
        r = r @ _global_to_local(*reference)
    return r


FRAMES = {
    "rx": ((0.6, 0.0, 0.0), None),
    "ry": ((0.0, -0.45, 0.0), None),
    "rz": ((0.0, 0.0, 0.8), None),
    "rx ry rz": ((0.3, -0.2, 0.5), None),
    "nested reference_cs": ((0.25, 0.1, -0.3), (0.4, -0.35, 0.2)),
}


def _surface(coating, frame, surface_type="standard"):
    (rx, ry, rz), reference = FRAMES[frame]
    optic = Optic()
    optic.surfaces.add(index=0, radius=be.inf, thickness=be.inf)
    kwargs = {"f": 50.0} if surface_type == "paraxial" else {}
    optic.surfaces.add(
        index=1,
        surface_type=surface_type,
        thickness=5.0,
        material=IdealMaterial(n=1.0),
        coating=coating,
        is_stop=True,
        **kwargs,
    )
    optic.surfaces.add(index=2)
    surface = optic.surfaces[1]
    reference_cs = None
    if reference is not None:
        reference_cs = CoordinateSystem(
            rx=reference[0], ry=reference[1], rz=reference[2]
        )
    surface.geometry.cs = CoordinateSystem(
        x=1.0, y=-2.0, z=3.0, rx=rx, ry=ry, rz=rz, reference_cs=reference_cs
    )
    return surface, _global_to_local(rx, ry, rz, reference)


def _with_incident_direction(rays):
    """Set the incident direction, as a previous surface does."""
    rays.L0, rays.M0, rays.N0 = be.copy(rays.L), be.copy(rays.M), be.copy(rays.N)
    return rays


def _normal_ray():
    """One polarized ray at the vertex along the local normal."""
    zero = be.zeros(1)
    one = be.ones(1)
    rays = PolarizedRays(zero, zero, zero, zero, zero, one, one, be.array([0.55]))
    return _with_incident_direction(rays)


def _interact(surface):
    rays = surface.interaction_model.interact_real_rays(_normal_ray())
    return be.to_numpy(rays.p)[0]


@pytest.mark.parametrize("surface_type", ["standard", "paraxial"])
@pytest.mark.parametrize("frame", list(FRAMES))
@pytest.mark.parametrize("kind", ["polarizer", "retarder", "diattenuator"])
def test_global_axis_on_turned_surface(set_test_backend, kind, frame, surface_type):
    surface, r = _surface(_coating(kind, AXIS), frame, surface_type)
    assert_allclose(surface.interaction_model._local_rotation(), r, 0, 1e-14)

    expected = _hand_matrix(kind, r @ AXIS)
    assert np.allclose(_interact(surface), expected, rtol=0, atol=1e-12)


@pytest.mark.parametrize("kind", ["polarizer", "retarder", "diattenuator"])
def test_in_plane_axis_on_tilted_surface(set_test_backend, kind):
    """A 45 degree axis in the plane of a surface turned by rx = 36 degrees
    acts at 45 degrees (not at atan(cos(36) tan(45)) = 39 degrees)."""
    rx = np.deg2rad(36.0)
    local = np.array([np.cos(np.pi / 4), np.sin(np.pi / 4), 0.0])
    axis = _rot(0, rx) @ local  # the global vector of the local 45 degree axis
    surface, _ = _surface(_coating(kind, axis), "rx")
    surface.geometry.cs = CoordinateSystem(rx=rx)

    result = _interact(surface)

    assert np.allclose(result, _hand_matrix(kind, local), rtol=0, atol=1e-12)
    if kind == "polarizer":
        angle = np.degrees(np.arctan2(result[1, 1].real, result[0, 1].real))
        assert abs(angle - 45.0) < 1e-10


@pytest.mark.parametrize("kind", ["polarizer", "retarder", "diattenuator"])
def test_untilted_surface_unchanged(set_test_backend, kind):
    surface, _ = _surface(_coating(kind, AXIS), "rx")
    surface.geometry.cs = CoordinateSystem(x=1.0, y=-2.0, z=3.0)

    assert np.allclose(
        _interact(surface), _hand_matrix(kind, AXIS), rtol=0, atol=1e-12
    )


@pytest.mark.parametrize("kind", ["polarizer", "retarder", "diattenuator"])
def test_rotation_none_keeps_the_matrix(set_test_backend, kind):
    if kind == "polarizer":
        jones = JonesLinearPolarizer(AXIS)
    elif kind == "retarder":
        jones = JonesLinearRetarder(RETARDANCE, AXIS)
    else:
        jones = JonesLinearDiattenuator(T_MIN, T_MAX, AXIS)
    rays = PolarizedRays(
        be.zeros(2),
        be.zeros(2),
        be.zeros(2),
        be.array([0.0, 0.3]),
        be.array([0.0, -0.2]),
        be.array([1.0, np.sqrt(1 - 0.13)]),
        be.ones(2),
        be.array([0.55, 0.55]),
    )
    rays = _with_incident_direction(rays)
    old = jones.calculate_matrix(rays, reflect=False, aoi=be.zeros(2))
    new = jones.calculate_matrix(rays, reflect=False, aoi=be.zeros(2), rotation=None)
    assert_allclose(be.real(new), be.real(old), 0, 0)
    assert_allclose(be.imag(new), be.imag(old), 0, 0)

    r = _global_to_local(0.3, -0.2, 0.5)
    rotated = jones.calculate_matrix(rays, aoi=be.zeros(2), rotation=be.array(r))
    axis = jones.axis
    jones.axis = be.array(r @ be.to_numpy(axis))
    expected = jones.calculate_matrix(rays, aoi=be.zeros(2))
    jones.axis = axis
    assert_allclose(be.real(rotated), be.real(expected), 0, 1e-14)
    assert_allclose(be.imag(rotated), be.imag(expected), 0, 1e-14)


def test_coating_without_axis_ignores_the_frame(set_test_backend):
    """A Fresnel coating at normal incidence is the same on a turned surface."""
    air, glass = IdealMaterial(n=1.0), IdealMaterial(n=1.5)
    results = []
    for frame in ("rx ry rz", None):
        surface, _ = _surface(FresnelCoating(air, glass), "rx ry rz")
        surface.material_post = glass
        if frame is None:
            surface.geometry.cs = CoordinateSystem()
        results.append(_interact(surface))
    assert np.allclose(results[0], results[1], rtol=0, atol=1e-14)


@pytest.mark.parametrize("kind", ["polarizer", "retarder"])
def test_serialization_unchanged(set_test_backend, kind):
    coating = _coating(kind, AXIS)
    data = coating.to_dict()
    expected = {"type": type(coating).__name__, "axis": list(AXIS)}
    if kind == "retarder":
        expected["retardance"] = RETARDANCE
    assert data == expected
    restored = type(coating).from_dict(data)
    assert_allclose(restored.jones.axis, coating.jones.axis, 0, 0)


def test_rotation_keeps_torch_gradient():
    """The rotation is computed with backend operations (no conversion to a
    Python number), so the gradient reaches the surface tilt."""
    pytest.importorskip("torch")
    be.set_backend("torch")
    be.set_device("cpu")
    be.grad_mode.enable()
    be.set_precision("float64")
    try:
        import torch

        surface, _ = _surface(_coating("polarizer", AXIS), "rx")
        rx = torch.tensor(0.6, dtype=torch.float64, requires_grad=True)
        surface.geometry.cs._rx = rx
        rays = surface.interaction_model.interact_real_rays(_normal_ray())
        transmitted = torch.abs(rays.p[0, 0, 0]) ** 2
        transmitted.backward()
        assert rx.grad is not None
        assert torch.isfinite(rx.grad)
        assert float(rx.grad) != 0.0
    finally:
        be.set_backend("numpy")
