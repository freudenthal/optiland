"""Tests of grating orders in branch enumeration (both backends).

Closed forms: (a) a plane reflective grating (1200 lines/mm, 0.5 µm) in an
Ebert layout (one concave mirror used twice): each order branch follows the
grating equation, the evanescent orders are booked, ``sequence(key)``
reproduces each branch, and the order-1 branch equals the layout traced as
the sequence ``[0, 1, 2, 1, 3]`` over a single mirror entry; (b) a
transmission grating with an efficiency table: the ledger closes; (c) conical
incidence against the vector grating equation; (d) two gratings in series:
one branch per pair of orders, only the (+1, +1) branch passes the exit slit;
(e) the incidence domain of a Littrow scan.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

import optiland.backend as be
from optiland.interactions.diffractive_model import (
    DiffractiveInteractionModel,
    order_label,
    parse_order_label,
)
from optiland.materials import IdealMaterial
from optiland.optic import Optic
from optiland.physical_apertures import RadialAperture
from optiland.rays import AnisotropicRays, PolarizationState
from optiland.raytrace.branches import BranchChild, BranchTracer, SplittingModel
from optiland.raytrace.incidence import (
    Scan,
    grating_frame,
    grating_incidence,
    incidence_domain,
)
from optiland.sequences import SequencedOptic

TOL = 1e-12
PERIOD = 1 / 1.2  # 1200 lines/mm, in µm
WL = 0.5
TILT = math.radians(23.5)  # the grating of the Ebert layout
SLIT = (0.0, 20.0, -100.0)


def _np(x):
    return np.asarray(be.to_numpy(x), dtype=float)


def _state():
    return PolarizationState(
        is_polarized=True, Ex=0.6, Ey=0.8, phase_x=0.0, phase_y=0.0
    )


def _unpolarized():
    return PolarizationState(is_polarized=False)


def _rays_from(point, targets, wavelength=WL):
    """Rays from a point towards target points, shape (n,)."""
    p = np.asarray(point, dtype=float)
    d = np.asarray(targets, dtype=float) - p
    d /= np.linalg.norm(d, axis=1)[:, None]
    n = len(d)
    ones = np.ones(n)
    return AnisotropicRays(
        be.array(p[0] * ones),
        be.array(p[1] * ones),
        be.array(p[2] * ones),
        be.array(d[:, 0]),
        be.array(d[:, 1]),
        be.array(d[:, 2]),
        be.array(ones),
        be.array(wavelength * ones),
    )


def _slit_fan():
    targets = [(x, y, 0.0) for x in (-1.0, 0.0, 1.5) for y in (-3.0, 0.0, 2.0)]
    return _rays_from(SLIT, targets)


def _ebert(twice: bool = True, orders=(-2, -1, 0, 1, 2)) -> Optic:
    """The Ebert layout: slit (the launch point), a concave mirror, a plane
    reflective grating, the mirror again, the exit slit plane.

    With ``twice`` the mirror is listed twice (the second entry tied to the
    first by pickups); else once, for the sequence ``[0, 1, 2, 1, 3]``.
    """
    optic = Optic()
    optic.surfaces.add(index=0, radius=be.inf, thickness=be.inf)
    mirror = {
        "surface_type": "even_asphere",
        "radius": -200.0,
        "conic": -0.1,
        "coefficients": [0.0, 1e-9],
        "material": "mirror",
    }
    optic.surfaces.add(index=1, x=0.0, y=0.0, z=0.0, comment="M", **mirror)
    optic.surfaces.add(
        index=2, x=0.0, y=-20.0, z=-100.0, rx=TILT, material="mirror",
        surface_type="grating", grating_order=1, grating_period=PERIOD,
        groove_orientation_angle=0.0, comment="G", is_stop=True,
    )  # fmt: skip
    if twice:
        other = {**mirror, "radius": -150.0, "conic": 0.0, "coefficients": [0.0, 0.0]}
        optic.surfaces.add(index=3, x=0.0, y=5.0, z=1.0, comment="M2", **other)
        for attr in (
            "radius",
            "conic",
            "surfaces.surfaces[i].geometry.coefficients",
            "surfaces.surfaces[i].geometry.cs.x",
            "surfaces.surfaces[i].geometry.cs.y",
            "surfaces.surfaces[i].geometry.cs.z",
        ):
            optic.pickups.add(1, attr, 3)
    optic.surfaces.add(index=4 if twice else 3, x=0.0, y=-40.0, z=-100.0)
    optic.set_aperture(aperture_type="EPD", value=5.0)
    optic.fields.set_type(field_type="angle")
    optic.fields.add(y=0.0)
    optic.wavelengths.add(value=WL, is_primary=True)
    optic.updater.set_polarization(_state())
    optic.surfaces.surfaces[2].interaction_model.orders = orders
    return optic


def _grating_axes(tilt):
    """The normal and the grating vector of a plane grating turned by rx."""
    normal = np.array([0.0, -math.sin(tilt), math.cos(tilt)])
    g = np.array([0.0, math.cos(tilt), math.sin(tilt)])
    return normal, g


def _vector_grating_equation(k_in, normal, g, m, ratio, n1=1.0, n2=1.0, reflect=False):
    """n2 k_t,out = n1 k_t,in + m (λ/Λ) ĝ; the normal part leaves the surface
    (on the incident side for a reflection). Rows of k_in are unit vectors."""
    along = k_in @ normal
    tangential = n1 * (k_in - along[:, None] * normal) + m * ratio * g
    t = tangential / n2
    root = np.sqrt(1.0 - np.sum(t**2, axis=1))
    out = np.sign(along) * (-1.0 if reflect else 1.0)
    return t + (root * out)[:, None] * normal


def _record(view):
    return {name: _np(getattr(view, name)) for name in ("x", "y", "z", "L", "M", "N")}


def _same_rays(a, b, tol=TOL):
    for name in ("x", "y", "z", "L", "M", "N", "opd"):
        np.testing.assert_allclose(
            _np(getattr(a, name)), _np(getattr(b, name)), rtol=0, atol=tol
        )
    np.testing.assert_allclose(_np(a.p), _np(b.p), rtol=0, atol=tol)


# ----------------------------------------------------------------- the protocol


def test_order_labels():
    assert [order_label(m) for m in (-2, -1, 0, 1, 2)] == [
        "m-2",
        "m-1",
        "m0",
        "m+1",
        "m+2",
    ]
    for m in (-3, 0, 4):
        assert parse_order_label(order_label(m)) == m
    for bad in ("T", "m1", "m+0", "mx", "o"):
        with pytest.raises(ValueError):
            parse_order_label(bad)


def test_model_is_a_splitting_model(set_test_backend):
    optic = _ebert()
    model = optic.surfaces.surfaces[2].interaction_model
    assert isinstance(model, SplittingModel)
    children = model.branch_children(None)
    assert children == [BranchChild("F", (order_label(m),)) for m in (-2, -1, 0, 1, 2)]
    assert model.branch_children(None, (1,)) == [BranchChild("F", ("m+1",))]
    with pytest.raises(ValueError):
        BranchChild("X")
    with pytest.raises(ValueError):
        model.configure_branch(BranchChild("T", ("m+1",)), "G")
    with pytest.raises(ValueError):
        DiffractiveInteractionModel(None, False, orders=(1, 1))
    with pytest.raises(ValueError):
        DiffractiveInteractionModel(None, False, efficiency={1: 1.5})


def test_serialization_round_trip(set_test_backend):
    model = DiffractiveInteractionModel(
        None,
        False,
        orders=(-1, 0, 1),
        efficiency={-1: 0.2, 1: lambda w, t, p: w},
        label="G",
    )
    data = model.to_dict()
    assert data["orders"] == [-1, 0, 1]
    assert data["efficiency"] == {"-1": 0.2, "1": None}
    back = DiffractiveInteractionModel.from_dict(data, None)
    assert back.orders == (-1, 0, 1)
    assert back.efficiency == {-1: 0.2}
    assert back.label == "G"


def test_tracer_errors(set_test_backend):
    optic = _ebert()
    with pytest.raises(ValueError, match="splitting model"):
        BranchTracer(optic, ghosts=[2])
    with pytest.raises(ValueError, match="no splitting model"):
        BranchTracer(optic, orders={1: (0,)})
    tracer = BranchTracer(optic)
    with pytest.raises(ValueError):
        tracer.sequence((("G", "F", "m+3"),))
    with pytest.raises(ValueError):
        tracer.sequence((("G", "T", "m+1"),))


# --------------------------------------------------------------- (a) Ebert


def test_a_ebert_orders(set_test_backend):
    """Orders -2 … +2 of a plane reflective grating in an Ebert layout."""
    optic = _ebert()
    tracer = BranchTracer(optic, threshold=0.0)
    result = tracer.trace_all(rays=_slit_fan())

    keys = [(("G", "F", m),) for m in ("m0", "m+1", "m+2")]
    assert result.keys() == keys
    # -2 and -1 are evanescent (|k_t| > 1): the full power of two orders.
    assert result.ledger.evanescent == pytest.approx(2.0, abs=TOL)
    assert result.pruned == {}
    assert result.ledger.kept == pytest.approx(3.0, abs=TOL)  # no efficiency

    normal, g = _grating_axes(TILT)
    for key in keys:
        branch = result[key]
        before, after = _record(branch.views[1]), _record(branch.views[2])
        k_in = np.stack([before["L"], before["M"], before["N"]], axis=1)
        k_out = np.stack([after["L"], after["M"], after["N"]], axis=1)
        m = parse_order_label(key[0][2])
        expected = _vector_grating_equation(
            k_in, normal, g, m, WL / PERIOD, reflect=True
        )
        np.testing.assert_allclose(k_out, expected, rtol=0, atol=TOL)
        # sequence(key) reproduces the branch
        seq = tracer.sequence(key)
        rays = seq.surfaces.trace(_slit_fan())
        _same_rays(rays, branch.rays)
        assert rays.branch_key == key
        assert [v.base_surface for v in seq.surfaces] == [
            optic.surfaces.surfaces[i] for i in (0, 1, 2, 3, 4)
        ]


def test_a_ebert_single_mirror_sequence(set_test_backend):
    """The order-1 branch equals the layout with one mirror entry traced as
    the sequence [0, 1, 2, 1, 3]."""
    branch = BranchTracer(_ebert(), threshold=0.0).trace_all(rays=_slit_fan())[
        (("G", "F", "m+1"),)
    ]
    single = _ebert(twice=False)
    seq = SequencedOptic(single, "ebert", [0, 1, 2, 1, 3])
    rays = seq.surfaces.trace(_slit_fan())
    _same_rays(rays, branch.rays)
    # the running intensity and the ray power
    np.testing.assert_allclose(_np(rays.i), _np(branch.running_intensity), atol=TOL)


# ------------------------------------------------------- (b) efficiency table


def _transmission(efficiency, orders, period=2.0, n_after=1.0):
    optic = Optic()
    optic.surfaces.add(index=0, radius=be.inf, thickness=be.inf)
    optic.surfaces.add(
        index=1, thickness=10.0, is_stop=True, surface_type="grating",
        grating_order=1, grating_period=period, groove_orientation_angle=0.0,
        material=IdealMaterial(n_after), comment="G",
    )  # fmt: skip
    optic.surfaces.add(index=2, material=IdealMaterial(n_after))
    optic.set_aperture(aperture_type="EPD", value=4.0)
    optic.fields.set_type(field_type="angle")
    optic.fields.add(y=0.0)
    optic.wavelengths.add(value=WL, is_primary=True)
    optic.updater.set_polarization(_unpolarized())
    model = optic.surfaces.surfaces[1].interaction_model
    model.orders = orders
    model.efficiency = efficiency
    return optic


def _oblique_rays(theta, phi, n=4, wavelength=WL):
    """Rays at (θ, φ) in the grating frame of a plane grating with ĝ = ŷ and
    normal ẑ (x_g = ŷ, y_g = −x̂), reaching the plane z = 0 near the origin."""
    k = np.array(
        [
            -math.sin(theta) * math.sin(phi),
            math.sin(theta) * math.cos(phi),
            math.cos(theta),
        ]
    )
    starts = np.array([(0.3 * j, -0.2 * j, 0.0) for j in range(n)]) - k
    return AnisotropicRays(
        be.array(starts[:, 0]),
        be.array(starts[:, 1]),
        be.array(starts[:, 2]),
        be.array(np.full(n, k[0])),
        be.array(np.full(n, k[1])),
        be.array(np.full(n, k[2])),
        be.array(np.ones(n)),
        be.array(np.full(n, wavelength)),
    )


def test_b_efficiency_table_closes_the_ledger(set_test_backend):
    table = {-2: 0.05, -1: 0.25, 0: 0.4, 1: 0.25, 2: 0.05}
    optic = _transmission(table, (-2, -1, 0, 1, 2))
    result = BranchTracer(optic, threshold=0.0).trace_all(
        rays=_oblique_rays(math.radians(10), 0.0)
    )
    assert len(result) == 5
    assert abs(result.ledger.absorbed) < TOL
    assert abs(result.ledger.total() - 1.0) < TOL
    for m, value in table.items():
        assert result[(("G", "T", order_label(m)),)].power == pytest.approx(
            value, abs=TOL
        )


def test_b_efficiency_table_with_loss(set_test_backend):
    table = {-1: 0.3, 0: 0.3, 1: 0.3}  # sums to 0.9
    result = BranchTracer(_transmission(table, (-1, 0, 1)), threshold=0.0).trace_all(
        rays=_oblique_rays(0.0, 0.0)
    )
    assert result.ledger.kept == pytest.approx(0.9, abs=TOL)
    assert result.ledger.absorbed == pytest.approx(0.1, abs=TOL)


def test_b_callable_efficiency_and_evanescent_order(set_test_backend):
    """A callable efficiency of (λ, θ, φ); an evanescent order with
    efficiency is booked as evanescent, the orders not listed as absorbed."""
    seen = {}

    def eff(w, theta, phi):
        seen["theta"], seen["phi"], seen["w"] = _np(theta), _np(phi), _np(w)
        return 0.5 + 0 * w

    theta, phi = math.radians(20), math.radians(30)
    table = {0: eff, 1: 0.2, 4: 0.1}  # order 4 is evanescent at Λ = 2 µm
    optic = _transmission(table, (0, 1, 4))
    result = BranchTracer(optic, threshold=0.0).trace_all(
        rays=_oblique_rays(theta, phi)
    )
    np.testing.assert_allclose(seen["theta"], 20.0, atol=1e-12)
    np.testing.assert_allclose(seen["phi"], 30.0, atol=1e-12)
    np.testing.assert_allclose(seen["w"], WL, atol=0)
    assert result.keys() == [(("G", "T", "m0"),), (("G", "T", "m+1"),)]
    assert result.ledger.evanescent == pytest.approx(0.1, abs=TOL)
    assert result.ledger.absorbed == pytest.approx(0.2, abs=TOL)
    assert result.ledger.kept == pytest.approx(0.7, abs=TOL)


def test_b_orders_argument_selects_orders(set_test_backend):
    table = {-1: 0.3, 0: 0.4, 1: 0.3}
    optic = _transmission(table, (-1, 0, 1))
    result = BranchTracer(optic, threshold=0.0, orders={1: (1, -1)}).trace_all(
        rays=_oblique_rays(0.0, 0.0)
    )
    assert result.keys() == [(("G", "T", "m+1"),), (("G", "T", "m-1"),)]
    assert result.ledger.absorbed == pytest.approx(0.4, abs=TOL)
    # the threshold prunes a weak order as it prunes a weak mode
    pruned = BranchTracer(optic, threshold=0.35).trace_all(rays=_oblique_rays(0.0, 0.0))
    assert pruned.keys() == [(("G", "T", "m0"),)]
    assert pruned.pruned[(("G", "T", "m-1"),)] == (
        pytest.approx(0.3, abs=TOL),
        "threshold",
    )


def test_b_nominal_trace_uses_the_efficiency(set_test_backend):
    optic = _transmission({1: 0.25}, None)
    optic.updater.set_polarization("ignore")
    rays = optic.trace(0.0, 0.0, WL, 3)
    np.testing.assert_allclose(_np(rays.i), 0.25, atol=TOL)
    np.testing.assert_allclose(_np(rays.M), WL / 2.0, atol=TOL)


# ----------------------------------------------------- (c) conical incidence


@pytest.mark.parametrize("n_after", [1.0, 1.5])
def test_c_conical_incidence(set_test_backend, n_after):
    theta, phi = math.radians(20), math.radians(30)
    ratio = 0.6328 / 1.0
    optic = _transmission(None, (-2, -1, 0, 1, 2), period=1.0, n_after=n_after)
    rays = _oblique_rays(theta, phi, wavelength=0.6328)
    result = BranchTracer(optic, threshold=0.0).trace_all(rays=rays)
    k_in = np.stack([_np(rays.L), _np(rays.M), _np(rays.N)], axis=1)
    normal, g = np.array([0.0, 0.0, 1.0]), np.array([0.0, 1.0, 0.0])
    propagating = []
    for m in (-2, -1, 0, 1, 2):
        t = k_in[0] - (k_in[0] @ normal) * normal + m * ratio * g
        if t @ t < n_after**2:
            propagating.append(m)
    assert result.keys() == [(("G", "T", order_label(m)),) for m in propagating]
    lost = 5 - len(propagating)
    assert lost > 0
    assert result.ledger.evanescent == pytest.approx(lost, abs=TOL)
    for m in propagating:
        r = result[(("G", "T", order_label(m)),)].rays
        k_out = np.stack([_np(r.L), _np(r.M), _np(r.N)], axis=1)
        expected = _vector_grating_equation(k_in, normal, g, m, ratio, n2=n_after)
        np.testing.assert_allclose(k_out, expected, rtol=0, atol=TOL)
    # the incidence in the grating frame
    theta_deg, phi_deg = grating_incidence(
        (rays.L, rays.M, rays.N),
        (be.zeros_like(rays.L), be.zeros_like(rays.L), -be.ones_like(rays.L)),
        (be.zeros_like(rays.L), be.ones_like(rays.L), 0.1 * be.ones_like(rays.L)),
    )
    np.testing.assert_allclose(_np(theta_deg), 20.0, atol=TOL)
    np.testing.assert_allclose(_np(phi_deg), 30.0, atol=TOL)


def test_c_grating_frame(set_test_backend):
    k = (be.array([0.0, 0.3]), be.array([0.0, 0.0]), be.array([1.0, math.sqrt(0.91)]))
    normal = (be.array([0.0, 0.0]), be.array([0.0, 0.0]), be.array([-1.0, -1.0]))
    g = (be.array([1.0, 1.0]), be.array([0.0, 0.0]), be.array([0.5, 0.5]))
    x, y, z = grating_frame(k, normal, g)
    np.testing.assert_allclose(np.stack([_np(c) for c in z]), [[0, 0], [0, 0], [1, 1]])
    np.testing.assert_allclose(np.stack([_np(c) for c in x]), [[1, 1], [0, 0], [0, 0]])
    np.testing.assert_allclose(np.stack([_np(c) for c in y]), [[0, 0], [1, 1], [0, 0]])
    theta, phi = grating_incidence(k, normal, g)
    np.testing.assert_allclose(
        _np(theta), [0.0, math.degrees(math.asin(0.3))], atol=TOL
    )
    np.testing.assert_allclose(
        _np(phi), [0.0, 0.0], atol=TOL
    )  # φ = 0 at normal incidence
    with pytest.raises(ValueError):
        grating_frame(k, normal, normal)


# ------------------------------------------------ (d) two gratings in series


def test_d_double_grating_exit_slit(set_test_backend):
    eff1 = {-1: 0.3, 0: 0.4, 1: 0.3}
    eff2 = {-1: 0.25, 0: 0.5, 1: 0.25}
    f = 100.0
    ratio1, ratio2 = WL / 2.0, WL / 3.0
    y_slit = f * math.tan(math.asin(ratio1 + ratio2))  # the (+1, +1) focus

    optic = Optic()
    optic.surfaces.add(index=0, radius=be.inf, thickness=be.inf)
    grating = {
        "surface_type": "grating",
        "grating_order": 1,
        "groove_orientation_angle": 0.0,
    }
    optic.surfaces.add(
        index=1,
        thickness=5.0,
        is_stop=True,
        grating_period=2.0,
        comment="G1",
        **grating,
    )
    optic.surfaces.add(
        index=2, thickness=5.0, grating_period=3.0, comment="G2", **grating
    )
    optic.surfaces.add(index=3, surface_type="paraxial", f=f, thickness=f)
    optic.surfaces.add(
        index=4, x=0.0, y=y_slit, z=10.0 + f, aperture=RadialAperture(r_max=1.0),
        comment="slit",
    )  # fmt: skip
    optic.surfaces.add(index=5, x=0.0, y=y_slit, z=11.0 + f)
    optic.set_aperture(aperture_type="EPD", value=4.0)
    optic.fields.set_type(field_type="angle")
    optic.fields.add(y=0.0)
    optic.wavelengths.add(value=WL, is_primary=True)
    optic.updater.set_polarization(_unpolarized())
    for index, eff in ((1, eff1), (2, eff2)):
        model = optic.surfaces.surfaces[index].interaction_model
        model.orders = (-1, 0, 1)
        model.efficiency = eff

    result = BranchTracer(optic, threshold=0.0).trace_all(rays=_oblique_rays(0.0, 0.0))
    assert len(result) == 9  # 3 × 3 orders
    passed = {key: b.power for key, b in result.branches.items() if b.power > 0}
    key = (("G1", "T", "m+1"), ("G2", "T", "m+1"))
    assert list(passed) == [key]
    assert passed[key] == pytest.approx(0.3 * 0.25, abs=TOL)
    assert result.ledger.clipped == pytest.approx(1 - 0.3 * 0.25, abs=TOL)
    assert abs(result.ledger.absorbed) < TOL
    # the passing branch lands at the slit centre (paraxial lens: exact focus)
    ys = _np(result[key].views[4].y)
    np.testing.assert_allclose(ys, y_slit, atol=1e-9)


# ----------------------------------------------------- (e) incidence domain


def _littrow_optic(tilt):
    optic = Optic()
    optic.surfaces.add(index=0, radius=be.inf, thickness=be.inf)
    optic.surfaces.add(
        index=1, x=0.0, y=0.0, z=0.0, rx=tilt, material="mirror",
        surface_type="grating", grating_order=-1, grating_period=PERIOD,
        groove_orientation_angle=0.0, is_stop=True, comment="G",
    )  # fmt: skip
    optic.surfaces.add(index=2, x=0.0, y=0.0, z=-20.0)
    optic.set_aperture(aperture_type="EPD", value=4.0)
    optic.fields.set_type(field_type="angle")
    optic.fields.add(y=0.0)
    optic.wavelengths.add(value=0.45, is_primary=True)
    return optic


def test_e_incidence_domain_littrow_scan(set_test_backend):
    wavelengths = (0.45, 0.5, 0.55)
    tilts = [math.asin(w / (2 * PERIOD)) for w in wavelengths]  # Littrow, m = -1
    scan = Scan(
        wavelengths=wavelengths,
        configurations=[lambda _o, t=t: _littrow_optic(t) for t in tilts],
        num_rays=3,
    )
    domain = incidence_domain(_littrow_optic(tilts[0]), 1, scan)
    assert domain.theta_deg[0] == pytest.approx(math.degrees(tilts[0]), abs=TOL)
    assert domain.theta_deg[1] == pytest.approx(math.degrees(tilts[-1]), abs=TOL)
    assert domain.phi_deg == pytest.approx((0.0, 0.0), abs=TOL)
    assert domain.wavelength == (0.45, 0.55)
    kx = domain.k_parallel[0]
    assert kx == pytest.approx((math.sin(tilts[0]), math.sin(tilts[-1])), abs=TOL)
    assert domain.samples["theta_deg"].size == 3 * 3 * 37
    assert np.all(domain.contains(domain.samples["theta_deg"], domain.samples["phi_deg"],
                                  domain.samples["wavelength"]))  # fmt: skip
    assert not domain.contains(math.degrees(tilts[-1]) + 1e-6, 0.0, 0.5)
    assert not domain.contains(20.0, 0.0, 0.6)

    # every traced ray lies in the domain, and each Littrow order returns
    for tilt, wavelength in zip(tilts, wavelengths, strict=True):
        optic = _littrow_optic(tilt)
        rays = optic.trace(0.0, 0.0, wavelength, 3)
        np.testing.assert_allclose(_np(rays.N), -1.0, atol=TOL)
        before = optic.surfaces.surfaces[0]
        theta, phi = grating_incidence(
            (before.L, before.M, before.N),
            (
                0 * before.L,
                -math.sin(tilt) + 0 * before.L,
                math.cos(tilt) + 0 * before.L,
            ),
            (
                0 * before.L,
                math.cos(tilt) + 0 * before.L,
                math.sin(tilt) + 0 * before.L,
            ),
        )
        assert np.all(domain.contains(theta, phi, wavelength))

    # margins widen the ranges
    wide = incidence_domain(
        _littrow_optic(tilts[0]), 1, scan, angle_margin_deg=0.5, wavelength_margin=0.01
    )
    assert wide.theta_deg == pytest.approx(
        (domain.theta_deg[0] - 0.5, domain.theta_deg[1] + 0.5), abs=TOL
    )
    assert wide.phi_deg == pytest.approx((-0.5, 0.5), abs=TOL)
    assert wide.wavelength == pytest.approx((0.44, 0.56), abs=TOL)


def test_e_incidence_domain_of_a_sequence(set_test_backend):
    """The domain at the grating step of a sequence equals the domain at the
    grating surface of the optic; a surface that is not a grating raises."""
    optic = _littrow_optic(math.radians(10.0))
    optic.fields.add(y=3.0)
    scan = Scan(wavelengths=(0.5,), fields=((0.0, 0.0), (0.0, 1.0)), num_rays=3)
    seq = SequencedOptic(optic, "littrow", [0, 1, 2])
    a = incidence_domain(optic, 1, scan)
    b = incidence_domain(seq, 1, scan)
    np.testing.assert_allclose(a.samples["theta_deg"], b.samples["theta_deg"], atol=TOL)
    np.testing.assert_allclose(a.samples["phi_deg"], b.samples["phi_deg"], atol=TOL)
    # the 3° field turns the beam in the plane of the grating vector
    assert a.theta_deg[1] - a.theta_deg[0] == pytest.approx(3.0, abs=1e-9)
    assert a.phi_deg == pytest.approx((0.0, 0.0), abs=TOL)
    with pytest.raises(ValueError, match="not a grating"):
        incidence_domain(optic, 2, scan)
