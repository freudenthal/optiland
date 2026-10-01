"""Tests of OptimizationProblem.update_optics with optics that own no variable."""

from __future__ import annotations

import optiland.backend as be
from optiland.multiconfig.multi_configuration import MultiConfiguration
from optiland.optic import Optic
from optiland.optimization import LeastSquares, OptimizationProblem

from .utils import assert_allclose

TARGET_F2 = 80.0


def _singlet():
    optic = Optic()
    optic.surfaces.add(index=0, radius=be.inf, thickness=be.inf)
    optic.surfaces.add(index=1, radius=60.0, thickness=5.0, material="N-BK7")
    optic.surfaces.add(index=2, radius=-60.0, thickness=50.0, is_stop=True)
    optic.surfaces.add(index=3)
    optic.set_aperture(aperture_type="EPD", value=10.0)
    optic.fields.set_type(field_type="angle")
    optic.fields.add(y=0)
    optic.wavelengths.add(value=0.55, is_primary=True)
    return optic


def _linked_problem():
    """A variable on configuration 0 and an operand on configuration 1 only."""
    mc = MultiConfiguration(_singlet())
    config0 = mc.current_config(0)
    config1 = mc.add_configuration()
    problem = OptimizationProblem()
    problem.add_operand(
        operand_type="f2", target=TARGET_F2, weight=1.0, input_data={"optic": config1}
    )
    problem.add_variable(config0, "radius", surface_number=1)
    return problem, config0, config1


def test_dependent_optic_follows_during_optimization(set_test_backend):
    problem, config0, config1 = _linked_problem()

    LeastSquares(problem).optimize(maxiter=50, tol=1e-12)

    radius0 = config0.surfaces[1].geometry.radius
    radius1 = config1.surfaces[1].geometry.radius
    assert_allclose(radius1, radius0, 0, 0)
    assert abs(float(be.to_numpy(radius0)) - 60.0) > 1.0

    fresh = _singlet()
    fresh.updater.set_radius(radius0, 1)
    assert_allclose(fresh.paraxial.f2(), config1.paraxial.f2(), 0, 0)
    assert_allclose(config1.paraxial.f2(), TARGET_F2, 0, 1e-6)


def test_operand_value_agrees_with_reevaluation(set_test_backend):
    problem, _, config1 = _linked_problem()

    LeastSquares(problem).optimize(maxiter=50, tol=1e-12)
    seen = problem.operands[0].value
    config1.updater.update()

    assert_allclose(seen, TARGET_F2, 0, 1e-6)
    assert_allclose(problem.operands[0].value, seen, 0, 0)


def test_each_optic_is_updated_once(set_test_backend, monkeypatch):
    """No extra update when the operand optic is the variable optic."""
    optic = _singlet()
    problem = OptimizationProblem()
    problem.add_operand(
        operand_type="f2", target=TARGET_F2, weight=1.0, input_data={"optic": optic}
    )
    problem.add_operand(
        operand_type="f1", target=-TARGET_F2, weight=1.0, input_data={"optic": optic}
    )
    problem.add_variable(optic, "radius", surface_number=1)
    problem.add_variable(optic, "radius", surface_number=2)

    calls = []
    monkeypatch.setattr(optic.updater, "update", lambda: calls.append(optic))
    problem.update_optics()

    assert calls == [optic]


def test_variable_optics_are_updated_before_operand_optics(set_test_backend):
    problem, config0, config1 = _linked_problem()

    order = []
    for optic in (config0, config1):
        optic.updater.update = lambda optic=optic: order.append(optic)
    problem.update_optics()

    assert order == [config0, config1]
