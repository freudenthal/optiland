"""Interaction model for diffraction.

This module implements the DiffractiveInteractionModel class, which handles
ray interactions with surfaces through diffraction.

A sequential trace follows one diffraction order, ``geometry.grating_order``.
The model can also list several orders, each with an optional efficiency, for
branch enumeration (``optiland.raytrace.branches``): ``BranchTracer`` then
traces one branch per listed order, and the model traces the order of each
branch. The branch key entry of an order is ``(label, "T", "m<±k>")`` for a
transmission grating and ``(label, "F", "m<±k>")`` for a reflective grating
(a mirror of the listing: the branch keeps the listed order of the surfaces),
with m in the sign of the grating equation of ``RealRays.gratingdiffract``:
n2 k_t,out = n1 k_t,in + m (λ/Λ) ĝ.

Polarization: the model keeps Optiland's behaviour for every order. Without a
coating the PRT of an order is the geometrical transformation from the
incident to the diffracted local (s, p) frame (the Jones matrix is the
identity); a ``FresnelCoating`` applies the planar Fresnel coefficients of the
interface in the frame of the diffracted ray. The efficiency is the fraction
of the incident power in the order; it multiplies the flux factor of
polarized rays (the ray power) and the intensity of unpolarized rays.

Kramer Harrison, 2025
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import TYPE_CHECKING, Any

import optiland.backend as be
from optiland.interactions.refractive_reflective_model import (
    RefractiveReflectiveModel,
)

if TYPE_CHECKING:
    # pragma: no cover
    from optiland.rays import ParaxialRays, RealRays

#: An order efficiency: a number, or a callable of (wavelength in µm, θ in
#: degrees, φ in degrees) that returns one value per ray.
Efficiency = float | Callable[[Any, Any, Any], Any]


def order_label(order: int) -> str:
    """Return the branch key label of a diffraction order: ``m0``, ``m+1``, ``m-2``."""
    order = int(order)
    return "m0" if order == 0 else f"m{order:+d}"


def parse_order_label(label: str) -> int:
    """Return the order of a label made by :func:`order_label`.

    Raises:
        ValueError: If the label is not an order label.
    """
    if not label.startswith("m"):
        raise ValueError(f"{label!r} is not a diffraction order label.")
    try:
        order = int(label[1:])
    except ValueError:
        raise ValueError(f"{label!r} is not a diffraction order label.") from None
    if order_label(order) != label:
        raise ValueError(f"{label!r} is not a diffraction order label.")
    return order


class DiffractiveInteractionModel(RefractiveReflectiveModel):
    """Interaction model for diffraction.

    Args:
        parent_surface: The surface of the grating.
        is_reflective: True for a reflective grating.
        coating: An optional coating.
        bsdf: An optional BSDF.
        orders: The orders that branch enumeration lists, in branch order.
            None lists the single order ``geometry.grating_order``.
        efficiency: None (every order carries the power of the incident ray,
            as in a sequential trace), or ``{order: efficiency}``: a number or
            a callable ``f(wavelength, theta_deg, phi_deg)`` of the incident
            ray in the grating frame
            (:func:`optiland.raytrace.incidence.grating_incidence`).
            An order that is not in the mapping has efficiency 0.
        label: The surface label in branch keys (None: the surface comment,
            else ``s<index>``).

    Attributes:
        order: The order this model traces; None traces
            ``geometry.grating_order``. Set per branch by
            :meth:`configure_branch`.
    """

    interaction_type = "diffractive"

    def __init__(
        self,
        parent_surface: Any,
        is_reflective: bool,
        coating: Any = None,
        bsdf: Any = None,
        orders: Any = None,
        efficiency: Mapping[int, Efficiency] | None = None,
        label: str | None = None,
    ):
        super().__init__(parent_surface, is_reflective, coating, bsdf)
        self.orders = None if orders is None else tuple(int(m) for m in orders)
        if self.orders is not None and len(set(self.orders)) != len(self.orders):
            raise ValueError(f"The orders {self.orders} repeat an order.")
        if efficiency is not None:
            efficiency = {int(m): value for m, value in efficiency.items()}
            for m, value in efficiency.items():
                if not callable(value) and not 0.0 <= float(value) <= 1.0:
                    raise ValueError(
                        f"The efficiency of order {m} is {value}, not in [0, 1]."
                    )
        self.efficiency = efficiency
        self.label = label
        self.order: int | None = None
        self._branch_entry: tuple[str, ...] | None = None

    # -- serialization ---------------------------------------------------------

    def to_dict(self):
        """Returns a dictionary representation of the model.

        A callable efficiency is not serialized (stored as None).
        """
        data = super().to_dict()
        efficiency = None
        if self.efficiency is not None:
            efficiency = {
                str(m): (None if callable(value) else float(value))
                for m, value in self.efficiency.items()
            }
        data.update(
            {
                "orders": None if self.orders is None else list(self.orders),
                "efficiency": efficiency,
                "label": self.label,
            }
        )
        return data

    @classmethod
    def _deserialize_init_data(cls, data):
        init_data = super()._deserialize_init_data(data)
        efficiency = init_data.get("efficiency")
        if efficiency is not None:
            init_data["efficiency"] = {
                int(m): value for m, value in efficiency.items() if value is not None
            }
        return init_data

    # -- orders ----------------------------------------------------------------

    @property
    def traced_order(self) -> Any:
        """The order that the model traces (``order``, else the geometry's)."""
        if self.order is not None:
            return self.order
        return self.geometry.grating_order

    def listed_orders(self) -> tuple[int, ...]:
        """Return the orders that branch enumeration lists."""
        if self.orders is not None:
            return self.orders
        return (int(be.to_numpy(self.geometry.grating_order)),)

    def branch_children(self, view: Any, select: Any = None) -> list[Any]:
        """Return the children of a branch step (the splitting protocol).

        Args:
            view: The surface view of the step (unused: the children of a
                grating do not depend on the direction of travel).
            select: The orders to list (the ``orders`` entry of
                ``BranchTracer``), overriding :meth:`listed_orders`.

        Returns:
            One ``BranchChild`` per order: side ``"F"`` for a reflective
            grating (the branch keeps the listed order), else ``"T"``.
        """
        from optiland.raytrace.branches import BranchChild

        side = "F" if self.is_reflective else "T"
        selected = self.listed_orders() if select is None else select
        return [BranchChild(side, (order_label(m),)) for m in selected]

    def configure_branch(self, child: Any, label: str) -> None:
        """Trace one child of :meth:`branch_children` (the splitting protocol).

        Sets :attr:`order` and the key entry ``(label, side, "m<±k>")`` that
        the model appends to ``rays.branch_key``.
        """
        if child.side not in ("T", "F") or len(child.entry) != 1:
            raise ValueError(f"{child!r} is not a child of a grating.")
        if (child.side == "F") != bool(self.is_reflective):
            raise ValueError(f"{child!r} does not match the grating side.")
        self.order = parse_order_label(child.entry[0])
        self._branch_entry = (label, child.side, *child.entry)

    def order_efficiency(
        self, order: int, wavelength: Any, theta_deg: Any, phi_deg: Any
    ) -> Any:
        """Return the efficiency of an order per ray (1 without ``efficiency``)."""
        if self.efficiency is None:
            return be.ones_like(wavelength)
        value = self.efficiency.get(int(order), 0.0)
        if callable(value):
            return be.array(value(wavelength, theta_deg, phi_deg)) + 0 * wavelength
        return float(value) + 0 * wavelength

    # -- tracing ----------------------------------------------------------------

    def interact_real_rays(self, rays: RealRays) -> RealRays:
        """Interact with real rays, causing diffraction.

        Args:
            rays (RealRays): The incoming real rays.

        Returns:
            RealRays: The outgoing real rays.
        """
        # find surface normals
        nx, ny, nz = self.geometry.surface_normal(rays)

        # Interact with surface (refract or reflect)
        n1 = self.material_pre.n(rays.w)
        n2 = self.material_post.n(rays.w)

        # find grating vector
        fx, fy, fz = self.geometry.grating_vector(rays)

        # grating period
        pp = self.geometry.grating_period

        # correct grating period considering projection effect on the surface
        pp = pp / be.sqrt(fx**2 + fy**2)

        # grating order
        m = self.traced_order

        k_in = (be.copy(rays.L), be.copy(rays.M), be.copy(rays.N))

        rays.gratingdiffract(nx, ny, nz, fx, fy, fz, m, pp, n1, n2, self.is_reflective)

        # Apply coating and BSDF
        rays = self._apply_coating_and_bsdf(rays, nx, ny, nz)

        if self.efficiency is not None:
            from optiland.raytrace.incidence import grating_incidence

            theta, phi = grating_incidence(k_in, (nx, ny, nz), (fx, fy, fz))
            eff = self.order_efficiency(int(be.to_numpy(m)), rays.w, theta, phi)
            if hasattr(rays, "flux_factor"):
                rays.flux_factor = rays.flux_factor * eff
            else:
                rays.i = rays.i * eff

        if self._branch_entry is not None and hasattr(rays, "branch_key"):
            rays.branch_key = (*rays.branch_key, self._branch_entry)

        return rays

    def interact_paraxial_rays(self, rays: ParaxialRays) -> ParaxialRays:
        """Interact with paraxial rays, causing diffraction.

        Args:
            rays (ParaxialRays): The incoming paraxial rays.

        Returns:
            ParaxialRays: The outgoing paraxial rays.
        """
        # grating period
        d = self.geometry.grating_period

        # grating order
        m = self.traced_order

        if self.is_reflective:
            # reflect (derived from paraxial equations when n'=-n)
            n = self.material_pre.n(rays.w)
            rays.u = -rays.u - 2 * n * rays.y / self.geometry.radius
            rays.u = rays.u + m * rays.w / d
        else:
            # surface power
            n1 = self.material_pre.n(rays.w)
            n2 = self.material_post.n(rays.w)
            power = (n2 - n1) / self.geometry.radius

            # refract
            rays.u = (n1 / n2) * rays.u - rays.y * power / n2 - m * rays.w / (d * n2)

        return rays


__all__ = [
    "DiffractiveInteractionModel",
    "Efficiency",
    "order_label",
    "parse_order_label",
]
