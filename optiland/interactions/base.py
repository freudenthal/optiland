"""Base Interaction Model

Defines the abstract base class for ray-surface interaction models.

Kramer Harrison, 2025
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

import numpy as np

import optiland.backend as be

if TYPE_CHECKING:
    # pragma: no cover
    from optiland.coatings import BaseCoating
    from optiland.rays import ParaxialRays, RealRays
    from optiland.scatter import BaseBSDF
    from optiland.surfaces import Surface


class BaseInteractionModel(ABC):
    """Abstract base class for ray-surface interaction models."""

    _registry = {}

    def __init__(
        self,
        parent_surface: Surface | None,
        is_reflective: bool,
        coating: BaseCoating | None = None,
        bsdf: BaseBSDF | None = None,
    ):
        self.parent_surface = parent_surface
        self.is_reflective = is_reflective
        self.coating = coating
        self.bsdf = bsdf

    @property
    def material_pre(self):
        return (
            self.parent_surface.material_post
            if self.parent_surface.previous_surface is None
            else self.parent_surface.previous_surface.material_post
        )

    @property
    def material_post(self):
        return self.parent_surface.material_post

    @property
    def geometry(self):
        return self.parent_surface.geometry

    def __init_subclass__(cls, **kwargs):
        """Automatically register subclasses."""
        super().__init_subclass__(**kwargs)
        BaseInteractionModel._registry[cls.__name__] = cls

    @abstractmethod
    def interact_real_rays(self, rays: RealRays) -> RealRays:
        """Interact with real rays."""
        pass  # pragma: no cover

    @abstractmethod
    def interact_paraxial_rays(self, rays: ParaxialRays) -> ParaxialRays:
        """Interact with paraxial rays."""
        pass  # pragma: no cover

    @abstractmethod
    def flip(self):
        """Flip the interaction model."""
        pass  # pragma: no cover

    def to_dict(self):
        """Returns a dictionary representation of the interaction model."""
        return {
            "type": self.__class__.__name__,
            "is_reflective": self.is_reflective,
            "coating": self.coating.to_dict() if self.coating else None,
            "bsdf": self.bsdf.to_dict() if self.bsdf else None,
        }

    @classmethod
    def _deserialize_init_data(cls, data):
        from optiland.coatings import BaseCoating
        from optiland.scatter import BaseBSDF

        init_data = data.copy()
        init_data.pop("type", None)
        init_data.pop("material_pre", None)

        if init_data.get("coating") is not None:
            init_data["coating"] = BaseCoating.from_dict(init_data["coating"])

        if init_data.get("bsdf") is not None:
            init_data["bsdf"] = BaseBSDF.from_dict(init_data["bsdf"])

        return init_data

    @classmethod
    def from_dict(cls, data, parent_surface):
        """Creates an interaction model from a dictionary representation."""
        interaction_type = data["type"]
        subclass = cls._registry.get(interaction_type)
        if subclass is None:
            raise ValueError(f"Unknown interaction model type: {interaction_type}")

        init_data = subclass._deserialize_init_data(data)

        return subclass(
            parent_surface=parent_surface,
            **init_data,
        )

    def _apply_coating_and_bsdf(
        self, rays: RealRays, nx: float, ny: float, nz: float
    ) -> RealRays:
        """Apply coating and BSDF to the rays."""
        if self.bsdf:
            rays = self.bsdf.scatter(rays, nx, ny, nz)

        if self.coating:
            rays = self._interact_coating(rays, nx, ny, nz)
        else:
            rays.update()
        return rays

    def _interact_coating(
        self, rays: RealRays, nx: float, ny: float, nz: float
    ) -> RealRays:
        """Apply the coating to the rays.

        The rays are in the local frame of the surface. A polarized coating
        also gets the rotation from the global frame to that local frame, so
        that an axis given in global coordinates is used as a global axis.
        """
        from optiland.coatings import BaseCoatingPolarized

        if isinstance(self.coating, BaseCoatingPolarized):
            return self.coating.interact(
                rays,
                reflect=self.is_reflective,
                nx=nx,
                ny=ny,
                nz=nz,
                rotation=self._local_rotation(),
            )
        return self.coating.interact(
            rays,
            reflect=self.is_reflective,
            nx=nx,
            ny=ny,
            nz=nz,
        )

    def _local_rotation(self) -> be.ndarray:
        """Return the rotation R from the global to the surface frame, (3, 3).

        v_local = R v_global. The columns of R are the global axes as the
        coordinate system of the surface localizes them.
        """
        from optiland.rays import RealRays

        eye = np.eye(3)
        zero = be.zeros((3,))
        probe = RealRays(
            zero,
            zero,
            zero,
            be.array(eye[0]),
            be.array(eye[1]),
            be.array(eye[2]),
            be.ones((3,)),
            be.ones((3,)),
        )
        self.geometry.localize(probe)
        return be.stack([probe.L, probe.M, probe.N], axis=0)
