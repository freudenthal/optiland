"""Interaction model for diffraction.

This module implements the DiffractiveInteractionModel class, which handles
ray interactions with surfaces through diffraction.

Kramer Harrison, 2025
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import optiland.backend as be
from optiland.interactions.refractive_reflective_model import (
    RefractiveReflectiveModel,
)

if TYPE_CHECKING:
    # pragma: no cover
    from optiland.rays import ParaxialRays, RealRays


class DiffractiveInteractionModel(RefractiveReflectiveModel):
    """Interaction model for diffraction."""

    interaction_type = "diffractive"

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

        # local grating period along the surface
        pp = self.geometry.grating_period / self._period_factor(fx, fy)

        # grating order
        m = self.geometry.grating_order

        rays.gratingdiffract(nx, ny, nz, fx, fy, fz, m, pp, n1, n2, self.is_reflective)

        # Apply coating and BSDF
        rays = self._apply_coating_and_bsdf(rays, nx, ny, nz)

        return rays

    def _period_factor(self, fx, fy):
        """Return the ratio of the grating period to the local period.

        The grooves are the intersections of the surface with the planes
        h . r = j d (j an integer, d the grating period), where
        h = (-sin(alpha), cos(alpha), 0) and alpha is the groove orientation
        angle. Along the surface the phase 2 pi (h . r) / d has the gradient
        (2 pi / d) (h - (h . n) n), so the local period is d / |h - (h . n) n|.
        The unit grating vector f lies along h - (h . n) n, so
        |h - (h . n) n| = h . f. On a plane this is 1.

        Args:
            fx (be.ndarray): The x-component of the unit grating vector.
            fy (be.ndarray): The y-component of the unit grating vector.

        Returns:
            be.ndarray: The factor |h - (h . n) n|.
        """
        alpha = self.geometry.groove_orientation_angle
        return be.abs(-be.sin(alpha) * fx + be.cos(alpha) * fy)

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
        m = self.geometry.grating_order

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
