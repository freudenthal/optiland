"""Straight-line propagation in a homogeneous anisotropic medium.

The default propagation model of the tensor materials of
``optiland.materials.anisotropic``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from optiland.propagation.base import BasePropagationModel

if TYPE_CHECKING:
    from optiland.materials.base import BaseMaterial
    from optiland.rays.real_rays import RealRays


class AnisotropicPropagation(BasePropagationModel):  # type: ignore[no-untyped-call]
    """Propagates rays along their ray direction in a tensor medium.

    The rays must carry a wave vector (``AnisotropicRays``). A step t moves a
    ray by t (L, M, N), the ray direction (the Poynting vector S), and
    multiplies its power by exp(-2 k0 Im(k) · Δr). The optical path Re(k) · Δr
    is added by the surface (``AnisotropicRays.optical_path``).

    Args:
        material: The parent tensor material.
    """

    def __init__(self, material: BaseMaterial | None = None):
        self.material = material

    def propagate(self, rays: RealRays, t: float) -> None:
        """Propagate the rays a distance t along their ray directions.

        Args:
            rays: The rays to propagate (``AnisotropicRays``).
            t: The distance to propagate, in mm.

        Raises:
            TypeError: If the rays carry no wave vector.
        """
        if not hasattr(rays, "optical_path"):
            raise TypeError(
                "Rays in a tensor material must carry a wave vector "
                "(AnisotropicRays). Set a polarization state of the optic."
            )
        r: Any = rays
        r.x = r.x + t * r.L
        r.y = r.y + t * r.M
        r.z = r.z + t * r.N
        r.i = r.i * r.attenuation(t)

    @classmethod
    def from_dict(
        cls, d: dict[str, Any], material: BaseMaterial | None = None
    ) -> AnisotropicPropagation:
        """Create the model from its dictionary representation.

        Args:
            d: The dictionary representation of the model.
            material: The parent material.

        Returns:
            AnisotropicPropagation: The model.
        """
        return cls(material=material)
