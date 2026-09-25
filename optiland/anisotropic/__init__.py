"""Anisotropic and Bianisotropic Interfaces

This package computes the plane-wave eigenmodes of a homogeneous medium with a
6 x 6 constitutive matrix (``optiland.materials.anisotropic``) and the four
child modes (two reflected, two transmitted) at an interface between two such
media at any surface normal. Both backends (NumPy and Torch) are supported.

Modules:

* ``frames``: the interface frame and the rotation of the 6 x 6 matrices.
* ``eigenmodes``: the Berreman matrix, the sorted and flux-scaled eigenmodes,
  and the plane-wave modes along a wave normal.
* ``interface``: the boundary matching, the child modes, their PRT matrices
  and the energy report.
"""

from __future__ import annotations

from optiland.anisotropic.eigenmodes import (
    Eigenmodes,
    PlaneWaveModes,
    berreman_delta,
    medium_modes,
    plane_wave_modes,
)
from optiland.anisotropic.frames import (
    interface_frame,
    rotate_constitutive,
    to_global,
)
from optiland.anisotropic.interface import (
    CHILD_LABELS,
    InterfaceResult,
    constitutive_matrix,
    solve_interface,
)

__all__ = [
    "CHILD_LABELS",
    "Eigenmodes",
    "InterfaceResult",
    "PlaneWaveModes",
    "berreman_delta",
    "constitutive_matrix",
    "interface_frame",
    "medium_modes",
    "plane_wave_modes",
    "rotate_constitutive",
    "solve_interface",
    "to_global",
]
