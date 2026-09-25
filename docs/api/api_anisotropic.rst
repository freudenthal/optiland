Anisotropic Interfaces
======================

:mod:`optiland.anisotropic` computes the plane-wave eigenmodes of a homogeneous
medium with a 6 x 6 constitutive matrix [[ε, ξ], [ζ, μ]] (see
:mod:`optiland.materials.anisotropic`) and the four child modes of an interface
between two such media at any surface normal: two reflected and two
transmitted. It works per ray, on the NumPy and the Torch backend. Wave vectors
are in units of k0 = 2π/λ, the time dependence is exp(-iωt) and H' = η0 H.

* :func:`~optiland.anisotropic.medium_modes` gives the four eigenmodes of the
  Berreman matrix Δ for a tangential wave number K, sorted as forward s-like,
  forward p-like, backward s-like, backward p-like. A mode is forward if
  Im q > 1E-9 max(1, |q|), else if its power flux goes into +z; the sign of
  Re q is not used. Each mode is scaled to unit flux. A degenerate pair (an
  isotropic medium, an optic axis, a binormal) gets the basis E_x = 0 and
  E_y = 0 and the flag ``degenerate``.
* :func:`~optiland.anisotropic.plane_wave_modes` gives the two modes along a
  wave normal (the incident eigenmode of a crystal).
* :func:`~optiland.anisotropic.solve_interface` gives, for each child, the wave
  vector, the fields E and H', the Poynting vector, the ray direction, the
  walk-off, the power fraction, the flags ``evanescent`` and ``degenerate``,
  and a 3 x 3 PRT matrix P_j (P_j E_in = the child field, P_j k̂_in = k̂_j,
  det P_j = 0). ``reflectance`` and ``transmittance`` are the fluxes of the
  summed reflected and transmitted fields.
* :func:`~optiland.anisotropic.constitutive_matrix` gives the 6 x 6 matrix of a
  scalar or a tensor material.

.. code-block:: python

   import numpy as np

   from optiland.anisotropic import constitutive_matrix, solve_interface
   from optiland.materials import IdealMaterial, Material, UniaxialMaterial

   calcite = UniaxialMaterial(
       Material("CaCO3", reference="Ghosh-o"),
       Material("CaCO3", reference="Ghosh-e"),
       optic_axis=(1.0, 0.0, 1.0),
   )
   air = constitutive_matrix(IdealMaterial(n=1.0), 0.5893)
   crystal = constitutive_matrix(calcite, 0.5893)
   result = solve_interface(
       normal=[0.0, 0.0, 1.0],
       medium_a=air,
       medium_b=crystal,
       k_in=[0.0, 0.0, 1.0],
       E_in=[1.0, 0.0, 0.0],
   )
   result.power  # (1, 4): r1, r2, t1, t2
   result.ray  # (1, 4, 3): the e child walks off by 6.2°

.. autosummary::
   :toctree: anisotropic/
   :caption: Anisotropic Interface Modules

   anisotropic.frames
   anisotropic.eigenmodes
   anisotropic.interface
