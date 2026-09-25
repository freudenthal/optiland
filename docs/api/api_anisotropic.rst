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

Sequential tracing
------------------

A sequential system can contain plates, prisms and lenses of tensor materials.
Each surface with a tensor material on one or both sides uses
:class:`~optiland.interactions.anisotropic_model.AnisotropicInteractionModel`
(``interaction_type="anisotropic"``). The optic must have a polarization state:
the rays are then
:class:`~optiland.rays.anisotropic_rays.AnisotropicRays`, which carry the wave
vector k beside the ray direction.

* A ray moves along its ray direction (L, M, N) = S / |S|. Surface
  intersections use it.
* A step Δr adds the optical path Re(k) · Δr to the OPD (in a crystal
  n |Δr| cos ρ, ρ the walk-off) and multiplies the power by
  exp(-2 k0 Im(k) · Δr).
* At each anisotropic surface the rays follow one mode, the ``mode`` of the
  model: ``"o"`` or ``"e"`` into a uniaxial material, ``"slow"`` or ``"fast"``
  (the larger or the smaller index) into any tensor material, and ``"T"`` (the
  summed transmitted field) into an isotropic medium. The defaults are
  ``"slow"`` and ``"T"``. ``"t1"`` and ``"t2"`` are the s-like and the p-like
  child of the solver; these labels depend on the plane of incidence of each
  ray.
* The model multiplies the PRT matrix of the selected child into ``rays.p`` and
  the power factor |S · n̂| / |E|² of the child mode over that of the incident
  mode into ``rays.flux_factor``. The ray power is ``|P E|**2`` times this
  factor.
* ``rays.mode`` is the mode after the last anisotropic surface;
  ``rays.branch_key`` lists ``(surface label, mode)`` for each anisotropic
  surface.
* The paraxial trace uses the index of the mode of each medium along the
  local z axis.

The incident wave at a surface is a homogeneous wave along the real wave
normal of the ray (in an isotropic medium: along the ray direction), as in
:class:`~optiland.jones.JonesFresnel`. For lossless media the trace equals the
exact plane-wave solution of a plate. In an absorbing medium the field of the
solver is the exact (complex) field of the mode; the PRT then differs from the
isotropic path with ``JonesFresnel`` (real s and p vectors) by about the
extinction coefficient.

.. code-block:: python

   import numpy as np

   from optiland.materials import IdealMaterial, UniaxialMaterial
   from optiland.optic import Optic
   from optiland.rays import create_polarization

   c = np.sqrt(0.5)
   calcite = UniaxialMaterial(
       IdealMaterial(1.6583434042), IdealMaterial(1.4861300612), (c, 0.0, c)
   )
   optic = Optic()
   optic.surfaces.add(index=0, radius=np.inf, thickness=np.inf)
   optic.surfaces.add(
       index=1, thickness=2.0, material=calcite, is_stop=True,
       interaction_type="anisotropic",
   )
   optic.surfaces.add(index=2, thickness=5.0, interaction_type="anisotropic")
   optic.surfaces.add(index=3)
   optic.set_aperture(aperture_type="EPD", value=2.0)
   optic.fields.set_type(field_type="angle")
   optic.fields.add(y=0.0)
   optic.wavelengths.add(value=0.5893, is_primary=True)
   optic.updater.set_polarization(create_polarization("H"))
   optic.surfaces[1].interaction_model.mode = "e"
   rays = optic.trace(Hx=0, Hy=0, wavelength=0.5893, num_rays=5)
   rays.x  # the e rays walk off by -0.218 mm (2 mm of calcite cut at 45°)

.. autosummary::
   :toctree: anisotropic/
   :caption: Anisotropic Interface Modules

   anisotropic.frames
   anisotropic.eigenmodes
   anisotropic.interface
