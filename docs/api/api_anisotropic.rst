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
  E_y = 0 and the flag ``degenerate``. In a lossless medium at a real K a
  propagating mode has a real q (``eig`` of the complex Δ of an optically
  active or a gyrotropic medium leaves |Im q| of about 1E-16).
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
  ``rays.branch_key`` has one entry per anisotropic surface:
  ``(label, "T")`` into an isotropic medium, ``(label, "T", mode)`` into an
  anisotropic one (``"R"`` for a reflection and ``"F"`` for a fold, below).
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

Folds inside crystals
---------------------

A mirror surface next to a tensor material is a fold inside the crystal: the
hypotenuse of a right-angle prism (total internal reflection), a roof or Porro
face, a metal or a perfect mirror on a crystal face. List it as any Optiland
mirror, with ``material="mirror"`` and ``interaction_type="anisotropic"``. The
crystal is then on both sides of the surface, and the next surface sees it as
its incident medium. The model reflects into one mode of the crystal (its
``mode``, default ``"slow"``) and takes the medium behind the face from
``far_material``: None for air (a bare face; below the critical angle part of
the power leaves through it), any scalar or tensor material (a metal), or
``"perfect_conductor"`` (tangential E = 0, r = -1 at normal incidence). The
branch key entry of a fold is ``(label, "F", mode)``: a reflection that keeps
the listed order of the surfaces, while ``"R"`` sends a branch back.

.. code-block:: python

   import math

   optic.surfaces.add(index=1, thickness=5.0, material=calcite, is_stop=True,
                      interaction_type="anisotropic")
   optic.surfaces.add(index=2, x=0.0, y=0.0, z=5.0, rx=math.pi / 4,
                      material="mirror", interaction_type="anisotropic")
   optic.surfaces.add(index=3, x=0.0, y=5.0, z=5.0, rx=-math.pi / 2,
                      interaction_type="anisotropic")
   optic.surfaces.add(index=4, x=0.0, y=10.0, z=5.0, rx=-math.pi / 2)
   optic.surfaces[2].interaction_model.far_material = None  # air: TIR

In :class:`~optiland.raytrace.branches.BranchTracer` a fold splits into its
reflected modes; the power of its transmitted children is the ledger field
``escaped``, and the absorption of a metal fold is in ``absorbed``.

Branch enumeration
------------------

One trace follows one mode per surface. :class:`~optiland.raytrace.branches.BranchTracer`
traces every branch: each transmitted mode of each anisotropic surface and, for
the surfaces in ``ghosts``, the reflected children as well (``"R"`` into an
isotropic medium, or the two modes of a crystal). A reflection sends the
branch back through the earlier surfaces (the views of
:mod:`optiland.sequences`); an isotropic ghost surface is split with the
bare-interface solver, so that R + T = 1. Branches below ``threshold`` times
the launch power (default 1E-6) or past ``max_reflections`` are pruned, and
their power is reported. The result maps each branch key to its rays and
keeps a power ledger (kept, returned, pruned, unfollowed, escaped,
evanescent, clipped, absorbed); for a lossless system the ledger adds to 1.

.. code-block:: python

   from optiland.raytrace.branches import BranchTracer

   result = BranchTracer(optic, ghosts="all").trace_all(num_rays=5)
   for key, branch in result.branches.items():
       print(key, branch.power, branch.terminal)
   result.ledger.as_dict()
   # Coherent sum of branches that reach the image (a Savart fringe):
   result.coherent_intensity(analyzer=(0.7071, 0.7071, 0.0))

``BranchTracer.sequence(key)`` returns a
:class:`~optiland.sequences.SequencedOptic` that traces one branch, for any
analysis of that branch.

Grating orders
--------------

A grating surface (``surface_type="grating"``) splits too: one branch per
listed diffraction order. ``DiffractiveInteractionModel.orders`` lists the
orders (default: the geometry's ``grating_order``), ``efficiency`` gives the
fraction of the incident power in each order (a number, or a callable of the
wavelength and the incidence (θ, φ) in the grating frame), and
``BranchTracer(orders={surface: (...)})`` selects orders per surface. The key
entry of an order is ``(label, "T", "m+1")`` through a transmission grating and
``(label, "F", "m+1")`` at a reflective grating, which keeps the listed order
of the surfaces as any mirror does. An evanescent order is booked as
``evanescent``, the orders that are not listed are in ``absorbed``. Without an
efficiency each order carries the power of its parent (as a sequential trace
does), so the ledger does not close.

.. code-block:: python

   grating = optic.surfaces[3].interaction_model
   grating.orders = (-1, 0, 1)
   grating.efficiency = {-1: 0.2, 0: 0.5, 1: 0.3}
   result = BranchTracer(optic).trace_all(num_rays=5)
   result[(("G", "T", "m+1"),)].power  # 0.3 of the launch

Any interaction model can split in the same way: it implements
:class:`~optiland.raytrace.branches.SplittingModel`, listing its children as
:class:`~optiland.raytrace.branches.BranchChild` (the side ``"T"``, ``"F"`` or
``"R"`` and the rest of the key entry) and tracing one child after
``configure_branch``. :func:`~optiland.raytrace.incidence.incidence_domain`
gives the range of (θ, φ, λ) that a scan of wavelengths, fields and
configurations puts on a grating, for an element that tabulates its orders.

.. autosummary::
   :toctree: anisotropic/
   :caption: Anisotropic Interface Modules

   anisotropic.frames
   anisotropic.eigenmodes
   anisotropic.interface
