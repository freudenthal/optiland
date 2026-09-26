"""Interaction model for surfaces next to anisotropic media

This module implements the AnisotropicInteractionModel class. It refracts rays
at a surface that has a tensor material (``optiland.materials.anisotropic``)
on one or both sides. The eigenmode interface solver
(``optiland.anisotropic.solve_interface``) gives the four child modes of each
ray; the model follows one of them (the mode of the surface) or, into an
isotropic medium, the transmitted field.

A ray carries its wave vector k beside its ray direction (``AnisotropicRays``).
At the surface:

1. The incident wave. In an isotropic medium A it is the homogeneous wave
   k = (n + ik) d̂ along the ray direction d̂. In an anisotropic medium A it is
   the plane-wave mode of A along the real wave normal of the ray that is
   nearest to the carried k (``plane_wave_modes``).
2. The solver gives the children (r1, r2, t1, t2) with their wave vectors,
   ray directions, mode fields and PRT matrices.
3. The model selects the child of its mode: a transmitted child, or a
   reflected child when ``is_reflective`` is set (a reflecting step of a
   ``SurfaceView``: the ghost and total-internal-reflection branches of
   ``optiland.raytrace.branches``). It sets the ray direction to the child
   ray S / |S| (into an isotropic medium: the unit Re k), sets k to the child
   wave vector, multiplies the child PRT matrix into ``rays.p`` and the power
   factor into ``rays.flux_factor``.

Modes (the ``mode`` argument):

* ``"T"``: the transmitted field into an isotropic medium B, the sum of the
  two transmitted children. Their PRT matrices recombine as
  P = P_t1 + P_t2 - k̂_out w₃ (w₃ the k̂ row of O_in⁻¹; Yun, Crabtree and
  Chipman 2011, Eq. (38)). The default into an isotropic medium.
* ``"R"``: the reflected field into an isotropic medium A, the sum of the two
  reflected children (the same recombination with the reflected k̂). The
  default of a reflecting step in an isotropic medium.
* ``"slow"``, ``"fast"``: the mode of the larger or the smaller index
  Re √(k · k) (the outer or the inner sheet of the index surface) of the exit
  medium (B, or A for a reflection). ``"slow"`` is the default into an
  anisotropic medium.
* ``"o"``, ``"e"``: the ordinary or the extraordinary mode of a
  ``UniaxialMaterial`` exit medium (the ordinary mode has k · k = ε_o).
* ``"t1"``, ``"t2"``: the s-like or the p-like child of the solver (r1, r2
  for a reflection). These labels depend on the plane of incidence of each ray; the
  same label can be a different physical mode for two rays of a fan.

The power factor of the surface is g_out / g_in, with g = |S · n̂| / |E|² of
the incident mode and of the selected child mode (unit amplitude). The ray
power is then ``|P E|**2 * flux_factor`` (as for ``JonesFresnel``, whose factor
Re(n' cos θ') / Re(n cos θ) is this ratio for an isotropic interface).

The paraxial trace uses the index of the mode of each medium along the local
z axis (an approximation for the first-order properties only).

References:

* S. C. McClain, L. W. Hillman, R. A. Chipman, "Polarization ray tracing in
  anisotropic optically active media. I. Algorithms," J. Opt. Soc. Am. A 10,
  2371-2382 (1993).
* G. Yun, K. Crabtree, R. A. Chipman, "Three-dimensional polarization
  ray-tracing calculus I: definition and diattenuation," Appl. Opt. 50,
  2855-2865 (2011).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

import optiland.backend as be
from optiland.anisotropic.eigenmodes import plane_wave_modes
from optiland.anisotropic.frames import rotate_constitutive, to_global
from optiland.anisotropic.interface import constitutive_matrix, solve_interface
from optiland.interactions.base import BaseInteractionModel
from optiland.materials.anisotropic import BaseTensorMaterial, UniaxialMaterial
from optiland.rays.anisotropic_rays import AnisotropicRays
from optiland.rays.polarized_rays import PolarizedRays
from optiland.rays.real_rays import RealRays

if TYPE_CHECKING:
    from optiland.coatings import BaseCoating
    from optiland.rays import ParaxialRays
    from optiland.scatter import BaseBSDF
    from optiland.surfaces import Surface

Array = Any

#: The mode labels of :class:`AnisotropicInteractionModel`.
MODES = ("T", "R", "slow", "fast", "o", "e", "t1", "t2")

_R1 = 0  # the indices of the children r1, r2, t1, t2 on the child axis
_R2 = 1
_T1 = 2
_T2 = 3


def _vectors(x: Array, y: Array, z: Array, n: int) -> Array:
    """Return the real vectors (x, y, z) as an array of shape (n, 3)."""
    zero = be.zeros((n,))
    return be.stack([x + zero, y + zero, z + zero], axis=1)


def _dot(a: Array, b: Array) -> Array:
    """Return the row dot products of a and b (no conjugate), shape (N,)."""
    return a[:, 0] * b[:, 0] + a[:, 1] * b[:, 1] + a[:, 2] * b[:, 2]


def _unit(v: Array) -> Array:
    """Return the unit real vectors of v, shape (N, 3)."""
    return v / be.sqrt(_dot(v, v))[:, None]


def _norm2(v: Array) -> Array:
    """Return |v|² of complex vectors, shape (N,)."""
    return be.real(_dot(v, be.real(v) - 1j * be.imag(v)))


def _flux_density(e: Array, h: Array, normal: Array) -> Array:
    """Return g = |S · n̂| / |E|² of mode fields E, H', shape (N,)."""
    h_conj = be.real(h) - 1j * be.imag(h)
    s = 0.5 * be.real(be.cross(e, h_conj))
    return be.abs(_dot(s, normal)) / _norm2(e)


def _transverse(d: Array) -> Array:
    """Return a unit real vector normal to each unit vector d, shape (N, 3)."""
    n = d.shape[0]
    x_axis = _vectors(1.0, 0.0, 0.0, n)
    y_axis = _vectors(0.0, 1.0, 0.0, n)
    ref = be.where((be.abs(d[:, 0]) > 0.9)[:, None], y_axis, x_axis)
    return _unit(be.cross(d, ref))


def _outer(a: Array, b: Array) -> Array:
    """Return the outer products a bᵀ of rows, shape (N, 3, 3)."""
    return a[:, :, None] * b[:, None, :]


def _pick(second: Array, a: Array, b: Array) -> Array:
    """Return b where ``second`` is True, else a (rows of any shape)."""
    mask = be.reshape(second, (-1,) + (1,) * (len(a.shape) - 1))
    return be.where(mask, b, a)


def _index(k: Array) -> Array:
    """Return the index Re √(k · k) of wave vectors k, shape (N,)."""
    return be.real(be.sqrt(_dot(k, k)))


def _ordinary_epsilon(material: Any, wavelength: Array) -> Array:
    """Return ε_o = (n_o + ik_o)² of a uniaxial material, shape (N,)."""
    n = be.to_complex(be.atleast_1d(material.ordinary.n(wavelength)))
    k = be.to_complex(be.atleast_1d(material.ordinary.k(wavelength)))
    return (n + 1j * k) ** 2


class AnisotropicInteractionModel(BaseInteractionModel):  # type: ignore[no-untyped-call]
    """Interaction model for a surface next to one or two tensor materials.

    Args:
        parent_surface: The surface of the model.
        is_reflective: Must be False at construction (a mirror next to a
            tensor material is not supported). A reflecting ``SurfaceView``
            sets it on its own copy of the model; the model then follows a
            reflected child and takes both media from the view's
            ``interface_materials``.
        coating: Must be None. The interface solver gives the bare-interface
            physics; coatings on anisotropic surfaces are not supported.
        bsdf: Must be None.
        mode: The mode that the rays follow after the surface (``MODES``).
            None selects ``"T"`` (``"R"`` for a reflection) into an isotropic
            medium and ``"slow"`` into an anisotropic medium.
        label: The surface label in the branch key of the rays. None uses the
            surface comment.
    """

    interaction_type = "anisotropic"

    def __init__(
        self,
        parent_surface: Surface | None,
        is_reflective: bool = False,
        coating: BaseCoating | None = None,
        bsdf: BaseBSDF | None = None,
        mode: str | None = None,
        label: str | None = None,
    ):
        if is_reflective:
            raise NotImplementedError(
                "A reflective surface next to a tensor material is not supported."
            )
        if mode is not None and mode not in MODES:
            raise ValueError(f"Unknown mode {mode!r}; the modes are {MODES}.")
        super().__init__(parent_surface, is_reflective, coating, bsdf)
        self.mode = mode
        self.label = label

    def to_dict(self) -> dict[str, Any]:
        """Returns a dictionary representation of the model."""
        data: dict[str, Any] = super().to_dict()  # type: ignore[no-untyped-call]
        data.update({"mode": self.mode, "label": self.label})
        return data

    def flip(self) -> None:
        """Flip the interaction model (no state depends on the direction)."""

    # -- helpers --------------------------------------------------------------

    def _branch_label(self) -> str:
        """Return the surface label for the branch key."""
        if self.label is not None:
            return self.label
        surface = self.parent_surface
        return getattr(surface, "comment", "") if surface is not None else ""

    def _interface_media(self) -> tuple[Any, Any]:
        """Return the media (A, B) on the incident and the far side.

        A ``SurfaceView`` gives them by ``interface_materials`` (for a
        reflecting view, ``material_post`` is the incident medium).
        """
        media = getattr(self.parent_surface, "interface_materials", None)
        if media is not None:
            return media[0], media[1]
        if self.is_reflective:
            raise NotImplementedError(
                "A reflective surface next to a tensor material is supported "
                "only as a reflecting SurfaceView (branch tracing)."
            )
        return self.material_pre, self.material_post

    def _resolve_mode(self, isotropic_exit: bool, exit_material: Any) -> str:
        """Return the mode of the surface and check it against the exit medium.

        The exit medium is B for a transmission and A for a reflection.
        """
        summed = "R" if self.is_reflective else "T"
        mode = self.mode
        if mode is None:
            return summed if isotropic_exit else "slow"
        if mode in ("T", "R") and mode != summed:
            step = "reflecting" if self.is_reflective else "transmitting"
            raise ValueError(
                f"Mode {mode!r} does not fit a {step} step; use {summed!r}."
            )
        if mode == summed and not isotropic_exit:
            raise ValueError(
                f"Mode {mode!r} needs an isotropic exit medium: the two modes "
                "of an anisotropic medium have different directions. Select "
                "one of them."
            )
        if mode != summed and isotropic_exit:
            raise ValueError(
                f"Mode {mode!r} selects one mode, but the exit medium is "
                f"isotropic. Use mode {summed!r}."
            )
        if mode in ("o", "e") and not isinstance(exit_material, UniaxialMaterial):
            raise ValueError(
                f"Mode {mode!r} needs a UniaxialMaterial exit medium; "
                "use 'slow' or 'fast'."
            )
        return mode

    @staticmethod
    def _select(mode: str, k1: Array, k2: Array, material: Any, w: Array) -> Array:
        """Return True where the mode is the second of two modes, shape (N,).

        Args:
            mode: A mode label other than "T".
            k1: Wave vectors of the first mode, shape (N, 3).
            k2: Wave vectors of the second mode, shape (N, 3).
            material: The material of the two modes.
            w: The wavelengths in µm, shape (N,).
        """
        if mode in ("t1", "t2"):
            second = be.zeros((k1.shape[0],)) + (1.0 if mode == "t2" else 0.0)
            return second > 0.5
        if mode in ("slow", "fast"):
            n1, n2 = _index(k1), _index(k2)
            return n2 > n1 if mode == "slow" else n2 < n1
        eps_o = _ordinary_epsilon(material, w)
        d1 = be.abs(_dot(k1, k1) - eps_o)
        d2 = be.abs(_dot(k2, k2) - eps_o)
        return d2 < d1 if mode == "o" else d2 > d1

    def _local_rotation(self) -> Array:
        """Return the rotation R from the global to the surface frame, (3, 3).

        v_local = R v_global. The columns of R are the global axes as the
        coordinate system of the surface localizes them.
        """
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

    @staticmethod
    def _incident(
        rays: AnisotropicRays, d: Array, normal: Array, m_a: Array, material_a: Any
    ) -> tuple[Array, Array, Array | None]:
        """Return the incident k, E and flux density g_in (None: isotropic A)."""
        if not isinstance(material_a, BaseTensorMaterial):
            n_c = be.sqrt(m_a[:, 0, 0])
            k_in = n_c[:, None] * be.to_complex(d)
            return k_in, be.to_complex(_transverse(d)), None
        k_ray = rays.k
        finite = be.isfinite(be.real(_dot(k_ray, k_ray)))
        k_ray = be.where(finite[:, None], k_ray, be.to_complex(d))
        wave = _unit(be.real(k_ray))
        modes = plane_wave_modes(wave, m_a)
        k0, k1 = modes.k[:, 0, :], modes.k[:, 1, :]
        second = _norm2(k1 - k_ray) < _norm2(k0 - k_ray)
        k_in = _pick(second, k0, k1)
        e_in = _pick(second, modes.E[:, 0, :], modes.E[:, 1, :])
        h_in = _pick(second, modes.H[:, 0, :], modes.H[:, 1, :])
        return k_in, e_in, _flux_density(e_in, h_in, normal)

    # -- real rays --------------------------------------------------------------

    def interact_real_rays(self, rays: RealRays) -> RealRays:
        """Refract (or, for a reflecting view, reflect) the rays into the mode.

        Args:
            rays (RealRays): The incoming rays; they must be
                ``AnisotropicRays``.

        Returns:
            RealRays: The outgoing rays.

        Raises:
            TypeError: If the rays are not ``AnisotropicRays``.
            ValueError: If a coating or a BSDF is set, or if the mode does not
                fit the medium after the surface.
        """
        if not isinstance(rays, AnisotropicRays):
            raise TypeError(
                "An anisotropic surface needs AnisotropicRays. Set a "
                "polarization state of the optic."
            )
        if self.coating is not None or self.bsdf is not None:
            raise ValueError(
                "Coatings and BSDFs are not supported on anisotropic surfaces."
            )
        n_rays = be.size(rays.x)
        nx, ny, nz = self.geometry.surface_normal(rays)
        d = _vectors(rays.L, rays.M, rays.N, n_rays)
        normal = _vectors(nx, ny, nz, n_rays)

        # Rays that are lost (NaN, for example after a total internal
        # reflection) go through the solver as a dummy ray and stay NaN.
        finite = be.isfinite(_dot(d, d) + _dot(normal, normal))
        z_axis = _vectors(0.0, 0.0, 1.0, n_rays)
        d = be.where(finite[:, None], d, z_axis)
        normal = be.where(finite[:, None], normal, z_axis)
        # The normal points from A into B: along the incident ray direction.
        cos_in = _dot(d, normal)
        sign = be.where(cos_in < 0, -1.0 + 0 * cos_in, 1.0 + 0 * cos_in)
        normal = normal * sign[:, None]

        # The rays are in the local frame of the surface; the tensors are in
        # the global frame.
        w = be.atleast_1d(rays.w) + be.zeros((n_rays,))
        rotation = self._local_rotation()[None]
        material_a, material_b = self._interface_media()
        m_a = rotate_constitutive(constitutive_matrix(material_a, w), rotation)
        m_b = rotate_constitutive(constitutive_matrix(material_b, w), rotation)
        k_in, e_in, g_in = self._incident(rays, d, normal, m_a, material_a)
        result = solve_interface(normal, m_a, m_b, k_in, e_in)

        reflect = bool(self.is_reflective)
        exit_modes = result.modes_a if reflect else result.modes_b
        exit_material = material_a if reflect else material_b
        isotropic_exit = exit_modes.isotropic
        mode = self._resolve_mode(bool(be.all(isotropic_exit)), exit_material)
        if g_in is None:  # isotropic A: the s-like forward mode of A
            g_in = be.abs(be.real(result.modes_a.S[:, 0, 2])) / _norm2(
                result.modes_a.E[:, 0, :]
            )

        c1, c2 = (_R1, _R2) if reflect else (_T1, _T2)
        k_c1, k_c2 = result.k[:, c1, :], result.k[:, c2, :]
        summed = mode in ("T", "R")
        if summed:
            second = be.zeros((n_rays,)) > 1.0
        else:
            second = self._select(mode, k_c1, k_c2, exit_material, w)
        k_out = _pick(second, k_c1, k_c2)
        prt = _pick(second, result.prt[:, c1], result.prt[:, c2])
        e_mode = _pick(second, result.E_mode[:, c1], result.E_mode[:, c2])
        h_mode = _pick(second, result.H_mode[:, c1], result.H_mode[:, c2])
        ray = _pick(second, result.ray[:, c1], result.ray[:, c2])
        if summed:
            # The two children of an isotropic exit medium leave together
            # (Yun I, Eq. (38)).
            e_forward = to_global(result.modes_a.E[:, :2, :], result.rotation)
            k_hat_in = be.to_complex(_unit(be.real(k_in)))
            o_in = be.concatenate(
                [be.transpose(e_forward, (0, 2, 1)), k_hat_in[:, :, None]], axis=-1
            )
            w3 = be.linalg.inv(o_in)[:, 2, :]
            k_hat_out = be.to_complex(_unit(be.real(k_out)))
            prt = result.prt[:, c1] + result.prt[:, c2] - _outer(k_hat_out, w3)
        # Into an isotropic medium the ray is along Re k.
        ray = be.where(isotropic_exit[:, None], _unit(be.real(k_out)), ray)
        g_out = _flux_density(e_mode, h_mode, normal)

        # A child that decays faster along n̂ than it propagates is not a ray
        # (total internal reflection; with a small loss in A the child carries
        # a small power at grazing exit). Such rays are lost (NaN), as in
        # ``RealRays.refract``.
        evanescent = _pick(second, result.evanescent[:, c1], result.evanescent[:, c2])
        q_out = _dot(k_out, be.to_complex(normal))
        lost = evanescent | (be.abs(be.imag(q_out)) >= be.abs(be.real(q_out)))
        nan = be.zeros_like(ray) + float("nan")
        ray = be.where((finite & ~lost)[:, None], ray, nan)
        k_out = be.where(finite[:, None], k_out, be.to_complex(nan))

        rays.L0 = be.copy(rays.L)
        rays.M0 = be.copy(rays.M)
        rays.N0 = be.copy(rays.N)
        rays.L = ray[:, 0]
        rays.M = ray[:, 1]
        rays.N = ray[:, 2]
        rays.is_normalized = True
        rays.set_k(k_out)

        rays.p = be.matmul(prt, be.to_complex(rays.p))
        rays.flux_factor = rays.flux_factor * g_out / g_in
        if hasattr(rays, "q"):
            k0 = _unit(be.real(k_in))
            k1 = _unit(be.real(k_out))
            *_, o_in_q, o_out_q = PolarizedRays.get_local_basis(k0, k1)
            rays.q = be.matmul(be.matmul(o_out_q, o_in_q), rays.q)
        rays.mode = mode
        rays.branch_key = (*rays.branch_key, self._key_entry(mode))
        return rays

    def _key_entry(self, mode: str) -> tuple[str, ...]:
        """Return the branch-key entry of this step for a resolved mode.

        ``(label, "T")`` or ``(label, "R")`` into an isotropic exit medium;
        ``(label, "T", mode)`` or ``(label, "R", mode)`` into an anisotropic
        one.
        """
        side = "R" if self.is_reflective else "T"
        if mode in ("T", "R"):
            return (self._branch_label(), side)
        return (self._branch_label(), side, mode)

    # -- paraxial rays ------------------------------------------------------------

    def paraxial_index(self, wavelength: Any) -> Array:
        """Return the index of the medium after the surface for paraxial use.

        A scalar material gives n. A tensor material gives the real index of
        the mode of the surface along the local z axis (``"T"`` and None: the
        first mode if the medium is isotropic along z, else the slow mode; the
        ``"o"`` and ``"e"`` modes of a material that is not uniaxial: the slow
        mode). This is an approximation for the first-order properties only.

        Args:
            wavelength: The wavelength in µm, a scalar or shape (N,).

        Returns:
            The index, a scalar array for a scalar wavelength, else shape (N,).
        """
        return self._paraxial_index(self.material_post, self.mode, wavelength)

    def _paraxial_index(self, material: Any, mode: str | None, w: Any) -> Array:
        """Return the paraxial index of a material for a mode (see above)."""
        if not isinstance(material, BaseTensorMaterial):
            return material.n(w)
        wl = be.atleast_1d(be.asarray(w))
        matrix = constitutive_matrix(material, wl)
        n_rows = matrix.shape[0]
        modes = plane_wave_modes(_vectors(0.0, 0.0, 1.0, n_rows), matrix)
        k1, k2 = modes.k[:, 0, :], modes.k[:, 1, :]
        if mode is None or mode == "T":
            mode = "t1" if bool(be.all(modes.degenerate)) else "slow"
        if mode in ("o", "e") and not isinstance(material, UniaxialMaterial):
            mode = "slow"
        second = self._select(mode, k1, k2, material, wl)
        index = _index(_pick(second, k1, k2))
        return index[0] if be.size(index) == 1 else index

    def interact_paraxial_rays(self, rays: ParaxialRays) -> ParaxialRays:
        """Refract paraxial rays with the index of the mode of each medium.

        Args:
            rays (ParaxialRays): The incoming paraxial rays.

        Returns:
            ParaxialRays: The outgoing paraxial rays.
        """
        surface = self.parent_surface
        previous = surface.previous_surface if surface is not None else None
        pre_model = getattr(previous, "interaction_model", None)
        if isinstance(pre_model, AnisotropicInteractionModel):
            n1 = pre_model.paraxial_index(rays.w)
        else:
            n1 = self.material_pre.n(rays.w)
        n2 = self.paraxial_index(rays.w)
        power = (n2 - n1) / self.geometry.radius
        rays.u = 1 / n2 * (n1 * rays.u - rays.y * power)
        return rays
