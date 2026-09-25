"""Anisotropic Rays

This module contains the `AnisotropicRays` class: polarized rays that carry
their wave vector k beside their ray direction. In an anisotropic medium the
wave vector (the phase) and the ray direction (the time-averaged Poynting
vector S, the energy) are not parallel; they differ by the walk-off angle.

Conventions:

* The ray direction (L, M, N) is the unit S. Rays move along it; surface
  intersections use it.
* The wave vector k = (kx, ky, kz) is in units of k0 = 2π / λ and can be
  complex (an absorbing medium). In an isotropic medium k = (n + ik) d̂.
* A step Δr along the ray adds the optical path Re(k) · Δr (the OPD grows by
  it) and multiplies the power by exp(-2 k0 Im(k) · Δr).

The anisotropic interaction model (``optiland.interactions.anisotropic_model``)
sets k, the ray direction and the mode label at each surface next to a tensor
material. The isotropic refraction and reflection of ``RealRays`` also update k,
so that k stays the wave vector of the ray in every medium.

References:

* S. C. McClain, L. W. Hillman, R. A. Chipman, "Polarization ray tracing in
  anisotropic optically active media. I. Algorithms," J. Opt. Soc. Am. A 10,
  2371-2382 (1993), Eq. (60): the optical path Re(k) · Δr.
* W. S. T. Lam, "Anisotropic ray trace," PhD dissertation, University of
  Arizona (2015), Eq. (2.17).
"""

from __future__ import annotations

from typing import Any

import optiland.backend as be
from optiland.rays.polarized_rays import PolarizedRays

Array = Any

#: Conversion of a length in mm to µm (the wavelength unit).
_MM_TO_UM = 1e3

#: Ray state that some branches of Optiland add to ``PolarizedRays`` (``q``, the
#: geometrical transformation of the path); copied by ``from_rays`` if present.
_OPTIONAL_STATE = ("q",)


class AnisotropicRays(PolarizedRays):
    """Polarized rays that carry a wave vector beside the ray direction.

    Inherits from :class:`PolarizedRays`. The attributes L, M, N are the unit
    ray direction (the Poynting vector S).

    Args:
        x: The x-coordinates of the ray starting positions.
        y: The y-coordinates of the ray starting positions.
        z: The z-coordinates of the ray starting positions.
        L: The x-components of the ray directions.
        M: The y-components of the ray directions.
        N: The z-components of the ray directions.
        intensity: The intensity values of the rays.
        wavelength: The wavelength values of the rays in µm.
        k: The wave vectors in units of k0, shape (N, 3), complex allowed.
            Default: (L, M, N), a wave in vacuum.

    Attributes:
        kx (be.ndarray): x-components of the wave vectors (complex).
        ky (be.ndarray): y-components of the wave vectors (complex).
        kz (be.ndarray): z-components of the wave vectors (complex).
        mode (str | None): The label of the mode that the rays follow after
            the last anisotropic surface (for example ``"e"``), or None.
        branch_key (tuple): One entry ``(surface label, mode)`` for each
            anisotropic surface that the rays passed, in trace order.
    """

    def __init__(
        self,
        x: Array,
        y: Array,
        z: Array,
        L: Array,
        M: Array,
        N: Array,
        intensity: Array,
        wavelength: Array,
        k: Array | None = None,
    ) -> None:
        super().__init__(x, y, z, L, M, N, intensity, wavelength)  # type: ignore[no-untyped-call]
        if k is None:
            self.kx = be.to_complex(be.copy(self.L))
            self.ky = be.to_complex(be.copy(self.M))
            self.kz = be.to_complex(be.copy(self.N))
        else:
            self.set_k(k)
        self.mode: str | None = None
        self.branch_key: tuple[tuple[str, str], ...] = ()

    @classmethod
    def from_rays(cls, rays: PolarizedRays) -> AnisotropicRays:
        """Return anisotropic rays with the state of polarized rays.

        The positions, directions, intensities, wavelengths, OPD, PRT
        matrices and flux factors are copied. The wave vector is (L, M, N)
        (a wave in vacuum).

        Args:
            rays: The polarized rays to copy.

        Returns:
            AnisotropicRays: The new rays.
        """
        new = cls(
            be.copy(rays.x),
            be.copy(rays.y),
            be.copy(rays.z),
            be.copy(rays.L),
            be.copy(rays.M),
            be.copy(rays.N),
            be.copy(rays.i),
            be.copy(rays.w),
        )
        new.opd = be.copy(rays.opd)
        new.p = be.copy(rays.p)
        new.flux_factor = be.copy(rays.flux_factor)
        new._i0 = be.copy(rays._i0)
        new._L0 = be.copy(rays._L0)
        new._M0 = be.copy(rays._M0)
        new._N0 = be.copy(rays._N0)
        for name in _OPTIONAL_STATE:
            if hasattr(rays, name):
                setattr(new, name, be.copy(getattr(rays, name)))
        return new

    # -- wave vector --------------------------------------------------------

    @property
    def k(self) -> Array:
        """The wave vectors in units of k0, shape (N, 3), complex."""
        return be.concatenate(
            [self.kx[:, None], self.ky[:, None], self.kz[:, None]], axis=1
        )

    def set_k(self, k: Array) -> None:
        """Set the wave vectors.

        Args:
            k: Wave vectors in units of k0, shape (N, 3), complex allowed.
        """
        k = be.to_complex(k)
        self.kx = k[:, 0]
        self.ky = k[:, 1]
        self.kz = k[:, 2]

    def optical_path(self, t: Array) -> Array:
        """Return the optical path of a step t along the ray direction.

        The optical path is Re(k) · Δr with Δr = t (L, M, N), in the unit of
        t (McClain I, Eq. (60)). In a crystal it is n |Δr| cos ρ, with ρ the
        walk-off angle.

        Args:
            t: The step lengths along the ray, shape (N,) or a scalar.

        Returns:
            The optical paths, shape (N,).
        """
        return t * (
            be.real(self.kx) * self.L
            + be.real(self.ky) * self.M
            + be.real(self.kz) * self.N
        )

    def attenuation(self, t: Array) -> Array:
        """Return the power factor exp(-2 k0 Im(k) · Δr) of a step t in mm.

        Args:
            t: The step lengths along the ray in mm, shape (N,) or a scalar.

        Returns:
            The power factors, shape (N,).
        """
        im_k = (
            be.imag(self.kx) * self.L
            + be.imag(self.ky) * self.M
            + be.imag(self.kz) * self.N
        )
        k0 = 2 * be.pi / self.w
        return be.exp(-2 * k0 * im_k * t * _MM_TO_UM)

    # -- coordinate transforms ------------------------------------------------

    def rotate_x(self, rx: Any) -> None:
        """Rotate the rays and their wave vectors about the x-axis.

        Args:
            rx: Rotation angle around x-axis in radians.
        """
        super().rotate_x(rx)
        c, s = be.cos(be.array(rx)), be.sin(be.array(rx))
        self.ky, self.kz = self.ky * c - self.kz * s, self.ky * s + self.kz * c

    def rotate_y(self, ry: Any) -> None:
        """Rotate the rays and their wave vectors about the y-axis.

        Args:
            ry: Rotation angle around y-axis in radians.
        """
        super().rotate_y(ry)
        c, s = be.cos(be.array(ry)), be.sin(be.array(ry))
        self.kx, self.kz = self.kx * c + self.kz * s, -self.kx * s + self.kz * c

    def rotate_z(self, rz: Any) -> None:
        """Rotate the rays and their wave vectors about the z-axis.

        Args:
            rz: Rotation angle around z-axis in radians.
        """
        super().rotate_z(rz)
        c, s = be.cos(be.array(rz)), be.sin(be.array(rz))
        self.kx, self.ky = self.kx * c - self.ky * s, self.kx * s + self.ky * c

    # -- isotropic interactions -------------------------------------------------

    def _set_isotropic_k(self, n: Array) -> None:
        """Set k = n (L, M, N), the wave vector in an isotropic medium."""
        n = be.to_complex(be.atleast_1d(n) + 0 * self.L)
        self.kx = n * be.to_complex(self.L)
        self.ky = n * be.to_complex(self.M)
        self.kz = n * be.to_complex(self.N)

    def refract(self, nx: Any, ny: Any, nz: Any, n1: Any, n2: Any) -> None:
        """Refract the rays and set k = n2 (L, M, N).

        Args:
            nx: The x-component of the surface normals.
            ny: The y-component of the surface normals.
            nz: The z-component of the surface normals.
            n1: The refractive index before the surface.
            n2: The refractive index after the surface.
        """
        super().refract(nx, ny, nz, n1, n2)
        self._set_isotropic_k(n2)

    def reflect(self, nx: Any, ny: Any, nz: Any) -> None:
        """Reflect the rays and their wave vectors on the surface.

        The wave vector is mirrored: k - 2 (k · n̂) n̂.

        Args:
            nx: The x-component of the surface normal.
            ny: The y-component of the surface normal.
            nz: The z-component of the surface normal.
        """
        super().reflect(nx, ny, nz)
        dot = self.kx * nx + self.ky * ny + self.kz * nz
        self.kx = self.kx - 2 * dot * nx
        self.ky = self.ky - 2 * dot * ny
        self.kz = self.kz - 2 * dot * nz

    def gratingdiffract(
        self,
        nx: Any,
        ny: Any,
        nz: Any,
        fx: Any,
        fy: Any,
        fz: Any,
        m: Any,
        d: Any,
        n1: Any,
        n2: Any,
        is_reflective: bool,
    ) -> None:
        """Diffract the rays on a grating and set k = n (L, M, N).

        n is n1 for a reflective grating, else n2. See
        :meth:`RealRays.gratingdiffract` for the arguments.
        """
        super().gratingdiffract(nx, ny, nz, fx, fy, fz, m, d, n1, n2, is_reflective)
        self._set_isotropic_k(n1 if is_reflective else n2)
