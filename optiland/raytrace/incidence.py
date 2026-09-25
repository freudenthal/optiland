"""Incidence of rays on a grating, and the incidence domain of a scan

A grating element that tabulates its order amplitudes (for example an RCWA
solver on a grid) needs the range of incidence that a planned use puts on it.
This module gives the incidence of each ray in the **grating frame** and
:func:`incidence_domain`, the range of (θ, φ, λ) that a scan of wavelengths,
fields and configurations (for example grating tilts) puts on one surface.

**Grating frame.** At the point where a ray meets the grating, with surface
normal n̂, grating vector ĝ and incident direction k̂:

* z_g = ±n̂, the sign chosen so that z_g · k̂ > 0 (the ray travels along +z_g);
* x_g = the tangential part of ĝ, normalized: (ĝ − (ĝ · z_g) z_g)/|…|;
* y_g = z_g × x_g.

The polar angle θ is measured from +z_g (0 ≤ θ < 90°) and the azimuth φ from
+x_g towards +y_g to the tangential part of k̂ (−180° < φ ≤ 180°; φ = 0 at
normal incidence). In this frame the grating equation reads
k_x,m = n sin θ cos φ + m λ/Λ, k_y,m = n sin θ sin φ, the form of a planar
RCWA solver whose grating vector is its +x axis with the incident medium on
the −z side.

The tangential wave vector k_∥ = n sin θ (cos φ, sin φ) (units of k0, n the
index of the incident medium) is the natural key of a table of amplitudes;
the domain reports its range too, which has no wrap-around at φ = ±180°.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import optiland.backend as be
from optiland.rays.real_rays import RealRays

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

Array = Any
Vector = tuple[Array, Array, Array]

#: Tangential parts below this norm (of a unit vector) count as normal incidence.
_NORMAL_INCIDENCE = 1e-12


def _dot(a: Vector, b: Vector) -> Array:
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]


def _unit(a: Vector) -> Vector:
    norm = be.sqrt(_dot(a, a))
    return (a[0] / norm, a[1] / norm, a[2] / norm)


def grating_frame(
    k: Vector, normal: Vector, grating_vector: Vector
) -> tuple[Vector, Vector, Vector]:
    """Return the grating frame (x_g, y_g, z_g) of rays, each a 3-tuple of arrays.

    Args:
        k: The incident ray directions (L, M, N).
        normal: The surface normals at the ray points (either sign).
        grating_vector: The grating vectors at the ray points (a normal part
            is removed).

    Raises:
        ValueError: If a grating vector is parallel to the normal.
    """
    n = _unit(normal)
    sign = be.where(_dot(k, n) < 0, -1.0, 1.0)
    z = (n[0] * sign, n[1] * sign, n[2] * sign)
    g_n = _dot(grating_vector, z)
    x = tuple(g - g_n * zc for g, zc in zip(grating_vector, z, strict=True))
    if be.any(be.sqrt(_dot(x, x)) < _NORMAL_INCIDENCE):
        raise ValueError("A grating vector is parallel to the surface normal.")
    x = _unit(x)
    y = (
        z[1] * x[2] - z[2] * x[1],
        z[2] * x[0] - z[0] * x[2],
        z[0] * x[1] - z[1] * x[0],
    )
    return x, y, z


def grating_incidence(
    k: Vector, normal: Vector, grating_vector: Vector
) -> tuple[Array, Array]:
    """Return the incidence (θ, φ) of rays in the grating frame, in degrees.

    Args:
        k: The incident ray directions (L, M, N).
        normal: The surface normals at the ray points (either sign).
        grating_vector: The grating vectors at the ray points.

    Returns:
        (theta_deg, phi_deg): θ from +z_g in [0, 90), φ from +x_g in
        (−180, 180], 0 at normal incidence.
    """
    x, y, z = grating_frame(k, normal, grating_vector)
    k = _unit(k)
    kx, ky, kz = _dot(k, x), _dot(k, y), _dot(k, z)
    kt = be.sqrt(kx**2 + ky**2)
    theta = be.rad2deg(be.arctan2(kt, kz))
    normal_incidence = kt < _NORMAL_INCIDENCE
    phi = be.rad2deg(
        be.arctan2(
            be.where(normal_incidence, 0.0, ky), be.where(normal_incidence, 1.0, kx)
        )
    )
    return theta, phi


# ----------------------------------------------------------------- the domain


@dataclass
class Scan:
    """A planned use of an optic: wavelengths × fields × configurations.

    Attributes:
        wavelengths: The wavelengths in µm.
        fields: The normalized fields (Hx, Hy).
        configurations: Callables that take the optic (or sequence) and return
            the object to trace for one configuration, for example a copy
            with the grating tilted. None traces the optic as it is.
        num_rays: The number of rays of the pupil distribution.
        distribution: The pupil distribution (as ``Optic.trace``).
    """

    wavelengths: Sequence[float]
    fields: Sequence[tuple[float, float]] = ((0.0, 0.0),)
    configurations: Sequence[Callable[[Any], Any]] | None = None
    num_rays: int = 16
    distribution: Any = "hexapolar"


@dataclass
class IncidenceDomain:
    """The incidence that a scan puts on a grating surface.

    Ranges are ``(low, high)`` with the margins applied. The φ range is the
    shortest arc that holds every sample; ``low`` > ``high`` never happens,
    but ``high`` can exceed 180° when the arc crosses ±180°.

    Attributes:
        theta_deg: The range of θ (degrees).
        phi_deg: The range of φ (degrees).
        wavelength: The range of λ (µm).
        k_parallel: The ranges of k_∥,x and k_∥,y (units of k0).
        samples: The traced samples: ``theta_deg``, ``phi_deg``,
            ``wavelength``, ``kx``, ``ky`` (NumPy arrays).
        angle_margin_deg: The margin added to each angle range.
        wavelength_margin: The margin added to the λ range (µm).
    """

    theta_deg: tuple[float, float]
    phi_deg: tuple[float, float]
    wavelength: tuple[float, float]
    k_parallel: tuple[tuple[float, float], tuple[float, float]]
    samples: dict[str, Any] = field(default_factory=dict)
    angle_margin_deg: float = 0.0
    wavelength_margin: float = 0.0

    def contains(self, theta_deg: Any, phi_deg: Any, wavelength: Any) -> Any:
        """Return True where (θ, φ, λ) lies in the domain (NumPy bool array)."""
        import numpy as np

        theta = np.asarray(be.to_numpy(theta_deg), dtype=float)
        phi = np.asarray(be.to_numpy(phi_deg), dtype=float)
        w = np.asarray(be.to_numpy(wavelength), dtype=float)
        low, high = self.phi_deg
        phi = low + np.mod(phi - low, 360.0)
        return (
            (theta >= self.theta_deg[0])
            & (theta <= self.theta_deg[1])
            & (phi <= high)
            & (w >= self.wavelength[0])
            & (w <= self.wavelength[1])
        )


def _phi_arc(phi: Any) -> tuple[float, float]:
    """Return the shortest arc (low, high) in degrees that holds every angle."""
    import numpy as np

    values = np.sort(np.mod(np.asarray(phi, dtype=float), 360.0))
    if values.size == 1:
        low = float(values[0])
        return _wrap(low), _wrap(low)
    gaps = np.diff(np.concatenate([values, [values[0] + 360.0]]))
    largest = int(np.argmax(gaps))
    low = float(values[(largest + 1) % values.size])
    high = float(values[largest])
    if high < low:
        high += 360.0
    low_wrapped = _wrap(low)
    return low_wrapped, low_wrapped + (high - low)


def _wrap(angle: float) -> float:
    """Return an angle in (−180, 180]."""
    wrapped = math.fmod(angle, 360.0)
    if wrapped > 180.0:
        wrapped -= 360.0
    elif wrapped <= -180.0:
        wrapped += 360.0
    return wrapped


def _surface_samples(
    obj: Any, surface: int, wavelength: float
) -> tuple[Any, Any, Any, Any]:
    """Return (θ, φ, kx, ky) of the traced rays that reach a surface."""
    import numpy as np

    surfaces = obj.surfaces.surfaces
    target, before = surfaces[surface], surfaces[surface - 1]
    rays = RealRays(
        be.copy(target.x),
        be.copy(target.y),
        be.copy(target.z),
        be.copy(before.L),
        be.copy(before.M),
        be.copy(before.N),
        be.copy(target.intensity),
        wavelength * be.ones_like(target.x),
    )
    geometry = target.geometry
    if not hasattr(geometry, "grating_vector"):
        raise ValueError(f"Surface {surface} is not a grating.")
    geometry.localize(rays)
    normal = geometry.surface_normal(rays)
    theta, phi = grating_incidence(
        (rays.L, rays.M, rays.N), normal, geometry.grating_vector(rays)
    )
    n = be.to_numpy(target.material_pre.n(wavelength))
    theta_np = np.atleast_1d(be.to_numpy(theta))
    phi_np = np.atleast_1d(be.to_numpy(phi))
    reached = np.atleast_1d(be.to_numpy(target.intensity)) > 0
    reached &= np.isfinite(theta_np) & np.isfinite(phi_np)
    theta_np, phi_np = theta_np[reached], phi_np[reached]
    sin_theta = n * np.sin(np.deg2rad(theta_np))
    kx = np.real(sin_theta * np.cos(np.deg2rad(phi_np)))
    ky = np.real(sin_theta * np.sin(np.deg2rad(phi_np)))
    return theta_np, phi_np, kx, ky


def incidence_domain(
    optic_or_sequence: Any,
    surface: int,
    scan: Scan,
    angle_margin_deg: float = 0.0,
    wavelength_margin: float = 0.0,
) -> IncidenceDomain:
    """Return the range of (θ, φ, λ) that a scan puts on a grating surface.

    Traces every configuration × wavelength × field of the scan (a real-ray
    trace with the scan's pupil distribution) and takes the incidence of each
    ray that reaches the surface in the grating frame of this module. The
    incident direction is the direction after the step before the surface.

    Args:
        optic_or_sequence: An ``Optic`` or a ``SequencedOptic``.
        surface: The index of the surface in ``optic_or_sequence.surfaces``
            (the step index for a sequence).
        scan: The scan.
        angle_margin_deg: A margin added on both sides of the θ and φ ranges
            (θ stays ≥ 0).
        wavelength_margin: A margin added on both sides of the λ range (µm).

    Returns:
        IncidenceDomain: The ranges, their margins and the samples.

    Raises:
        ValueError: If the surface is not a grating, or no ray reaches it.
    """
    import numpy as np

    configurations = scan.configurations or [lambda obj: obj]
    thetas, phis, waves, kxs, kys = [], [], [], [], []
    for configure in configurations:
        obj = configure(optic_or_sequence)
        for wavelength in scan.wavelengths:
            for hx, hy in scan.fields:
                obj.trace(hx, hy, wavelength, scan.num_rays, scan.distribution)
                theta, phi, kx, ky = _surface_samples(obj, surface, wavelength)
                thetas.append(theta)
                phis.append(phi)
                waves.append(np.full(theta.shape, float(wavelength)))
                kxs.append(kx)
                kys.append(ky)
    theta_all = np.concatenate(thetas)
    if theta_all.size == 0:
        raise ValueError(f"No ray of the scan reaches surface {surface}.")
    phi_all = np.concatenate(phis)
    wave_all = np.concatenate(waves)
    kx_all, ky_all = np.concatenate(kxs), np.concatenate(kys)
    phi_low, phi_high = _phi_arc(phi_all)
    a, dw = float(angle_margin_deg), float(wavelength_margin)
    return IncidenceDomain(
        theta_deg=(max(float(theta_all.min()) - a, 0.0), float(theta_all.max()) + a),
        phi_deg=(phi_low - a, phi_high + a),
        wavelength=(float(wave_all.min()) - dw, float(wave_all.max()) + dw),
        k_parallel=(
            (float(kx_all.min()), float(kx_all.max())),
            (float(ky_all.min()), float(ky_all.max())),
        ),
        samples={
            "theta_deg": theta_all,
            "phi_deg": phi_all,
            "wavelength": wave_all,
            "kx": kx_all,
            "ky": ky_all,
        },
        angle_margin_deg=a,
        wavelength_margin=dw,
    )


__all__ = [
    "IncidenceDomain",
    "Scan",
    "grating_frame",
    "grating_incidence",
    "incidence_domain",
]
