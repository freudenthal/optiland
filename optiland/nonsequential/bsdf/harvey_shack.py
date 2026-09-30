"""Harvey-Shack (ABg) BSDF for Non-Sequential Raytracing.

Models micro-roughness scatter from optical surfaces.

Kramer Harrison, 2026
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

import optiland.backend as be
from optiland.backend.utils import to_numpy
from optiland.nonsequential.bsdf.base import BaseBSDF
from optiland.nonsequential.rng import EventSlot

if TYPE_CHECKING:
    from optiland.nonsequential.rng import NSQRng

# Largest direction-cosine offset of a scattered ray from its reference ray:
# both lie in the unit disk, so |beta - beta0| <= 2.
_BETA_MAX = 2.0
# Nodes of the radial table. The nodes are log-spaced from _RHO_MIN_FACTOR * l0
# to _BETA_MAX, so the table resolves the lobe core of a polished surface
# (l0 of 1e-4 to 1e-3) as well as the far wing.
_TABLE_SIZE = 4096
_RHO_MIN_FACTOR = 1e-6
# Rows (|beta0| from 0 to 1) and azimuth quantiles of the reachable-region table.
_B_ROWS = 129
_PSI_NODES = 721
_QUANTILES = 1025


class HarveyShackBSDF(BaseBSDF):
    """Harvey-Shack scatter model in ABg form, for surface micro-roughness.

    The lobe depends only on the distance ``rho = |beta - beta0|`` in
    direction-cosine space between the scattered direction ``beta`` and the
    reference direction ``beta0`` (the specular reflection, or the
    straight-through ray of a transmissive lobe), both projected on the
    surface tangent plane::

        BSDF(rho) = b0 / (1 + (rho / l0)**s)

    This is the ABg model ``BSDF = A / (B + rho**g)`` with ``A = b0 * l0**s``,
    ``B = l0**s`` and ``g = s`` (see :meth:`from_abg`). ``b0`` is the level at
    the reference direction, the lobe is ``b0 / 2`` at ``rho = l0``, and far
    from the reference direction it falls as ``rho**-s``.

    Scattered directions exist only inside the unit disk ``|beta| < 1`` (the
    hemisphere of the reference ray). The lobe is therefore integrated and
    sampled over the part of the disk around ``beta0`` that a ray can reach.
    That region depends on the angle of the reference ray, so the integrated
    scatter does too (:meth:`integrated_scatter`).

    Attributes:
        b0: BSDF at the reference direction [sr^-1].
        l0: Shoulder of the lobe, in direction-cosine units.
        s: Roll-off exponent of the wing (positive).
        transmissive_fraction: Probability in [0, 1] that a given scatter
            event blurs the undeviated straight-through ray (the
            transmissive lobe, e.g. a diffuser sheet) instead of the
            specular reflection. Defaults to 0.0: a purely reflective
            blur, identical to this class's behaviour before D-5.
    """

    def __init__(
        self, b0: float, l0: float, s: float, transmissive_fraction: float = 0.0
    ) -> None:
        """Initialize HarveyShackBSDF.

        Args:
            b0: BSDF at the reference direction [sr^-1].
            l0: Shoulder of the lobe, in direction-cosine units (positive).
            s: Roll-off exponent of the wing (positive).
            transmissive_fraction: Probability in [0, 1] that a scatter
                event blurs the straight-through ray instead of the
                specular reflection.
        """
        self.b0 = float(b0)
        self.l0 = float(l0)
        self.s = float(s)
        self.transmissive_fraction = float(transmissive_fraction)
        self._rho_grid: np.ndarray | None = None
        self._radial_cdf: np.ndarray | None = None
        self._b_grid: np.ndarray | None = None
        self._scatter_by_b: np.ndarray | None = None
        self._psi_quantiles: np.ndarray | None = None

    @classmethod
    def from_abg(
        cls, A: float, B: float, g: float, transmissive_fraction: float = 0.0
    ) -> HarveyShackBSDF:
        """Build the model from the ABg parameters ``BSDF = A / (B + rho**g)``.

        Args:
            A: ABg numerator [sr^-1].
            B: ABg shoulder term (positive), ``l0**g``.
            g: ABg exponent (positive).
            transmissive_fraction: See the class docstring.

        Returns:
            The equivalent HarveyShackBSDF, with ``b0 = A / B``,
            ``l0 = B**(1/g)`` and ``s = g``.
        """
        return cls(
            b0=A / B,
            l0=B ** (1.0 / g),
            s=g,
            transmissive_fraction=transmissive_fraction,
        )

    def _abg(self, beta: np.ndarray) -> np.ndarray:
        """Evaluate the lobe at a direction-cosine offset.

        Args:
            beta: Magnitude of the direction-cosine offset from the reference
                direction.

        Returns:
            BSDF value [sr^-1].
        """
        return self.b0 / (1.0 + (beta / self.l0) ** self.s)

    def _lobe(self, rho: np.ndarray) -> np.ndarray:
        """The lobe used by the tables (the ABg form)."""
        return self._abg(rho)

    def _build_tables(self) -> None:
        """Build the radial and azimuthal sampling tables and the integrated scatter.

        In direction-cosine space the projected solid angle is
        ``cos(theta) dOmega = d(beta_x) d(beta_y)``. In polar coordinates
        ``(rho, psi)`` about ``beta0`` the scattered power is

            P(beta0) = integral over psi of F(rho_max(psi)) d(psi),
            F(r) = integral from 0 to r of BSDF(rho) * rho d(rho),

        where ``rho_max(psi)`` is the distance from ``beta0`` to the unit
        circle along the azimuth ``psi``. ``F`` is tabulated on log-spaced
        nodes (trapezoidal rule in ``ln(rho)``). For each ``|beta0|`` row the
        azimuth distribution ``F(rho_max(psi)) / P`` is stored as quantiles.
        """
        rho_min = min(_RHO_MIN_FACTOR * self.l0, 1e-3 * _BETA_MAX)
        rho = np.concatenate(
            [[0.0], np.geomspace(rho_min, _BETA_MAX, _TABLE_SIZE - 1)]
        )
        g = self._lobe(rho) * rho**2  # integrand of F in d(ln rho)
        seg = 0.5 * (g[2:] + g[1:-1]) * np.diff(np.log(rho[1:]))
        first = 0.5 * self._lobe(np.array([0.0]))[0] * rho[1] ** 2
        cdf = np.concatenate([[0.0], first + np.concatenate([[0.0], np.cumsum(seg)])])
        self._rho_grid, self._radial_cdf = rho, cdf

        b = np.linspace(0.0, 1.0, _B_ROWS)
        psi = np.linspace(0.0, np.pi, _PSI_NODES)
        g_psi = self._radial(self._rho_max(b[:, None], psi[None, :]))
        cum = np.concatenate(
            [
                np.zeros((b.size, 1)),
                np.cumsum(0.5 * (g_psi[:, 1:] + g_psi[:, :-1]) * np.diff(psi), axis=1),
            ],
            axis=1,
        )
        self._b_grid = b
        self._scatter_by_b = 2.0 * cum[:, -1]  # psi in [0, pi], symmetric in psi
        levels = np.linspace(0.0, 1.0, _QUANTILES)
        quant = np.empty((b.size, _QUANTILES))
        for j in range(b.size):
            if cum[j, -1] > 0.0:
                quant[j] = np.interp(levels, cum[j] / cum[j, -1], psi)
            else:  # no reachable region (|beta0| = 1 exactly)
                quant[j] = np.pi
        self._psi_quantiles = quant

    @staticmethod
    def _rho_max(b: np.ndarray, psi: np.ndarray) -> np.ndarray:
        """Distance from a point at radius ``b`` to the unit circle, along the
        azimuth ``psi`` measured from the point's own radial direction."""
        c = b * np.cos(psi)
        return -c + np.sqrt(np.clip(c * c + 1.0 - b * b, 0.0, None))

    def _radial(self, r: np.ndarray) -> np.ndarray:
        """F(r): the radial integral of ``BSDF(rho) * rho`` from 0 to r."""
        return np.interp(r, self._rho_grid, self._radial_cdf)

    def integrated_scatter(self, beta0: float | np.ndarray = 0.0) -> np.ndarray:
        """Fraction of the incident power scattered into the reachable region.

        Args:
            beta0: Magnitude of the reference direction projected on the
                surface tangent plane (``sin`` of its angle from the normal),
                in [0, 1]. 0 is normal incidence.

        Returns:
            The integrated BSDF over the part of the unit disk around
            ``beta0`` (not clipped).
        """
        if self._rho_grid is None:
            self._build_tables()
        return np.interp(np.abs(beta0), self._b_grid, self._scatter_by_b)

    @property
    def total_integrated_scatter(self) -> float:
        """Fraction of incident power scattered at normal incidence, in [0, 1].

        This is :meth:`integrated_scatter` at ``beta0 = 0``: the lobe integrated
        over the unit disk ``rho <= 1``. It is clipped to 1.0; a larger value
        means that ``b0`` is too large for a physical surface.

        Returns:
            TIS at normal incidence, clipped to 1.0.
        """
        return min(float(self.integrated_scatter(0.0)), 1.0)

    def sample(
        self,
        num_rays: int,
        incident_dirs: np.ndarray,
        normals: np.ndarray,
        wavelengths: np.ndarray,
        rng: NSQRng,
        ray_id: np.ndarray,
        bounce: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Sample scattered directions from the lobe about a reference ray.

        The scatter offset is drawn from the lobe restricted to the region
        that a scattered ray can reach (the unit disk of direction cosines).
        The azimuth about the reference direction comes from the tabulated
        distribution ``F(rho_max(psi)) / P`` of the row of ``|beta0|``; the
        radial offset ``rho`` comes from the inverse of ``F`` truncated at
        ``rho_max(psi)``. Every sample is reachable and carries its full
        flux, so the surface acts as a mirror (or diffuser sheet) whose
        reflection (or straight-through transmission) is blurred by the
        lobe, with no power lost: a polished surface (small ``l0``) stays
        near-specular/near-collimated, a rough one spreads.

        A per-ray draw against :attr:`transmissive_fraction` picks the
        reference ray the lobe is centred on: the specular reflection for a
        reflective draw, or the undeviated straight-through ray (the
        incident direction itself, unrefracted) for a transmissive one. Both
        references are expressed in the same tangent frame about ``normals``,
        so the existing ``spec_normal_sign`` (the reference ray's own sign
        against ``normals``) places the reconstructed sample on the correct
        side automatically -- no separate branch is needed downstream.

        To model the physically scaled picture instead, a bright specular beam
        plus a faint scatter halo, set the surface's ``scatter_fraction`` to
        :meth:`integrated_scatter` at the reference angle of interest
        (:attr:`total_integrated_scatter` for normal incidence). Then that
        fraction of the rays enter the halo and the rest reflect specularly.

        Sampling is detached (numpy); weights are plain scalars.

        Args:
            num_rays: Number of rays.
            incident_dirs: Incident directions, shape (N, 3).
            normals: Surface normals, shape (N, 3).
            wavelengths: Wavelengths [µm], shape (N,).
            rng: Keyed PCG32 RNG.
            ray_id: Per-ray identifiers, shape (N,).
            bounce: Per-ray bounce/step index, shape (N,).

        Returns:
            (scattered_dirs, flux_weights, transmitted).
        """
        if self._rho_grid is None:
            self._build_tables()

        n_np = np.asarray(to_numpy(normals), dtype=np.float64)
        d_np = np.asarray(to_numpy(incident_dirs), dtype=np.float64)

        # Specular reflection: d - 2(d.n)n
        cos_i = (d_np * n_np).sum(axis=1, keepdims=True)
        d_spec = d_np - 2.0 * cos_i * n_np

        # Per-ray reflective-vs-transmissive lobe draw: the reference
        # ray the blur is centred on. d_np itself (unrefracted) is
        # already a unit vector; only d_spec needs the below norm-guard.
        if self.transmissive_fraction > 0.0:
            u_lobe = rng.uniform(ray_id, bounce, EventSlot.BSDF_LOBE_BRANCH)
            transmitted_np = u_lobe < self.transmissive_fraction
        else:
            transmitted_np = np.zeros(n_np.shape[0], dtype=bool)
        d_ref = np.where(transmitted_np[:, None], d_np, d_spec)

        # Rays that hit nothing carry zero direction and normal, so the
        # reference vector is zero. Guard the normalisation: a NaN here
        # propagates into the returned weights for every ray.
        d_ref_norm = (d_ref * d_ref).sum(axis=1, keepdims=True) ** 0.5
        valid = d_ref_norm[:, 0] > 1e-12
        d_ref = np.divide(
            d_ref, d_ref_norm, out=np.zeros_like(d_ref), where=d_ref_norm > 1e-12
        )

        from optiland.nonsequential.bsdf.lambertian import (  # noqa: PLC0415
            _orthonormal_basis,
        )

        t_vec, b_vec = _orthonormal_basis(n_np)

        # Reference direction expressed in the local tangent frame.
        beta0_x = (d_ref * t_vec).sum(axis=1)
        beta0_y = (d_ref * b_vec).sum(axis=1)
        ref_normal_sign = np.sign((d_ref * n_np).sum(axis=1))
        ref_normal_sign[ref_normal_sign == 0.0] = 1.0

        # Polar frame about beta0: e_r along beta0 (any axis at beta0 = 0).
        b0_mag = np.clip(np.hypot(beta0_x, beta0_y), 0.0, 1.0)
        phi0 = np.arctan2(beta0_y, beta0_x)
        er_x, er_y = np.cos(phi0), np.sin(phi0)

        # Azimuth from the |beta0| row of the quantile table (linear in the
        # row and in the level), mirrored to (-pi, pi] by the half of the draw.
        u_azimuth = rng.uniform(ray_id, bounce, EventSlot.BSDF_U2)
        u_radial = rng.uniform(ray_id, bounce, EventSlot.BSDF_U1)
        side = np.where(u_azimuth < 0.5, 1.0, -1.0)
        level = np.clip(2.0 * u_azimuth - np.where(u_azimuth < 0.5, 0.0, 1.0), 0, 1)
        row = b0_mag * (_B_ROWS - 1)
        j = np.minimum(row.astype(int), _B_ROWS - 2)
        w_row = row - j
        col = level * (_QUANTILES - 1)
        k = np.minimum(col.astype(int), _QUANTILES - 2)
        w_col = col - k
        q = self._psi_quantiles
        psi_lo = (1 - w_col) * q[j, k] + w_col * q[j, k + 1]
        psi_hi = (1 - w_col) * q[j + 1, k] + w_col * q[j + 1, k + 1]
        psi = side * ((1 - w_row) * psi_lo + w_row * psi_hi)

        # Radius from F truncated at the rim of the disk along that azimuth.
        rho_max = self._rho_max(b0_mag, psi)
        target = u_radial * self._radial(rho_max)
        delta = np.minimum(
            np.interp(target, self._radial_cdf, self._rho_grid), rho_max
        )

        cos_psi, sin_psi = np.cos(psi), np.sin(psi)
        beta_x = beta0_x + delta * (cos_psi * er_x - sin_psi * er_y)
        beta_y = beta0_y + delta * (cos_psi * er_y + sin_psi * er_x)

        beta_sq = beta_x**2 + beta_y**2
        normal_comp = np.sqrt(np.clip(1.0 - beta_sq, 0.0, None))

        scattered = (
            beta_x[:, None] * t_vec
            + beta_y[:, None] * b_vec
            + (ref_normal_sign * normal_comp)[:, None] * n_np
        )
        scattered = np.where(valid[:, None], scattered, d_ref)

        norms = (scattered * scattered).sum(axis=1, keepdims=True) ** 0.5
        scattered = np.divide(
            scattered, norms, out=np.zeros_like(scattered), where=norms > 1e-12
        )

        # Full flux: the lobe redistributes energy rather than removing it.
        # The physical scatter level is applied via ``scatter_fraction``,
        # for which :meth:`integrated_scatter` is the natural value.
        flux_weights = np.where(valid, 1.0, 0.0)

        return (
            be.array(scattered.astype(np.float64)),
            be.array(flux_weights),
            be.array(transmitted_np),
        )

    def reflectance(
        self,
        incident_dirs: np.ndarray,
        normals: np.ndarray,
        wavelengths: np.ndarray,
    ) -> np.ndarray:
        """Return the fraction of incident power redistributed by the lobe.

        The sampler conserves energy (rays keep full flux and are only
        redirected, always into the reachable hemisphere), so this is 1.0.
        The scatter level itself is :meth:`integrated_scatter`, which is what
        a ``scatter_fraction`` should be set to for a physically scaled halo.

        Args:
            incident_dirs: Incident directions, shape (N, 3).
            normals: Surface normals, shape (N, 3).
            wavelengths: Wavelengths [µm], shape (N,).

        Returns:
            Approximate reflectance values, shape (N,).
        """
        return be.ones(incident_dirs.shape[0])
