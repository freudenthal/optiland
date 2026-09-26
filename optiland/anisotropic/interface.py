"""Interface Solver

The two reflected and the two transmitted child modes of a plane wave at an
interface between two media with 6 x 6 constitutive matrices, at any surface
normal, per ray, on both backends.

Procedure (Berreman 1972; Lekner 1991; McClain, Hillman and Chipman 1993):

1. The interface frame (``frames.interface_frame``): ẑ = n̂ from A into B, x̂
   along the tangential incident wave vector, K = k_in · x̂.
2. The constitutive matrices of A and B rotate into the frame.
3. The sorted, flux-scaled eigenmodes of A and B (``eigenmodes.medium_modes``).
4. The incident field is split on the two forward modes of A:
   E_in = a1 E_f1 + a2 E_f2 + c k̂_in, O_in = [E_f1, E_f2, k̂_in]. For an
   isotropic A and a transverse E_in, c = 0; for one eigenmode of an
   anisotropic A, E_in is that mode's field and c = 0. ``residual`` = c.
5. Continuity of (E_x, E_y, H'_x, H'_y) at z = 0:
   a ψ_f(A) + r1 ψ_b1(A) + r2 ψ_b2(A) = t1 ψ_f1(B) + t2 ψ_f2(B).
6. The children (r1, r2, t1, t2) rotate back to the global frame.

Outputs per child j:

* The power fraction is ±S_j · n̂ / S_in · n̂ with S = Re(E × H'*) / 2 at the
  solved amplitudes; an evanescent child has power 0 and no ray (NaN).
  With loss the fluxes of two children are not additive: ``reflectance`` and
  ``transmittance`` are the fluxes of the summed reflected and transmitted
  fields.
* The PRT matrix P_j maps the forward-mode fields of A to the child field and
  the incident wave normal to the child wave normal: P_j E_fi(A) = X_ji E_j,
  P_j k̂_in = k̂_j (the third column is k̂, not Ŝ), det P_j = 0. With
  O_in⁻¹ = [w1; w2; w3]: P_j = E_j (X_j1 w1 + X_j2 w2) + k̂_j w3. For
  transverse E_in in an isotropic A, Σ_j P_j E_in (transmitted children) is the
  transmitted field of the Yun, Crabtree and Chipman PRT matrix.
* The ray direction S_j / |S_j| is separate ray state; the walk-off is the
  angle between Re k_j and S_j.

References:

* D. W. Berreman, J. Opt. Soc. Am. 62, 502-510 (1972).
* J. Lekner, J. Phys.: Condens. Matter 3, 6121-6133 (1991).
* S. C. McClain, L. W. Hillman, R. A. Chipman, "Polarization ray tracing in
  anisotropic optically active media. I. Algorithms," J. Opt. Soc. Am. A 10,
  2371-2382 (1993).
* G. Yun, K. Crabtree, R. A. Chipman, "Three-dimensional polarization ray-tracing
  calculus I: definition and diattenuation," Appl. Opt. 50, 2855-2865 (2011).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

import optiland.backend as be
from optiland.anisotropic.eigenmodes import (
    Eigenmodes,
    _c,
    _flux_z,
    _modes,
    _poynting,
)
from optiland.anisotropic.frames import (
    _complex_vectors,
    _sum,
    _unit,
    interface_frame,
    rotate_constitutive,
    to_global,
)
from optiland.materials.anisotropic import BaseTensorMaterial, _complex

Array = Any

__all__ = [
    "CHILD_LABELS",
    "InterfaceResult",
    "constitutive_matrix",
    "solve_interface",
]

#: The child order on axis 1 of the results.
CHILD_LABELS = ("r1", "r2", "t1", "t2")

_SIGN = np.array([-1.0, -1.0, 1.0, 1.0])
_IS_T = np.array([0.0, 0.0, 1.0, 1.0])


def constitutive_matrix(material: Any, wavelength: Any) -> Array:
    """Return the 6 x 6 constitutive matrix of any Optiland material.

    A tensor material gives ``constitutive_6x6``. A scalar material gives
    ε = (n + ik)² I, μ = I, ξ = ζ = 0.

    Args:
        material: A ``BaseTensorMaterial`` or a scalar Optiland material.
        wavelength: Wavelengths in µm, scalar or shape (N,).

    Returns:
        The matrices [[ε, ξ], [ζ, μ]] in the global frame, shape (N, 6, 6),
        complex.
    """
    if isinstance(material, BaseTensorMaterial):
        return material.constitutive_6x6(wavelength)
    w = be.reshape(be.atleast_1d(be.asarray(wavelength)), (-1,))
    n = be.to_complex(be.asarray(material.n(w)))
    k = be.to_complex(be.asarray(material.k(w)))
    eps = be.reshape((n + 1j * k) ** 2, (-1,))
    eps_block = _c(np.diag([1.0, 1.0, 1.0, 0.0, 0.0, 0.0]))
    mu_block = _c(np.diag([0.0, 0.0, 0.0, 1.0, 1.0, 1.0]))
    return eps[:, None, None] * eps_block[None] + mu_block[None]


@dataclass
class InterfaceResult:
    """The four child modes of an interface, per ray.

    Child axis (axis 1) order: ``CHILD_LABELS`` = (r1, r2, t1, t2), the
    s-like and p-like backward modes of A and the s-like and p-like forward
    modes of B. Vectors are in the global frame; wave vectors in units of k0.

    Attributes:
        k: Child wave vectors, shape (N, 4, 3), complex.
        E: Child electric fields at the solved amplitudes, shape (N, 4, 3).
        H: Child fields H' = η0 H at the solved amplitudes, shape (N, 4, 3).
        S: Child Poynting vectors Re(E × H'*) / 2, shape (N, 4, 3).
        ray: Unit ray directions S / |S| of the child modes (NaN for an
            evanescent child), shape (N, 4, 3). Defined also for a child of
            zero amplitude.
        walkoff: Angle between Re k and S of the child modes in radians (NaN
            for an evanescent child), shape (N, 4).
        amplitude: Child amplitudes on the flux-scaled mode fields, shape
            (N, 4), complex: E = amplitude E_mode.
        jones: Child amplitudes per unit amplitude of each forward mode of A,
            shape (N, 4, 2), complex (the partial Jones matrices on the modes).
        E_mode: Flux-scaled child mode fields, shape (N, 4, 3), complex.
        H_mode: Flux-scaled child mode fields H', shape (N, 4, 3), complex.
        power: Power fraction of each child, shape (N, 4) (0 if evanescent).
        prt: PRT matrix of each child, shape (N, 4, 3, 3), complex.
        evanescent: True for a child that carries no power along n̂, (N, 4).
        reflectance: Flux of the summed reflected field / incident flux, (N,).
        transmittance: Flux of the summed transmitted field / incident flux,
            (N,).
        balance: reflectance + transmittance - 1, shape (N,) (0 for lossless
            media).
        incident_amplitude: Amplitudes (a1, a2) of E_in on the forward modes
            of A, shape (N, 2), complex.
        residual: The k̂_in component c of E_in (0 if E_in is a sum of forward
            modes of A), shape (N,), complex.
        incident_flux: S_in · n̂ of the incident field, shape (N,).
        degenerate: Degenerate-pair flags (forward A, backward A, forward B,
            backward B), shape (N, 4).
        fallback: Sorting fallback flags (A, B), shape (N, 2).
        rotation: Frame rotations (rows x̂, ŷ, n̂), shape (N, 3, 3).
        tangential: Tangential wave numbers K, shape (N,), complex.
        modes_a: The eigenmodes of A in the frame.
        modes_b: The eigenmodes of B in the frame.
    """

    k: Array
    E: Array
    H: Array
    S: Array
    ray: Array
    walkoff: Array
    amplitude: Array
    jones: Array
    E_mode: Array
    H_mode: Array
    power: Array
    prt: Array
    evanescent: Array
    reflectance: Array
    transmittance: Array
    balance: Array
    incident_amplitude: Array
    residual: Array
    incident_flux: Array
    degenerate: Array
    fallback: Array
    rotation: Array
    tangential: Array
    modes_a: Eigenmodes
    modes_b: Eigenmodes


def _matrices(matrix: Any, n_rays: int) -> Array:
    """Return constitutive matrices as complex (n_rays, 6, 6) backend arrays."""
    m = _complex(matrix)
    if len(m.shape) == 2:
        m = be.reshape(m, (1, 6, 6))
    return be.broadcast_to(m, (n_rays, 6, 6))


def solve_interface(
    normal: Any,
    medium_a: Any,
    medium_b: Any,
    k_in: Any,
    E_in: Any,
    x_ref: Any = None,
) -> InterfaceResult:
    """Solve the interface between media A and B for incident plane waves.

    Args:
        normal: Unit surface normals from A into B, shape (N, 3) or (3,).
        medium_a: Constitutive matrices [[ε, ξ], [ζ, μ]] of the incident
            medium in the global frame, shape (N, 6, 6) or (6, 6) (for
            example ``constitutive_matrix(material, wavelength)``).
        medium_b: Constitutive matrices of the exit medium, the same shapes.
        k_in: Incident wave vectors in units of k0, shape (N, 3) or (3,),
            complex allowed. For an isotropic A, k_in = n d̂; for an anisotropic
            A, the wave vector of the incident eigenmode (``plane_wave_modes``).
        E_in: Incident electric fields, shape (N, 3) or (3,), complex. For an
            isotropic A any field transverse to k_in; for an anisotropic A the
            field of the incident eigenmode (any scale and phase).
        x_ref: Reference vectors for x̂ at normal incidence (see
            ``interface_frame``).

    Returns:
        The four children and the energy report of each ray.
    """
    k_vec = _complex_vectors(k_in)
    e_vec = _complex_vectors(E_in)
    normals = be.asarray(normal)
    n_normals = 1 if len(normals.shape) == 1 else normals.shape[0]
    n_rays = max(k_vec.shape[0], e_vec.shape[0], n_normals)
    for m in (medium_a, medium_b):
        shape = getattr(m, "shape", np.shape(m))
        if len(shape) == 3:
            n_rays = max(n_rays, shape[0])
    k_vec = be.broadcast_to(k_vec, (n_rays, 3))
    e_vec = be.broadcast_to(e_vec, (n_rays, 3))

    rotation, tangential = interface_frame(normal, k_vec, x_ref)
    rotation = be.broadcast_to(rotation, (n_rays, 3, 3))
    tangential = be.broadcast_to(tangential, (n_rays,))
    rc = be.to_complex(rotation)
    ma = rotate_constitutive(_matrices(medium_a, n_rays), rotation)
    mb = rotate_constitutive(_matrices(medium_b, n_rays), rotation)
    modes_a = _modes(tangential, ma)
    modes_b = _modes(tangential, mb)

    # The incident field on the forward modes of A.
    k_frame = be.matmul(rc, k_vec[:, :, None])[:, :, 0]
    e_frame = be.matmul(rc, e_vec[:, :, None])[:, :, 0]
    k_hat_in = be.to_complex(_unit(be.real(k_frame)))
    e3 = _c(np.eye(3))
    o_in = be.concatenate(
        [be.transpose(modes_a.E[:, :2, :], (0, 2, 1)), k_hat_in[:, :, None]], axis=-1
    )
    w = be.linalg.inv(o_in)
    coeff = be.matmul(w, e_frame[:, :, None])[:, :, 0]
    a = coeff[:, :2]
    residual = coeff[:, 2]

    # Boundary matching: [ψ_b1(A), ψ_b2(A), -ψ_f1(B), -ψ_f2(B)] X = -[ψ_f(A)].
    match = be.concatenate([modes_a.psi[:, :, 2:4], -modes_b.psi[:, :, 0:2]], axis=-1)
    incident_psi = modes_a.psi[:, :, 0:2]
    jones = be.linalg.solve(match, -incident_psi)  # (N, 4, 2)
    amplitude = be.matmul(jones, a[:, :, None])[:, :, 0]

    # Child modes in the frame.
    def children(a_part: Array, b_part: Array) -> Array:
        """Return (b1(A), b2(A), f1(B), f2(B)) on axis 1."""
        return be.concatenate([a_part[:, 2:4], b_part[:, 0:2]], axis=1)

    q = children(modes_a.q, modes_b.q)
    e_mode = children(modes_a.E, modes_b.E)
    h_mode = children(modes_a.H, modes_b.H)
    psi_child = be.concatenate(
        [modes_a.psi[:, :, 2:4], modes_b.psi[:, :, 0:2]], axis=-1
    )  # (N, 4, 4): columns = children
    evanescent = be.concatenate(
        [modes_a.evanescent[:, 2:], modes_b.evanescent[:, :2]], axis=-1
    )

    k_child = (
        tangential[:, None, None] * e3[0][None, None, :]
        + q[:, :, None] * e3[2][None, None, :]
    )
    field = amplitude[:, :, None] * e_mode
    field_h = amplitude[:, :, None] * h_mode

    # Energy.
    # An incident field without flux (not a propagating mode of A) gives NaN.
    flux_in = _flux_z(be.matmul(incident_psi, a[:, :, None]))[:, 0]
    has_flux = flux_in != 0
    nan = be.zeros_like(flux_in) + float("nan")
    inv_flux = be.where(
        has_flux, 1.0 / be.where(has_flux, flux_in, 1.0 + 0 * flux_in), nan
    )
    flux_child = _flux_z(psi_child * amplitude[:, None, :])  # (N, 4)
    sign = be.asarray(_SIGN)
    power = be.where(
        evanescent,
        be.zeros_like(flux_child),
        sign[None, :] * flux_child * inv_flux[:, None],
    )
    is_t = _c(_IS_T)[None, None, :]
    summed_t = be.matmul(psi_child * is_t, amplitude[:, :, None])
    summed_r = be.matmul(psi_child * (1.0 - is_t), amplitude[:, :, None])
    transmittance = _flux_z(summed_t)[:, 0] * inv_flux
    reflectance = -_flux_z(summed_r)[:, 0] * inv_flux

    # PRT per child in the frame: P_j = E_j (X_j1 w1 + X_j2 w2) + k̂_j w3.
    k_hat_child = be.to_complex(_unit(be.real(k_child)))
    rows = be.matmul(jones, w[:, :2, :])  # (N, 4, 3)
    prt_frame = (
        e_mode[:, :, :, None] * rows[:, :, None, :]
        + k_hat_child[:, :, :, None] * w[:, None, None, 2, :]
    )
    rct = be.transpose(rc, (0, 2, 1))
    prt = be.matmul(be.matmul(rct[:, None], prt_frame), rc[:, None])

    # Global frame.
    k_g = to_global(k_child, rotation)
    e_g = to_global(field, rotation)
    h_g = to_global(field_h, rotation)
    s_g = _poynting(e_g, h_g)
    # The ray and the walk-off are properties of the child mode (defined also
    # for a child of zero amplitude).
    e_mode_g = to_global(e_mode, rotation)
    h_mode_g = to_global(h_mode, rotation)
    s_mode = _poynting(e_mode_g, h_mode_g)
    s_norm = be.sqrt(_sum(s_mode**2, axis=-1))
    safe = be.where(s_norm > 0, s_norm, be.ones_like(s_norm))
    nan = be.zeros_like(s_mode) + float("nan")
    ray = be.where(evanescent[:, :, None], nan, s_mode / safe[:, :, None])
    k_real = be.real(k_g)
    cross = be.cross(k_real, s_mode)
    walk = be.arctan2(be.sqrt(_sum(cross**2, axis=-1)), _sum(k_real * s_mode, axis=-1))
    walkoff = be.where(evanescent, be.zeros_like(walk) + float("nan"), walk)

    return InterfaceResult(
        k=k_g,
        E=e_g,
        H=h_g,
        S=s_g,
        ray=ray,
        walkoff=walkoff,
        amplitude=amplitude,
        jones=jones,
        E_mode=e_mode_g,
        H_mode=h_mode_g,
        power=power,
        prt=prt,
        evanescent=evanescent,
        reflectance=reflectance,
        transmittance=transmittance,
        balance=reflectance + transmittance - 1.0,
        incident_amplitude=a,
        residual=residual,
        incident_flux=flux_in,
        degenerate=be.concatenate([modes_a.degenerate, modes_b.degenerate], axis=-1),
        fallback=be.concatenate(
            [modes_a.fallback[:, None], modes_b.fallback[:, None]], axis=-1
        ),
        rotation=rotation,
        tangential=tangential,
        modes_a=modes_a,
        modes_b=modes_b,
    )
