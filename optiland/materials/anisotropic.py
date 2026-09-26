"""Anisotropic and Bianisotropic Materials

This module defines materials whose optical response is a set of 3 x 3
tensors instead of one refractive index. A tensor material gives the four
constitutive tensors (ε, μ, ξ, ζ) in the global frame at any wavelength, on
the NumPy and the Torch backend.

Conventions:

* Time dependence exp(-iωt). Loss is Im ε > 0 (index n + ik with k ≥ 0).
* Tellegen form with dimensionless blocks and H' = η0 H:
  D' = ε E + ξ H', B' = ζ E + μ H'. A dielectric has μ = I and ξ = ζ = 0.
* Wavelengths are in µm (the unit of Optiland materials).
* Orientation: a proper rotation R maps the crystal frame into the global
  frame, X_global = R X_crystal Rᵀ for each of the four tensors. The columns
  of R are the crystal axes in global coordinates. ``euler_zxz_matrix`` gives
  R = Rz(φ) Rx(θ) Rz(ψ) (the z-x'-z'' proper Euler sequence).
* Uniaxial: ε = ε_o I + (ε_e - ε_o) ĉ ĉᵀ, ĉ the unit optic axis. Biaxial:
  ε = diag(n_x², n_y², n_z²) in the principal (crystal) frame.
* Optical activity is the Tellegen form ξ = iα, ζ = -iαᵀ with a real α for a
  lossless crystal. A Pasteur medium has α = κI. κ > 0 gives n_L > n_R: a
  linear polarization turns clockwise as seen by an observer who looks toward
  the source (dextrorotatory, right quartz). A gyration (Landau) tensor γ,
  D = ε_L E + i(γk) x E, maps to α = (tr γ / 2) I - γᵀ (inverse
  γ = (tr α) I - αᵀ), and ε = ε_L + α αᵀ. The bulk plane waves of the two
  forms are the same. Interfaces must be matched in the Tellegen form.

The tensor materials subclass ``BaseMaterial``, so a surface can hold them.
A tensor material has no single refractive index: ``n`` and ``k`` raise
``TypeError``. The scalar Optiland materials do not change.

References:

* D. W. Berreman, "Optics in stratified and anisotropic media: 4 x 4-matrix
  formulation," J. Opt. Soc. Am. 62, 502-510 (1972).
* J. Lekner, "Reflection and refraction by uniaxial crystals," J. Phys.:
  Condens. Matter 3, 6121-6133 (1991). Eq. (18): the uniaxial ε.
* T. G. Mackay, A. Lakhtakia, Electromagnetic Anisotropy and
  Bianisotropy (World Scientific, 2010). The Tellegen form, the Post
  constraint tr(μ⁻¹(ξ + ζ)) = 0 and the reciprocity conditions
  ε = εᵀ, μ = μᵀ, ξ = -ζᵀ.
* O. Arteaga, A. Canillas, G. E. Jellison, "Determination of the components
  of the gyration tensor of quartz by oblique incidence transmission
  two-modulator generalized ellipsometry," Appl. Opt. 48, 5307-5317 (2009),
  and O. Arteaga, J. H. Freudenthal, B. Kahr, "Reckoning electromagnetic
  principles with polarimetric measurements of anisotropic optically active
  crystals," J. Appl. Cryst. 45, 279-291 (2012): the Tellegen form of optical
  activity and, from its Table 1, γ11/γ33 = -0.525 for quartz near 589 nm.
* E. U. Condon, "Theories of optical rotatory power," Rev. Mod. Phys. 9,
  432-457 (1937).
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Any

import numpy as np

import optiland.backend as be
from optiland.materials.base import BaseMaterial
from optiland.propagation.anisotropic import AnisotropicPropagation

if TYPE_CHECKING:
    from collections.abc import Callable

    from optiland.propagation.base import BasePropagationModel

Array = Any
TensorSpec = Any  # a 3 x 3 constant, a callable of the wavelength or a material

__all__ = [
    "BaseTensorMaterial",
    "BiaxialMaterial",
    "BianisotropicMaterial",
    "QuartzMaterial",
    "TensorMaterial",
    "UniaxialMaterial",
    "alpha_from_gyration",
    "euler_zxz_matrix",
    "gyration_from_alpha",
    "kappa_from_rotatory_power",
    "quartz_alpha",
    "quartz_rotatory_power",
    "standard_air_index",
]

_ROTATION_TOLERANCE = 1e-9


# ---------------------------------------------------------------------------
# Backend helpers
# ---------------------------------------------------------------------------


def _complex(x: Any) -> Array:
    """Return x as a complex array of the active backend.

    ``be.asarray`` casts to the real default precision (on Torch it drops the
    imaginary part), so the real and the imaginary parts are converted one at
    a time.
    """
    if hasattr(x, "detach"):
        if be.get_backend() == "torch":
            return be.to_complex(x)
        x = x.detach().cpu().numpy()
    array = np.asarray(x)
    if be.get_backend() == "numpy":
        return array.astype(np.complex128)
    real = be.to_complex(be.asarray(np.ascontiguousarray(np.real(array))))
    if np.iscomplexobj(array):
        imag = be.to_complex(be.asarray(np.ascontiguousarray(np.imag(array))))
        return real + 1j * imag
    return real


def _numpy(x: Any) -> Any:
    """Return a Torch tensor as a NumPy array; return other values unchanged."""
    if hasattr(x, "detach"):
        return x.detach().cpu().numpy()
    return x


def _wavelengths(wavelength: float | Array) -> Array:
    """Return the wavelengths as a 1D real array of the active backend."""
    if hasattr(wavelength, "detach") and be.get_backend() == "numpy":
        wavelength = wavelength.detach().cpu().numpy()
    w = be.atleast_1d(be.asarray(wavelength))
    return be.ravel(w)


def _identity(n: int) -> Array:
    """Return n copies of the complex 3 x 3 identity, shape (n, 3, 3)."""
    return be.to_complex(be.zeros((n, 3, 3))) + be.to_complex(be.eye(3))


def _swap(x: Array) -> Array:
    """Return the transpose of each 3 x 3 matrix of x, shape (N, 3, 3)."""
    return be.transpose(x, (0, 2, 1))


def _trace(x: Array) -> Array:
    """Return the trace of each matrix of x, shape (...,)."""
    return x[..., 0, 0] + x[..., 1, 1] + x[..., 2, 2]


def _complex_index(material: BaseMaterial, w: Array) -> Array:
    """Return n + ik of a scalar material at the wavelengths w."""
    n = be.to_complex(be.asarray(material.n(w)))
    k = be.to_complex(be.asarray(material.k(w)))
    return be.reshape(n + 1j * k, (-1,))


def _check_rotation(rotation: Any) -> np.ndarray:
    """Return a rotation as a float64 array; raise if it is not proper."""
    if rotation is None:
        return np.eye(3)
    r = np.asarray(_numpy(rotation), dtype=np.float64)
    if r.shape != (3, 3):
        raise ValueError(f"rotation must be a 3x3 matrix, got shape {r.shape}.")
    if not np.allclose(r @ r.T, np.eye(3), atol=_ROTATION_TOLERANCE, rtol=0.0):
        raise ValueError("rotation must be orthogonal (R Rᵀ = I).")
    if np.linalg.det(r) < 0:
        raise ValueError(
            "rotation must be proper (det R = +1): an improper rotation "
            "changes the handedness of an optically active medium."
        )
    return r


def _rotate(x: Array, rotation: np.ndarray) -> Array:
    """Return R X Rᵀ for each matrix X of x, shape (N, 3, 3)."""
    if np.array_equal(rotation, np.eye(3)):
        return x
    r = _complex(rotation)
    return be.matmul(be.matmul(r, x), be.transpose(r, (1, 0)))


# ---------------------------------------------------------------------------
# Tensor sources: a constant, a callable or a material
# ---------------------------------------------------------------------------


def _material_to_dict(material: Any) -> dict[str, Any]:
    """Return ``material.to_dict()`` (the base class is not annotated)."""
    data: dict[str, Any] = material.to_dict()
    return data


def _material_from_dict(data: dict[str, Any]) -> BaseMaterial:
    """Return ``BaseMaterial.from_dict(data)`` (the base class is not annotated)."""
    factory: Any = BaseMaterial.from_dict
    material: BaseMaterial = factory(data)
    return material


def _to_list(x: np.ndarray) -> dict[str, list[list[float]]]:
    return {"real": np.real(x).tolist(), "imag": np.imag(x).tolist()}


class _Source:
    """A 3 x 3 tensor of the wavelength, in the crystal frame.

    Args:
        spec: A 3 x 3 constant (array-like, complex allowed), a callable
            ``f(wavelength) -> (3, 3) or (N, 3, 3)`` with the wavelength in µm
            as a 1D backend array, a scalar ``BaseMaterial`` (the tensor
            (n + ik)² I) or a ``BaseTensorMaterial`` (its ε in the global
            frame of that material).
        name: The name of the tensor, for messages.
    """

    def __init__(self, spec: TensorSpec, name: str):
        self.name = name
        self.constant: np.ndarray | None = None
        self.function: Callable[[Array], Array] | None = None
        self.material: BaseMaterial | None = None
        if isinstance(spec, BaseMaterial):
            self.material = spec
        elif callable(spec):
            self.function = spec
        else:
            c = np.asarray(_numpy(spec))
            c = c.astype(np.complex128)
            if c.ndim == 0:
                c = c * np.eye(3)
            if c.shape != (3, 3):
                raise ValueError(
                    f"{name} must be a number, a 3x3 tensor, a callable or a "
                    f"material; got shape {c.shape}."
                )
            self.constant = c

    def __call__(self, w: Array) -> Array:
        n = int(be.size(w))
        if self.material is not None:
            if isinstance(self.material, BaseTensorMaterial):
                return self.material.epsilon(w)
            index = _complex_index(self.material, w)
            return _identity(n) * (index * index)[:, None, None]
        if self.function is not None:
            value = _complex(self.function(w))
        else:
            value = _complex(self.constant)
        shape = tuple(value.shape)
        if shape not in ((3, 3), (n, 3, 3)):
            raise ValueError(
                f"{self.name} must evaluate to shape (3, 3) or ({n}, 3, 3), "
                f"got {shape}."
            )
        return be.to_complex(be.zeros((n, 3, 3))) + value

    def to_dict(self) -> dict[str, Any]:
        if self.material is not None:
            return {"material": _material_to_dict(self.material)}
        if self.constant is not None:
            return {"constant": _to_list(self.constant)}
        raise TypeError(
            f"{self.name} is a callable; a material with a callable tensor "
            "cannot be serialized."
        )

    @staticmethod
    def from_dict(data: dict[str, Any] | None, name: str) -> _Source | None:
        if data is None:
            return None
        if "material" in data:
            return _Source(_material_from_dict(data["material"]), name)
        c = data["constant"]
        return _Source(np.asarray(c["real"]) + 1j * np.asarray(c["imag"]), name)


# ---------------------------------------------------------------------------
# Public helpers
# ---------------------------------------------------------------------------


def euler_zxz_matrix(phi: float, theta: float, psi: float) -> np.ndarray:
    """Return R = Rz(φ) Rx(θ) Rz(ψ), the z-x'-z'' proper Euler rotation.

    A tensor in the crystal frame goes to the global frame as R X Rᵀ. The
    inverse rotation is (-ψ, -θ, -φ).

    Args:
        phi: First rotation about z, in degrees.
        theta: Rotation about the new x axis, in degrees.
        psi: Second rotation about the new z axis, in degrees.

    Returns:
        np.ndarray: The 3 x 3 rotation matrix (float64). It is a constant of
        the material, not a backend array.
    """

    def rz(a: float) -> np.ndarray:
        c, s = math.cos(a), math.sin(a)
        return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])

    def rx(a: float) -> np.ndarray:
        c, s = math.cos(a), math.sin(a)
        return np.array([[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]])

    p, t, q = (math.radians(a) for a in (phi, theta, psi))
    result: np.ndarray = rz(p) @ rx(t) @ rz(q)
    return result


def alpha_from_gyration(gyration: Array) -> Array:
    """Return the Tellegen α of a gyration (Landau) tensor γ.

    α = (tr γ / 2) I - γᵀ. Then ξ = iα, ζ = -iαᵀ and ε = ε_L + α αᵀ give the
    same bulk plane waves as D = ε_L E + i(γk) x E, B' = H'.

    Args:
        gyration: γ, shape (3, 3) or (N, 3, 3), dimensionless (k in units
            of k0 = 2π/λ).

    Returns:
        be.ndarray: α, the same shape as ``gyration``.
    """
    g = be.asarray(gyration)
    half_trace = _trace(g) / 2
    return half_trace[..., None, None] * be.eye(3) - be.transpose(
        g, (*range(g.ndim - 2), g.ndim - 1, g.ndim - 2)
    )


def gyration_from_alpha(alpha: Array) -> Array:
    """Return the gyration (Landau) tensor γ of a Tellegen α.

    γ = (tr α) I - αᵀ, the inverse of :func:`alpha_from_gyration`.

    Args:
        alpha: α, shape (3, 3) or (N, 3, 3).

    Returns:
        be.ndarray: γ, the same shape as ``alpha``.
    """
    a = be.asarray(alpha)
    return _trace(a)[..., None, None] * be.eye(3) - be.transpose(
        a, (*range(a.ndim - 2), a.ndim - 1, a.ndim - 2)
    )


def kappa_from_rotatory_power(
    rotatory_power: float | Array, wavelength: float | Array
) -> Array:
    """Return the chirality κ of a rotatory power.

    κ = ρ λ0 / (2π), with λ0 the **vacuum** wavelength: a linear polarization
    turns by ρ d = k0 κ d over a length d. For quartz, ρ = 21.7283 °/mm at
    λ0 = 0.5894050191 µm gives κ = 3.5574359E-05.

    Args:
        rotatory_power: ρ in degrees per mm (positive for a dextrorotatory
            medium).
        wavelength: The vacuum wavelength in µm.

    Returns:
        be.ndarray: κ (dimensionless).
    """
    rho = be.asarray(rotatory_power) * (math.pi / 180.0)  # rad/mm
    wavelength_mm = be.asarray(wavelength) * 1e-3
    return rho * wavelength_mm / (2 * math.pi)


def quartz_alpha(kappa: float | Array, gyration_ratio: float = -0.525) -> Array:
    """Return the Tellegen α of a uniaxial crystal of class 32 (quartz).

    α = diag(κ, κ, γ11 - κ) in the crystal frame (optic axis z), with
    κ = γ33/2 = g33/(2 n_o) and γ11 = ``gyration_ratio`` γ33. Along the optic
    axis the modes are circular with n_{L,R} = √(n_o² + κ²) ± κ.

    Args:
        kappa: κ, a scalar or shape (N,).
        gyration_ratio: γ11/γ33; the default -0.525 is from Arteaga et al.
            (2012).

    Returns:
        be.ndarray: α, shape (3, 3) or (N, 3, 3).
    """
    k = be.asarray(kappa)[..., None, None]
    transverse = be.asarray(np.diag([1.0, 1.0, 0.0]))
    axial = be.asarray(np.diag([0.0, 0.0, 1.0]))
    return k * transverse + (k * (2 * gyration_ratio - 1)) * axial


def standard_air_index(wavelength: float | Array) -> Array:
    """Return the refractive index of standard dry air.

    Ciddor (1996), Eq. (1): n - 1 = 1E-8 (5792105 / (238.0185 - σ²)
    + 167917 / (57.362 - σ²)), σ = 1/λ0 in µm⁻¹. Standard air is 15 °C,
    101325 Pa, 0 % humidity and 450 ppm CO2. Use it to change a wavelength
    in air into the vacuum wavelength (λ0 = n λ_air) and back.

    Reference: P. E. Ciddor, "Refractive index of air: new equations for the
    visible and near infrared," Appl. Opt. 35, 1566-1573 (1996).

    Args:
        wavelength: The vacuum wavelength λ0 in µm.

    Returns:
        be.ndarray: n of standard air.
    """
    s2 = 1.0 / be.asarray(wavelength) ** 2
    return 1.0 + 1e-8 * (5792105.0 / (238.0185 - s2) + 167917.0 / (57.362 - s2))


def quartz_rotatory_power(wavelength: float | Array) -> Array:
    """Return the rotatory power of right quartz along its optic axis.

    Lowry and Coode-Adams (1927), formula (vi) (p. 395), at 20 °C:
    ρ = 9.5639 / (λ² - 0.0127493) - 2.3113 / (λ² - 0.000974) - 0.1905 °/mm,
    with λ the wavelength in air in µm. The formula agrees with the
    measurements from 0.2373 µm to 2.5 µm; the authors give ±0.002 °/mm in the
    visible. This function takes the vacuum wavelength λ0 and uses
    λ = λ0 / n_air(λ0) (:func:`standard_air_index`).

    Right quartz is dextrorotatory: ρ > 0 and κ > 0
    (:func:`kappa_from_rotatory_power`). Left quartz has -ρ.

    Reference: T. M. Lowry, W. R. C. Coode-Adams, "Optical rotatory
    dispersion. Part III. The rotatory dispersion of quartz in the
    infra-red, visible and ultra-violet regions of the spectrum," Phil.
    Trans. R. Soc. A 226, 391-466 (1927).

    Args:
        wavelength: The vacuum wavelength λ0 in µm.

    Returns:
        be.ndarray: ρ in degrees per mm.
    """
    w = be.asarray(wavelength)
    l2 = (w / standard_air_index(w)) ** 2
    return 9.5639 / (l2 - 0.0127493) - 2.3113 / (l2 - 0.000974) - 0.1905


# ---------------------------------------------------------------------------
# Materials
# ---------------------------------------------------------------------------


class BaseTensorMaterial(BaseMaterial):  # type: ignore[no-untyped-call]
    """Base class for materials described by 3 x 3 constitutive tensors.

    Subclasses implement ``_crystal_tensors``. The public methods give the
    tensors in the global frame, shape (N, 3, 3), complex, for N wavelengths.

    A tensor material has no single refractive index: ``n`` and ``k`` raise
    ``TypeError``.

    Args:
        rotation: The crystal-to-global rotation R (a proper 3 x 3 rotation).
            None is the identity.
        propagation_model: The propagation model. None gives the default
            ``AnisotropicPropagation``: rays move along their ray direction
            with the phase and the attenuation of their wave vector.
    """

    def __init__(
        self,
        rotation: Any = None,
        propagation_model: BasePropagationModel | None = None,
    ):
        if propagation_model is None:
            propagation_model = AnisotropicPropagation(self)
        super().__init__(propagation_model)
        self.rotation = _check_rotation(rotation)

    # -- scalar interface --------------------------------------------------

    def _calculate_n(self, wavelength: float | Array, **kwargs: Any) -> Array:
        raise TypeError(
            f"{type(self).__name__} is a tensor material and has no single "
            "refractive index; use epsilon() or constitutive_6x6()."
        )

    def _calculate_k(self, wavelength: float | Array, **kwargs: Any) -> Array:
        raise TypeError(
            f"{type(self).__name__} is a tensor material and has no single "
            "extinction coefficient; use epsilon() or constitutive_6x6()."
        )

    # -- tensors -----------------------------------------------------------

    def _crystal_tensors(self, w: Array) -> tuple[Array, Array, Array, Array]:
        """Return (ε, μ, ξ, ζ) in the crystal frame at the wavelengths w.

        Args:
            w: The wavelengths in µm, a 1D backend array of length N.

        Returns:
            Four complex backend arrays of shape (N, 3, 3).
        """
        raise NotImplementedError  # pragma: no cover

    def tensors(self, wavelength: float | Array) -> tuple[Array, Array, Array, Array]:
        """Return (ε, μ, ξ, ζ) in the global frame.

        Args:
            wavelength: The wavelength(s) in µm, a scalar or an array of N
                values.

        Returns:
            Four complex backend arrays of shape (N, 3, 3).
        """
        w = _wavelengths(wavelength)
        eps, mu, xi, zeta = self._crystal_tensors(w)
        return (
            _rotate(eps, self.rotation),
            _rotate(mu, self.rotation),
            _rotate(xi, self.rotation),
            _rotate(zeta, self.rotation),
        )

    def epsilon(self, wavelength: float | Array) -> Array:
        """Return the relative permittivity tensor ε in the global frame.

        Args:
            wavelength: The wavelength(s) in µm.

        Returns:
            be.ndarray: Complex, shape (N, 3, 3).
        """
        return self.tensors(wavelength)[0]

    def mu(self, wavelength: float | Array) -> Array:
        """Return the relative permeability tensor μ in the global frame.

        Args:
            wavelength: The wavelength(s) in µm.

        Returns:
            be.ndarray: Complex, shape (N, 3, 3).
        """
        return self.tensors(wavelength)[1]

    def xi(self, wavelength: float | Array) -> Array:
        """Return the magnetoelectric tensor ξ (D' = εE + ξH') in the global frame.

        Args:
            wavelength: The wavelength(s) in µm.

        Returns:
            be.ndarray: Complex, shape (N, 3, 3).
        """
        return self.tensors(wavelength)[2]

    def zeta(self, wavelength: float | Array) -> Array:
        """Return the magnetoelectric tensor ζ (B' = ζE + μH') in the global frame.

        Args:
            wavelength: The wavelength(s) in µm.

        Returns:
            be.ndarray: Complex, shape (N, 3, 3).
        """
        return self.tensors(wavelength)[3]

    def constitutive_6x6(self, wavelength: float | Array) -> Array:
        """Return the 6 x 6 constitutive matrix [[ε, ξ], [ζ, μ]] (global frame).

        (D', B') = M (E, H') with H' = η0 H.

        Args:
            wavelength: The wavelength(s) in µm.

        Returns:
            be.ndarray: Complex, shape (N, 6, 6).
        """
        eps, mu, xi, zeta = self.tensors(wavelength)
        m = be.to_complex(be.zeros((eps.shape[0], 6, 6)))
        m[:, :3, :3] = eps
        m[:, :3, 3:] = xi
        m[:, 3:, :3] = zeta
        m[:, 3:, 3:] = mu
        return m

    def post_constraint(self, wavelength: float | Array) -> Array:
        """Return tr(μ⁻¹(ξ + ζ)), which is zero for a medium that obeys Post.

        Args:
            wavelength: The wavelength(s) in µm.

        Returns:
            be.ndarray: Complex, shape (N,).
        """
        eps, mu, xi, zeta = (be.to_numpy(x) for x in self.tensors(wavelength))
        value = np.trace(np.linalg.solve(mu, xi + zeta), axis1=-2, axis2=-1)
        return _complex(value)

    # -- classification ----------------------------------------------------

    def _numpy_blocks(self, wavelength: float | Array) -> list[np.ndarray]:
        return [np.asarray(be.to_numpy(x)) for x in self.tensors(wavelength)]

    @staticmethod
    def _scale(blocks: list[np.ndarray]) -> float:
        return max(1.0, max(float(np.max(np.abs(b), initial=0.0)) for b in blocks))

    def is_isotropic(self, wavelength: float | Array, tol: float = 1e-12) -> bool:
        """Return True if ε, μ, ξ and ζ are multiples of I at every wavelength.

        A Pasteur (isotropic chiral) medium is isotropic in this sense.

        Args:
            wavelength: The wavelength(s) in µm.
            tol: The tolerance, relative to max(1, the largest element).

        Returns:
            bool: The result.
        """
        blocks = self._numpy_blocks(wavelength)
        limit = tol * self._scale(blocks)
        for b in blocks:
            mean = np.trace(b, axis1=-2, axis2=-1) / 3
            if np.max(np.abs(b - mean[:, None, None] * np.eye(3))) > limit:
                return False
        return True

    def is_transparent(self, wavelength: float | Array, tol: float = 1e-12) -> bool:
        """Return True if the medium is lossless: [[ε, ξ], [ζ, μ]] is Hermitian.

        That is ε = εᴴ, μ = μᴴ and ζ = ξᴴ at every wavelength.

        Args:
            wavelength: The wavelength(s) in µm.
            tol: The tolerance, relative to max(1, the largest element).

        Returns:
            bool: The result.
        """
        m = np.asarray(be.to_numpy(self.constitutive_6x6(wavelength)))
        limit = tol * max(1.0, float(np.max(np.abs(m))))
        return bool(np.max(np.abs(m - np.conj(np.swapaxes(m, -1, -2)))) <= limit)

    def is_reciprocal(self, wavelength: float | Array, tol: float = 1e-12) -> bool:
        """Return True if ε = εᵀ, μ = μᵀ and ξ = -ζᵀ at every wavelength.

        Args:
            wavelength: The wavelength(s) in µm.
            tol: The tolerance, relative to max(1, the largest element).

        Returns:
            bool: The result.
        """
        eps, mu, xi, zeta = self._numpy_blocks(wavelength)
        limit = tol * self._scale([eps, mu, xi, zeta])
        residuals = (
            eps - np.swapaxes(eps, -1, -2),
            mu - np.swapaxes(mu, -1, -2),
            xi + np.swapaxes(zeta, -1, -2),
        )
        return all(float(np.max(np.abs(r))) <= limit for r in residuals)

    # -- serialization -----------------------------------------------------

    def __eq__(self, value: object) -> bool:
        if value is self:
            return True
        if not isinstance(value, type(self)):
            return False
        try:
            return bool(value.to_dict() == self.to_dict())
        except TypeError:
            return False

    __hash__ = None  # type: ignore[assignment]

    def to_dict(self) -> dict[str, Any]:
        """Return a dictionary representation of the material.

        Returns:
            dict: The representation. It holds the rotation as a nested list.
        """
        data: dict[str, Any] = _material_to_dict(super())
        data["rotation"] = self.rotation.tolist()
        return data


class UniaxialMaterial(BaseTensorMaterial):  # type: ignore[no-untyped-call]
    """A uniaxial crystal from two scalar materials and an optic axis.

    ε = ε_o I + (ε_e - ε_o) ĉ ĉᵀ with ε_o = (n_o + ik_o)², ε_e = (n_e + ik_e)²
    and ĉ the unit optic axis in the global frame (Lekner 1991, Eq. (18)).

    Args:
        ordinary: The material of the ordinary index, for example
            ``Material("CaCO3", reference="Ghosh-o")``.
        extraordinary: The material of the extraordinary index, for example
            ``Material("CaCO3", reference="Ghosh-e")``.
        optic_axis: The optic axis in the global frame (normalized here).
        propagation_model: The propagation model.
    """

    def __init__(
        self,
        ordinary: BaseMaterial,
        extraordinary: BaseMaterial,
        optic_axis: Any = (0.0, 0.0, 1.0),
        propagation_model: BasePropagationModel | None = None,
    ):
        super().__init__(None, propagation_model)
        axis = np.asarray(_numpy(optic_axis), dtype=np.float64).reshape(-1)
        norm = float(np.linalg.norm(axis))
        if axis.shape != (3,) or norm == 0.0:
            raise ValueError("optic_axis must be a nonzero 3-vector.")
        self.ordinary = ordinary
        self.extraordinary = extraordinary
        self._axis_input = axis
        self.optic_axis = axis / norm

    def _cache_state(self) -> tuple[Any, ...] | None:
        return None

    def _crystal_tensors(self, w: Array) -> tuple[Array, Array, Array, Array]:
        n = int(be.size(w))
        n_o = _complex_index(self.ordinary, w)
        n_e = _complex_index(self.extraordinary, w)
        eps_o = n_o * n_o
        delta = n_e * n_e - eps_o
        cc = _complex(np.outer(self.optic_axis, self.optic_axis))
        identity = _identity(n)
        eps = identity * eps_o[:, None, None] + delta[:, None, None] * cc
        zero = identity * 0
        return eps, identity, zero, zero

    def to_dict(self) -> dict[str, Any]:
        """Return a dictionary representation of the material.

        Returns:
            dict: The representation.
        """
        data = super().to_dict()
        data.update(
            {
                "ordinary": _material_to_dict(self.ordinary),
                "extraordinary": _material_to_dict(self.extraordinary),
                "optic_axis": self._axis_input.tolist(),
            }
        )
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> UniaxialMaterial:
        """Create the material from its dictionary representation.

        Args:
            data: The dictionary of :meth:`to_dict`.

        Returns:
            UniaxialMaterial: The material.
        """
        return cls(
            _material_from_dict(data["ordinary"]),
            _material_from_dict(data["extraordinary"]),
            data.get("optic_axis", (0.0, 0.0, 1.0)),
        )


class BiaxialMaterial(BaseTensorMaterial):  # type: ignore[no-untyped-call]
    """A biaxial crystal from three scalar materials and an orientation.

    ε = R diag(ñ_x², ñ_y², ñ_z²) Rᵀ with ñ = n + ik of the principal
    materials along the crystal axes x, y, z. For KTP use
    ``Material("KTiOPO4", reference="Kato-alpha")``, ``-beta``, ``-gamma``.
    In the frame n_x < n_y < n_z the two optic axes (binormals) lie in the x-z
    plane at ±V from z, tan V = (n_z/n_x) √((n_y² - n_x²)/(n_z² - n_y²)).

    Args:
        material_x: The principal material along the crystal x axis.
        material_y: The principal material along the crystal y axis.
        material_z: The principal material along the crystal z axis.
        rotation: The crystal-to-global rotation R. None is the identity.
        propagation_model: The propagation model.
    """

    def __init__(
        self,
        material_x: BaseMaterial,
        material_y: BaseMaterial,
        material_z: BaseMaterial,
        rotation: Any = None,
        propagation_model: BasePropagationModel | None = None,
    ):
        super().__init__(rotation, propagation_model)
        self.material_x = material_x
        self.material_y = material_y
        self.material_z = material_z

    def _cache_state(self) -> tuple[Any, ...] | None:
        return None

    def _crystal_tensors(self, w: Array) -> tuple[Array, Array, Array, Array]:
        n = int(be.size(w))
        eps = be.to_complex(be.zeros((n, 3, 3)))
        for i, material in enumerate(
            (self.material_x, self.material_y, self.material_z)
        ):
            index = _complex_index(material, w)
            eps[:, i, i] = index * index
        identity = _identity(n)
        zero = identity * 0
        return eps, identity, zero, zero

    def to_dict(self) -> dict[str, Any]:
        """Return a dictionary representation of the material.

        Returns:
            dict: The representation.
        """
        data = super().to_dict()
        data.update(
            {
                "material_x": _material_to_dict(self.material_x),
                "material_y": _material_to_dict(self.material_y),
                "material_z": _material_to_dict(self.material_z),
            }
        )
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> BiaxialMaterial:
        """Create the material from its dictionary representation.

        Args:
            data: The dictionary of :meth:`to_dict`.

        Returns:
            BiaxialMaterial: The material.
        """
        return cls(
            _material_from_dict(data["material_x"]),
            _material_from_dict(data["material_y"]),
            _material_from_dict(data["material_z"]),
            rotation=data.get("rotation"),
        )


class TensorMaterial(BaseTensorMaterial):  # type: ignore[no-untyped-call]
    """A general anisotropic material: ε and μ from constants or callables.

    Each tensor is given in the crystal frame as a 3 x 3 constant (complex
    allowed), a callable ``f(wavelength) -> (3, 3) or (N, 3, 3)`` (the
    wavelength in µm as a 1D backend array), a scalar material (the tensor
    (n + ik)² I) or a tensor material (its ε).

    Args:
        epsilon: The relative permittivity ε.
        mu: The relative permeability μ. None is I.
        rotation: The crystal-to-global rotation R. None is the identity.
        propagation_model: The propagation model.
    """

    def __init__(
        self,
        epsilon: TensorSpec,
        mu: TensorSpec | None = None,
        rotation: Any = None,
        propagation_model: BasePropagationModel | None = None,
    ):
        super().__init__(rotation, propagation_model)
        self._epsilon = _Source(epsilon, "epsilon")
        self._mu = None if mu is None else _Source(mu, "mu")

    def _cache_state(self) -> tuple[Any, ...] | None:
        return None

    def _magnetoelectric(self, w: Array, eps: Array) -> tuple[Array, Array, Array]:
        """Return (ε, ξ, ζ) in the crystal frame; the base has ξ = ζ = 0."""
        zero = eps * 0
        return eps, zero, zero

    def _crystal_tensors(self, w: Array) -> tuple[Array, Array, Array, Array]:
        n = int(be.size(w))
        eps = self._epsilon(w)
        mu = _identity(n) if self._mu is None else self._mu(w)
        eps, xi, zeta = self._magnetoelectric(w, eps)
        return eps, mu, xi, zeta

    def to_dict(self) -> dict[str, Any]:
        """Return a dictionary representation of the material.

        Returns:
            dict: The representation.

        Raises:
            TypeError: If a tensor is a callable.
        """
        data = super().to_dict()
        data["epsilon"] = self._epsilon.to_dict()
        data["mu"] = None if self._mu is None else self._mu.to_dict()
        return data

    @staticmethod
    def _spec(data: dict[str, Any] | None, name: str) -> Any:
        source = _Source.from_dict(data, name)
        if source is None:
            return None
        return source.material if source.material is not None else source.constant

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> TensorMaterial:
        """Create the material from its dictionary representation.

        Args:
            data: The dictionary of :meth:`to_dict`.

        Returns:
            TensorMaterial: The material.
        """
        return cls(
            cls._spec(data["epsilon"], "epsilon"),
            mu=cls._spec(data.get("mu"), "mu"),
            rotation=data.get("rotation"),
        )


class BianisotropicMaterial(TensorMaterial):  # type: ignore[no-untyped-call]
    """A bianisotropic material: ε, μ, ξ and ζ in the Tellegen form.

    D' = εE + ξH', B' = ζE + μH' (H' = η0 H). Give ξ and ζ directly, or give
    optical activity with :meth:`from_optical_activity` or
    :meth:`from_gyration` (ξ = iα, ζ = -iαᵀ).

    Args:
        epsilon: The relative permittivity ε (see :class:`TensorMaterial`).
        xi: ξ. None is 0.
        zeta: ζ. None is 0.
        mu: The relative permeability μ. None is I.
        rotation: The crystal-to-global rotation R. None is the identity.
        propagation_model: The propagation model.
    """

    def __init__(
        self,
        epsilon: TensorSpec,
        xi: TensorSpec | None = None,
        zeta: TensorSpec | None = None,
        mu: TensorSpec | None = None,
        rotation: Any = None,
        propagation_model: BasePropagationModel | None = None,
    ):
        super().__init__(epsilon, mu, rotation, propagation_model)
        self._xi = None if xi is None else _Source(xi, "xi")
        self._zeta = None if zeta is None else _Source(zeta, "zeta")
        self._alpha: _Source | None = None
        self._add_alpha_squared = False

    @classmethod
    def from_optical_activity(
        cls,
        epsilon: TensorSpec,
        alpha: TensorSpec,
        epsilon_form: str = "landau",
        mu: TensorSpec | None = None,
        rotation: Any = None,
        propagation_model: BasePropagationModel | None = None,
    ) -> BianisotropicMaterial:
        """Create an optically active material from a Tellegen α.

        ξ = iα and ζ = -iαᵀ. A Pasteur medium is α = κI; quartz is
        :func:`quartz_alpha`. With ``epsilon_form="landau"`` the given ε is
        ε_L (the permittivity of the gyration form, as the measured n_o and
        n_e) and the material uses ε = ε_L + α αᵀ. With
        ``epsilon_form="tellegen"`` the given ε is used as is.

        Args:
            epsilon: ε_L or ε (see :class:`TensorMaterial`).
            alpha: α in the crystal frame: a real 3 x 3 constant or a
                callable of the wavelength (µm).
            epsilon_form: "landau" (default) or "tellegen".
            mu: The relative permeability μ. None is I.
            rotation: The crystal-to-global rotation R.
            propagation_model: The propagation model.

        Returns:
            BianisotropicMaterial: The material.
        """
        if epsilon_form not in ("landau", "tellegen"):
            raise ValueError(
                f"epsilon_form must be 'landau' or 'tellegen', got {epsilon_form!r}."
            )
        material = cls(
            epsilon, mu=mu, rotation=rotation, propagation_model=propagation_model
        )
        material._alpha = _Source(alpha, "alpha")
        material._add_alpha_squared = epsilon_form == "landau"
        return material

    @classmethod
    def from_gyration(
        cls,
        epsilon: TensorSpec,
        gyration: Any,
        mu: TensorSpec | None = None,
        rotation: Any = None,
        propagation_model: BasePropagationModel | None = None,
    ) -> BianisotropicMaterial:
        """Create an optically active material from a gyration (Landau) tensor γ.

        α = (tr γ/2) I - γᵀ (:func:`alpha_from_gyration`), ξ = iα, ζ = -iαᵀ and
        ε = ε_L + α αᵀ, with ε_L the given ε.

        Args:
            epsilon: ε_L (see :class:`TensorMaterial`).
            gyration: γ in the crystal frame: a real 3 x 3 constant or a
                callable of the wavelength (µm).
            mu: The relative permeability μ. None is I.
            rotation: The crystal-to-global rotation R.
            propagation_model: The propagation model.

        Returns:
            BianisotropicMaterial: The material.
        """
        if callable(gyration):

            def alpha(w: Array, g: Callable[[Array], Array] = gyration) -> Array:
                return alpha_from_gyration(g(w))

            alpha_spec: Any = alpha
        else:
            alpha_spec = np.asarray(
                be.to_numpy(alpha_from_gyration(np.asarray(gyration, dtype=float)))
            )
        return cls.from_optical_activity(
            epsilon, alpha_spec, "landau", mu, rotation, propagation_model
        )

    def _magnetoelectric(self, w: Array, eps: Array) -> tuple[Array, Array, Array]:
        if self._alpha is not None:
            alpha = self._alpha(w)
            alpha_t = _swap(alpha)
            if self._add_alpha_squared:
                eps = eps + be.matmul(alpha, alpha_t)
            return eps, 1j * alpha, -1j * alpha_t
        zero = eps * 0
        xi = zero if self._xi is None else self._xi(w)
        zeta = zero if self._zeta is None else self._zeta(w)
        return eps, xi, zeta

    def to_dict(self) -> dict[str, Any]:
        """Return a dictionary representation of the material.

        Returns:
            dict: The representation.

        Raises:
            TypeError: If a tensor is a callable.
        """
        data = super().to_dict()
        if self._alpha is not None:
            data["alpha"] = self._alpha.to_dict()
            data["epsilon_form"] = "landau" if self._add_alpha_squared else "tellegen"
        else:
            data["xi"] = None if self._xi is None else self._xi.to_dict()
            data["zeta"] = None if self._zeta is None else self._zeta.to_dict()
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> BianisotropicMaterial:
        """Create the material from its dictionary representation.

        Args:
            data: The dictionary of :meth:`to_dict`.

        Returns:
            BianisotropicMaterial: The material.
        """
        epsilon = cls._spec(data["epsilon"], "epsilon")
        mu = cls._spec(data.get("mu"), "mu")
        rotation = data.get("rotation")
        if "alpha" in data:
            alpha = cls._spec(data["alpha"], "alpha")
            return cls.from_optical_activity(
                epsilon,
                alpha,
                data.get("epsilon_form", "landau"),
                mu,
                rotation,
            )
        return cls(
            epsilon,
            xi=cls._spec(data.get("xi"), "xi"),
            zeta=cls._spec(data.get("zeta"), "zeta"),
            mu=mu,
            rotation=rotation,
        )


class QuartzMaterial(BianisotropicMaterial):  # type: ignore[no-untyped-call]
    """α-quartz (class 32) with its optical activity.

    ε_L = diag(n_o², n_o², n_e²) from two scalar materials (default Ghosh
    1999, ``Material("SiO2", reference="Ghosh-o")`` and ``"Ghosh-e"``,
    0.198-2.05 µm), the rotatory power of :func:`quartz_rotatory_power`
    (Lowry and Coode-Adams 1927, 20 °C), κ = ± ρ λ0 / (2π)
    (:func:`kappa_from_rotatory_power`), α = :func:`quartz_alpha` (κ,
    ``gyration_ratio``), ξ = iα, ζ = -iαᵀ and ε = ε_L + ααᵀ. The optic axis
    is the crystal z axis; ``rotation`` turns it into the global frame.

    The wavelength is the vacuum wavelength in µm. Along the optic axis the
    modes are circular, n_{L,R} = √(n_o² + κ²) ± κ, and a linear polarization
    turns by ρ d (right quartz: clockwise as seen by an observer who looks
    toward the source).

    Args:
        hand: ``"right"`` (dextrorotatory, κ > 0) or ``"left"`` (κ < 0).
        rotation: The crystal-to-global rotation R. None is the identity.
        gyration_ratio: γ11/γ33; the default -0.525 is from Arteaga et al.
            (2012).
        ordinary: The material of n_o. None is Ghosh 1999.
        extraordinary: The material of n_e. None is Ghosh 1999.
        propagation_model: The propagation model.
    """

    def __init__(
        self,
        hand: str = "right",
        rotation: Any = None,
        gyration_ratio: float = -0.525,
        ordinary: BaseMaterial | None = None,
        extraordinary: BaseMaterial | None = None,
        propagation_model: BasePropagationModel | None = None,
    ):
        if hand not in ("right", "left"):
            raise ValueError(f"hand must be 'right' or 'left', got {hand!r}.")
        from optiland.materials.material import Material

        if ordinary is None:
            ordinary = Material("SiO2", reference="Ghosh-o")
        if extraordinary is None:
            extraordinary = Material("SiO2", reference="Ghosh-e")
        super().__init__(
            UniaxialMaterial(ordinary, extraordinary),
            rotation=rotation,
            propagation_model=propagation_model,
        )
        self.hand = hand
        self.gyration_ratio = float(gyration_ratio)
        self.ordinary = ordinary
        self.extraordinary = extraordinary
        self._alpha = _Source(self._quartz_alpha, "alpha")
        self._add_alpha_squared = True

    def _quartz_alpha(self, w: Array) -> Array:
        sign = 1.0 if self.hand == "right" else -1.0
        kappa = sign * kappa_from_rotatory_power(quartz_rotatory_power(w), w)
        return quartz_alpha(kappa, self.gyration_ratio)

    def kappa(self, wavelength: float | Array) -> Array:
        """Return κ = ± ρ λ0 / (2π), + for right quartz.

        Args:
            wavelength: The vacuum wavelength(s) in µm.

        Returns:
            be.ndarray: κ, shape (N,).
        """
        w = _wavelengths(wavelength)
        sign = 1.0 if self.hand == "right" else -1.0
        return sign * kappa_from_rotatory_power(quartz_rotatory_power(w), w)

    def to_dict(self) -> dict[str, Any]:
        """Return a dictionary representation of the material.

        Returns:
            dict: The representation.
        """
        data = BaseTensorMaterial.to_dict(self)
        data.update(
            {
                "hand": self.hand,
                "gyration_ratio": self.gyration_ratio,
                "ordinary": _material_to_dict(self.ordinary),
                "extraordinary": _material_to_dict(self.extraordinary),
            }
        )
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> QuartzMaterial:
        """Create the material from its dictionary representation.

        Args:
            data: The dictionary of :meth:`to_dict`.

        Returns:
            QuartzMaterial: The material.
        """
        return cls(
            hand=data.get("hand", "right"),
            rotation=data.get("rotation"),
            gyration_ratio=data.get("gyration_ratio", -0.525),
            ordinary=_material_from_dict(data["ordinary"]),
            extraordinary=_material_from_dict(data["extraordinary"]),
        )
