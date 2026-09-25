"""Polarized Rays

This module contains the `PolarizedRays` class, which represents a class for
polarized rays in three-dimensional space. The class inherits from the
`RealRays` class.

Kramer Harrison, 2024
"""

from __future__ import annotations

import optiland.backend as be
from optiland.rays.polarization_state import PolarizationState
from optiland.rays.real_rays import RealRays


class PolarizedRays(RealRays):
    """Represents a class for polarized rays in three-dimensional space.

    Inherits from the `RealRays` class.

    Attributes:
        x (ndarray): The x-coordinates of the rays.
        y (ndarray): The y-coordinates of the rays.
        z (ndarray): The z-coordinates of the rays.
        L (ndarray): The x-components of the direction vectors of the rays.
        M (ndarray): The y-components of the direction vectors of the rays.
        N (ndarray): The z-components of the direction vectors of the rays.
        i (ndarray): The intensity of the rays.
        w (ndarray): The wavelength of the rays.
        opd (ndarray): The optical path length of the rays.
        p (be.ndarray): Array of polarization matrices of the rays.
        q (be.ndarray): Array of the geometrical transformations of the rays:
            the PRT matrices of the same paths without polarization effects
            (the parallel transport of the local frames; Yun, McClain and
            Chipman, Appl. Opt. 50, 2866 (2011)). Used for the retardance.
        flux_factor (be.ndarray): Product of the power factors
            Re(n' cos θ') / Re(n cos θ) of the refractions on the path. The
            PRT matrix maps field amplitudes; the ray power is |P E|**2 times
            this factor.

    Methods:
        get_output_field(E: be.ndarray) -> be.ndarray:
            Compute the output electric field given the input electric field.
        update_intensity(state: PolarizationState):
            Update the ray intensity based on the polarization state.
        update(jones_matrix: be.ndarray = None):
            Update the polarization matrices after interaction with a surface.
        _get_3d_electric_field(state: PolarizationState) -> be.ndarray:
            Get the 3D electric fields given the polarization state and
            initial rays.

    """

    def __init__(self, x, y, z, L, M, N, intensity, wavelength):
        super().__init__(x, y, z, L, M, N, intensity, wavelength)

        self.p = be.tile(be.eye(3), (be.size(self.x), 1, 1))
        self.q = be.tile(be.eye(3), (be.size(self.x), 1, 1))
        self.flux_factor = be.ones_like(self.x)
        self._i0 = be.copy(intensity)
        self._L0 = be.copy(L)
        self._M0 = be.copy(M)
        self._N0 = be.copy(N)

    def _rotate_matrices(self, rotation: be.ndarray) -> None:
        """Express the PRT matrices in a rotated frame: P <- R P.

        The PRT matrix maps the launch field (global frame) to the current
        field. The coordinate system of a surface rotates the rays into its
        local frame before the interaction (whose matrix is local) and back
        after it, so the output side of P must turn with the rays. The
        geometrical transformation ``q`` (if the rays carry it) turns alike.

        Args:
            rotation: The rotation matrix R, shape (3, 3).
        """
        # R + 0 P has the dtype of P (real or complex), on both backends.
        self.p = be.matmul(rotation + 0 * self.p, self.p)
        if hasattr(self, "q"):
            self.q = be.matmul(rotation + 0 * self.q, self.q)

    @staticmethod
    def _rotation(axis: int, angle) -> be.ndarray:
        """Return the matrix of a rotation by ``angle`` about a coordinate axis.

        The same rotation as ``RealRays.rotate_x/y/z`` applies to (L, M, N).
        """
        angle = be.array(angle)
        c, s = be.cos(angle), be.sin(angle)
        one, zero = be.ones_like(c), be.zeros_like(c)
        if axis == 0:
            rows = [[one, zero, zero], [zero, c, -s], [zero, s, c]]
        elif axis == 1:
            rows = [[c, zero, s], [zero, one, zero], [-s, zero, c]]
        else:
            rows = [[c, -s, zero], [s, c, zero], [zero, zero, one]]
        return be.stack([be.stack(row) for row in rows])

    def rotate_x(self, rx):
        """Rotate the rays and their PRT matrices about the x-axis.

        Args:
            rx: Rotation angle around x-axis in radians.
        """
        super().rotate_x(rx)
        self._rotate_matrices(self._rotation(0, rx))

    def rotate_y(self, ry):
        """Rotate the rays and their PRT matrices about the y-axis.

        Args:
            ry: Rotation angle around y-axis in radians.
        """
        super().rotate_y(ry)
        self._rotate_matrices(self._rotation(1, ry))

    def rotate_z(self, rz):
        """Rotate the rays and their PRT matrices about the z-axis.

        Args:
            rz: Rotation angle around z-axis in radians.
        """
        super().rotate_z(rz)
        self._rotate_matrices(self._rotation(2, rz))

    def get_output_field(self, E: be.ndarray) -> be.ndarray:
        """Compute the output electric field given the input electric field.

        Args:
            E (be.ndarray): The input electric field as a numpy array.

        Returns:
            be.ndarray: The computed output electric field as a numpy array.

        """
        return be.mult_p_E(self.p, E)

    def _compute_unscaled_exit_fields(
        self, state: PolarizationState | None
    ) -> list[be.ndarray]:
        """Compute the unscaled exit electric field(s) for the rays.

        Args:
            state (PolarizationState | None): The polarization state.

        Returns:
            list[be.ndarray]: A list of unscaled 3D electric field arrays.
        """
        if state is not None and state.is_polarized:
            E0 = self._get_3d_electric_field(state)
            E1 = self.get_output_field(E0)
            return [E1]
        else:
            state_x = PolarizationState(
                is_polarized=True,
                Ex=1.0,
                Ey=0.0,
                phase_x=0.0,
                phase_y=0.0,
            )
            E0_x = self._get_3d_electric_field(state_x)
            E1_x = self.get_output_field(E0_x)

            state_y = PolarizationState(
                is_polarized=True,
                Ex=0.0,
                Ey=1.0,
                phase_x=0.0,
                phase_y=0.0,
            )
            E0_y = self._get_3d_electric_field(state_y)
            E1_y = self.get_output_field(E0_y)

            return [E1_x, E1_y]

    def get_exit_fields(self, state: PolarizationState | None) -> list[be.ndarray]:
        """Compute the exit electric field(s) for the rays.

        Args:
            state (PolarizationState | None): The polarization state.

        Returns:
            list[be.ndarray]: A list of 3D electric field arrays. For polarized
            light, the list contains a single array. For unpolarized light, the
            list contains two orthogonal, incoherently superimposed arrays, each
            scaled down by 1/sqrt(2).
        """
        fields = self._compute_unscaled_exit_fields(state)
        scale_factor = be.unsqueeze_last(be.sqrt(self._i0 / len(fields)))
        return [E1 * scale_factor for E1 in fields]

    def update_intensity(self, state: PolarizationState):
        """Update ray intensity based on polarization state.

        Multiplies the running intensity (which includes bulk absorption and
        the other scalar losses of the trace) by the polarization transmittance
        |P E|**2 of the path and by :attr:`flux_factor`. Call it once, at the
        end of a trace.

        Args:
            state (PolarizationState): The polarization state of the ray.

        """
        fields = self._compute_unscaled_exit_fields(state)
        intensity = be.zeros_like(self.i)
        for E1 in fields:
            intensity = intensity + be.sum(be.abs(E1) ** 2, axis=1)
        self.i = self.i * self.flux_factor * intensity / len(fields)

    @staticmethod
    def get_local_basis(
        k0: be.ndarray, k1: be.ndarray
    ) -> tuple[be.ndarray, be.ndarray, be.ndarray, be.ndarray]:
        """Get the local s, p0, p1 vectors and transforming matrices.

        Args:
            k0: (N, 3) array of pre-interaction ray directions.
            k1: (N, 3) array of post-interaction ray directions.

        Returns:
            tuple: (s, p0, p1, o_in, o_out) where s, p0, p1 are (N, 3) vectors
            and o_in, o_out are the projection matrices.
        """
        # find s-component
        s = be.cross(k0, k1)
        mag = be.linalg.norm(s, axis=1)

        # handle case when mag = 0 (i.e., k0 parallel to k1)
        mask = mag == 0
        if be.any(mask):
            x = be.broadcast_to(be.array([1.0, 0.0, 0.0]), k0[mask].shape)
            p_fallback = be.cross(k0[mask], x)

            p_norms = be.linalg.norm(p_fallback, axis=1)
            y = be.broadcast_to(be.array([0.0, 1.0, 0.0]), k0[mask].shape)
            p_fallback = be.where(
                be.unsqueeze_last(p_norms == 0), be.cross(k0[mask], y), p_fallback
            )

            s[mask] = be.cross(p_fallback, k0[mask])
            mag = be.linalg.norm(s, axis=1)

        s = s / be.unsqueeze_last(mag)

        # find p-component pre and post surface
        p0 = be.cross(k0, s)
        p1 = be.cross(k1, s)

        # othogonal transformation matrices
        o_in = be.stack((s, p0, k0), axis=1)
        o_out = be.stack((s, p1, k1), axis=2)

        return s, p0, p1, o_in, o_out

    def update(self, jones_matrix: be.ndarray = None, flux_factor: be.ndarray = None):
        """Update polarization matrices after interaction with surface.

        Args:
            jones_matrix (be.ndarray, optional): Jones matrix representing the
                interaction with the surface. If not provided, the
                polarization matrix is computed assuming an identity matrix.
            flux_factor (be.ndarray, optional): Power factor of the surface
                that the Jones matrix does not contain (see
                :meth:`optiland.jones.BaseJones.calculate_flux_factor`). If not
                provided, the factor is 1.

        """
        # merge k-vector components into matrix for speed
        k0 = be.stack([self.L0, self.M0, self.N0]).T
        k1 = be.stack([self.L, self.M, self.N]).T

        s, p0, p1, o_in, o_out = self.get_local_basis(k0, k1)

        # compute polarization matrix for surface
        q = be.matmul(o_out, o_in)
        if jones_matrix is None:
            p = q
        else:
            p = be.batched_chain_matmul3(o_out, jones_matrix, o_in)

        # update polarization matrices of rays
        self.p = be.matmul(p, self.p)
        self.q = be.matmul(q, self.q)
        if flux_factor is not None:
            self.flux_factor = self.flux_factor * flux_factor

    def get_input_basis(self) -> tuple[be.ndarray, be.ndarray]:
        """Get the transverse basis of the input field of each ray.

        The input field is ``Ex s + Ey p`` with ``p = k0 x x / |k0 x x|`` and
        ``s = p x k0``, where ``k0`` is the initial ray direction. For
        ``k0 = z``, ``s = x`` and ``p = y``.

        Returns:
            tuple[be.ndarray, be.ndarray]: (s, p), each of shape (N, 3).

        """
        k = be.stack([self._L0, self._M0, self._N0]).T

        # TODO - efficiently handle case when k parallel to x-axis
        x = be.broadcast_to(be.array([1.0, 0.0, 0.0]), k.shape)
        p = be.cross(k, x)

        norms = be.linalg.norm(p, axis=1)
        if be.any(norms == 0):
            raise ValueError("k-vector parallel to x-axis is not currently supported.")

        p = p / be.unsqueeze_last(norms)

        s = be.cross(p, k)
        return s, p

    def _get_3d_electric_field(self, state: PolarizationState) -> be.ndarray:
        """Get 3D electric fields given polarization state and initial rays.

        Args:
            state (PolarizationState): The polarization state of the rays.

        Returns:
            be.ndarray: The 3D electric fields.

        """
        s, p = self.get_input_basis()

        E = (
            state.Ex * be.exp(1j * state.phase_x) * s
            + state.Ey * be.exp(1j * state.phase_y) * p
        )

        return E
