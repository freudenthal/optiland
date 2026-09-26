"""Ray splitting by branch enumeration

A ray that meets a surface next to an anisotropic medium splits into
eigenmodes (for example the o and e rays of calcite), and every surface splits
it into a reflected and a transmitted part. Optiland traces a fixed number of
rays through a fixed sequence of surfaces, so a sequential trace follows one
child at each surface. This module traces all of them: each **branch** is an
ordinary fixed-count trace that follows one choice at each splitting surface,
and :class:`BranchTracer` enumerates the branches depth first.

The splitting surfaces are:

* every surface with an ``AnisotropicInteractionModel``: one branch per
  transmitted eigenmode of the medium after the surface (``"o"``/``"e"`` for a
  ``UniaxialMaterial``, else ``"slow"``/``"fast"``; one branch ``"T"`` into an
  isotropic medium);
* on request (``ghosts``), the reflected children too: ``"R"`` into an
  isotropic incident medium, or its two eigenmodes. A requested isotropic
  surface (glass, air) is traced with the bare-interface solver of
  ``optiland.anisotropic`` in its branches, so that R + T = 1 for a lossless
  interface.

A reflection reverses the direction of travel: the branch then passes the
earlier surfaces backwards (``optiland.sequences.SurfaceView``). A branch ends
at the image surface (kept), when it leaves through the first surface
backwards (returned), when its power falls below ``threshold`` times the
launch power, or when it would take more than ``max_reflections`` reflections
(pruned).

**Branch key.** One entry per visit to a splitting surface, in trace order:
``(label, "T")`` or ``(label, "R")`` when the child leaves into an isotropic
medium, ``(label, "T", mode)`` or ``(label, "R", mode)`` when it leaves into
an anisotropic medium. The label is the model ``label``, else the surface
comment, else ``"s<index>"``. The key alone gives the path of the branch.

**Power ledger.** Each step of each branch adds its parent power to exactly
one of: the children that the enumeration follows (``kept`` or ``returned`` at
the end of their branch, ``pruned`` if dropped), ``unfollowed`` (the reflected
children of an anisotropic surface that is not in ``ghosts``), ``evanescent``
(rays that a surface loses: an evanescent child, or a total internal
reflection that is not followed), ``clipped`` (aperture) and ``absorbed`` (the
rest of the step: bulk absorption, and the reflection of a coated surface that
is not split). For a lossless system traced with every split followed,
``absorbed`` is 0 to rounding and the fields sum to the launch power.

**Detector sums.** Branches that reach the image surface share the launch
ray grid (ray j of each branch comes from launch ray j). The incoherent sum
adds their powers; the coherent sum adds their fields
E_b = √(I_b g_b) P_b E_0 exp(i k0 OPL_b) (exp(−iωt)), with the phase of each
branch carried to a common point by its local plane wave (k0 Re(k_b) · ΔX),
for example the fringe of a Savart plate.

References:

* G. Yun, K. Crabtree, R. A. Chipman, "Three-dimensional polarization
  ray-tracing calculus I: definition and diattenuation," Appl. Opt. 50,
  2855-2865 (2011): the PRT of split rays and their recombination, Eq. (38).
* S. C. McClain, L. W. Hillman, R. A. Chipman, "Polarization ray tracing in
  anisotropic optically active media. I. Algorithms," J. Opt. Soc. Am. A 10,
  2371-2382 (1993): the ray tree of a crystal system.
"""

from __future__ import annotations

import copy
import math
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import optiland.backend as be
from optiland.coatings import FresnelCoating
from optiland.distribution import create_distribution
from optiland.interactions.anisotropic_model import AnisotropicInteractionModel
from optiland.materials.anisotropic import BaseTensorMaterial, UniaxialMaterial
from optiland.rays.anisotropic_rays import AnisotropicRays
from optiland.sequences.sequenced_optic import SequencedOptic
from optiland.sequences.sequenced_surface_group import SequencedSurfaceGroup
from optiland.sequences.surface_view import SurfaceView, resolve_view_materials

if TYPE_CHECKING:
    from optiland.optic.optic import Optic
    from optiland.rays import PolarizationState
    from optiland.rays.polarized_rays import PolarizedRays
    from optiland.sequences.steps import RawStep as SequenceRawStep

Array = Any
BranchKey = tuple[tuple[str, ...], ...]
RawStep = int | tuple[int, str]

#: The names of the power ledger fields, in the order of :meth:`PowerLedger.as_dict`.
LEDGER_FIELDS = (
    "kept",
    "returned",
    "pruned",
    "unfollowed",
    "evanescent",
    "clipped",
    "absorbed",
)


@dataclass
class PowerLedger:
    """Where the launch power went, as fractions of the launch power.

    Attributes:
        launch: The launch power (the sum of the ray powers at launch).
        kept: Power of the branches that reach the image surface.
        returned: Power of the branches that leave through the first surface
            backwards (into object space).
        pruned: Power of the branches dropped by the power threshold or by the
            reflection limit, at the step where they were dropped.
        unfollowed: Power of the reflected children of anisotropic surfaces
            that are not in ``ghosts``.
        evanescent: Power of rays that a surface loses (an evanescent child;
            a total internal reflection that the enumeration does not follow).
        clipped: Power of rays clipped by an aperture.
        absorbed: The rest: bulk absorption, and the reflection of coated
            surfaces that are not split.
    """

    launch: float
    kept: float = 0.0
    returned: float = 0.0
    pruned: float = 0.0
    unfollowed: float = 0.0
    evanescent: float = 0.0
    clipped: float = 0.0
    absorbed: float = 0.0

    def total(self) -> float:
        """Return the sum of all fields (1 for a closed ledger)."""
        return float(sum(getattr(self, name) for name in LEDGER_FIELDS))

    def as_dict(self) -> dict[str, float]:
        """Return the fields (without ``launch``) as a dictionary."""
        return {name: getattr(self, name) for name in LEDGER_FIELDS}


@dataclass
class Branch:
    """One traced branch.

    Attributes:
        key: The branch key.
        steps: The raw sequence steps of the branch (surface indices and
            ``(index, "reflect")`` pairs; ``optiland.sequences``).
        rays: The rays at the end of the branch, after ``update_intensity``
            (``rays.i`` is the power of each ray).
        running_intensity: ``rays.i`` before ``update_intensity`` (the scalar
            losses of the path; for the exit fields).
        views: The surface views of the path, with the records of each step.
        power: The branch power as a fraction of the launch power.
        terminal: ``"image"`` or ``"returned"``.
    """

    key: BranchKey
    steps: list[RawStep]
    rays: AnisotropicRays
    running_intensity: Array
    views: list[SurfaceView]
    power: float
    terminal: str


@dataclass
class BranchResult:
    """The branches of a trace and the power ledger.

    Attributes:
        branches: ``{key: Branch}`` in depth-first order.
        ledger: The power ledger.
        pruned: ``{key: (power fraction, reason)}`` of the dropped branches;
            the reason is ``"threshold"`` or ``"reflections"``.
        state: The polarization state of the launch (None: unpolarized).
    """

    branches: dict[BranchKey, Branch]
    ledger: PowerLedger
    pruned: dict[BranchKey, tuple[float, str]] = field(default_factory=dict)
    state: PolarizationState | None = None

    def __getitem__(self, key: BranchKey) -> Branch:
        return self.branches[key]

    def __len__(self) -> int:
        return len(self.branches)

    def keys(self) -> list[BranchKey]:
        """Return the branch keys in depth-first order."""
        return list(self.branches)

    def rays(self) -> dict[BranchKey, AnisotropicRays]:
        """Return ``{key: rays}``."""
        return {key: branch.rays for key, branch in self.branches.items()}

    def _select(
        self, keys: list[BranchKey] | None, terminal: str | None = "image"
    ) -> list[Branch]:
        if keys is None:
            return [
                b
                for b in self.branches.values()
                if terminal is None or b.terminal == terminal
            ]
        return [self.branches[key] for key in keys]

    def incoherent_intensity(self, keys: list[BranchKey] | None = None) -> Array:
        """Return the summed power of each launch ray over branches, shape (N,).

        Args:
            keys: The branches to add; None adds every branch that reaches
                the image surface.
        """
        total = None
        for branch in self._select(keys):
            power = be.where(be.isfinite(branch.rays.i), branch.rays.i, 0.0)
            total = power if total is None else total + power
        if total is None:
            raise ValueError("No branch to add.")
        return total

    def exit_fields(self, key: BranchKey) -> list[Array]:
        """Return the exit fields of a branch, one (N, 3) array per input state.

        E = √(I g) P E_0 exp(i k0 OPL): I the running intensity, g the flux
        factor, P the PRT of the path, E_0 the launch field of the state
        (for an unpolarized launch: two fields, for the x and the y input),
        k0 = 2π/λ and OPL the optical path of the ray (exp(−iωt)).
        ``|E|²`` summed over the components is the ray power.
        """
        branch = self.branches[key]
        rays = branch.rays
        fields = rays._compute_unscaled_exit_fields(self.state)
        amplitude = be.sqrt(branch.running_intensity * rays.flux_factor)
        phase = be.exp(1j * _k0_per_length(rays.w) * rays.opd)
        return [f * (amplitude * phase)[:, None] for f in fields]

    def coherent_fields(
        self,
        keys: list[BranchKey] | None = None,
        reference: BranchKey | None = None,
    ) -> list[Array]:
        """Return the summed field of branches at common points.

        Each branch field is carried from its exit point X_b to the exit point
        X_r of the reference branch by its local plane wave:
        E_b exp(i k0 Re(k_b) · (X_r − X_b)). For parallel exit rays this is
        the optical path difference on a common wavefront:
        OPL_a − OPL_b − n ŝ · (X_a − X_b).

        Args:
            keys: The branches to add; None adds every branch that reaches
                the image surface.
            reference: The branch whose exit points are the common points;
                None uses the first of ``keys``.

        Returns:
            One (N, 3) array per input state.
        """
        selected = self._select(keys)
        if not selected:
            raise ValueError("No branch to add.")
        ref = self.branches[reference] if reference is not None else selected[0]
        x_ref = _positions(ref.rays)
        total: list[Array] | None = None
        for branch in selected:
            rays = branch.rays
            shift = _dot_real(be.real(rays.k), x_ref - _positions(rays))
            carry = be.exp(1j * _k0_per_length(rays.w) * shift)
            fields = [f * carry[:, None] for f in self.exit_fields(branch.key)]
            if total is None:
                total = fields
            else:
                total = [a + b for a, b in zip(total, fields, strict=True)]
        assert total is not None
        return total

    def coherent_intensity(
        self,
        keys: list[BranchKey] | None = None,
        reference: BranchKey | None = None,
        analyzer: Any = None,
    ) -> Array:
        """Return the power of the coherent sum of branches per ray, shape (N,).

        Args:
            keys: The branches to add (see :meth:`coherent_fields`).
            reference: The reference branch (see :meth:`coherent_fields`).
            analyzer: An optional unit transmission axis, shape (3,) or
                (N, 3): the power behind an ideal linear polarizer along it.
                None gives the total power.

        Returns:
            The power, averaged over the input states of an unpolarized launch.
        """
        fields = self.coherent_fields(keys, reference)
        total = None
        for f in fields:
            if analyzer is None:
                power = be.sum(be.abs(f) ** 2, axis=1)
            else:
                a = be.to_complex(be.array(analyzer)) + 0 * f
                projected: Array = be.sum(a * f, axis=1)
                power = be.abs(projected) ** 2
            total = power if total is None else total + power
        assert total is not None
        return be.where(be.isfinite(total), total, 0.0) / len(fields)


# ----------------------------------------------------------------- helpers


def _k0_per_length(wavelength: Array) -> Array:
    """Return k0 = 2π/λ per lens unit (mm), λ in µm."""
    return 2 * math.pi / (wavelength * 1e-3)


def _positions(rays: AnisotropicRays) -> Array:
    """Return the ray positions, shape (N, 3)."""
    return be.stack([rays.x, rays.y, rays.z], axis=1)


def _dot_real(a: Array, b: Array) -> Array:
    return a[:, 0] * b[:, 0] + a[:, 1] * b[:, 1] + a[:, 2] * b[:, 2]


def _to_float(x: Array) -> float:
    return float(be.to_numpy(x))


def ray_power(rays: PolarizedRays, state: PolarizationState | None) -> Array:
    """Return the power of each ray without changing the rays, shape (N,).

    The power that ``rays.update_intensity(state)`` would put into ``rays.i``:
    the running intensity × the flux factor × |P E_0|² (averaged over the x
    and the y input for an unpolarized state).

    Args:
        rays: The polarized rays.
        state: The launch polarization state (None: unpolarized).
    """
    fields = rays._compute_unscaled_exit_fields(state)
    total = None
    for f in fields:
        term = be.sum(be.abs(f) ** 2, axis=1)
        total = term if total is None else total + term
    return rays.i * rays.flux_factor * total / len(fields)


def _copy_rays(rays: AnisotropicRays) -> AnisotropicRays:
    """Return a copy of the rays with copies of their arrays."""
    new = copy.copy(rays)
    for name, value in vars(rays).items():
        if hasattr(value, "shape"):
            setattr(new, name, be.copy(value))
    return new


def _trace(view: Any, rays: AnisotropicRays) -> None:
    """Trace the rays through one surface view (in place)."""
    view.trace(rays)


def _alive(rays: AnisotropicRays, power: Array) -> Array:
    """Return True for rays with a finite direction and power, shape (N,)."""
    direction = rays.L + rays.M + rays.N
    return be.isfinite(direction) & be.isfinite(power)


# ----------------------------------------------------------------- the driver


@dataclass
class _Choice:
    """One child at a splitting step."""

    reflect: bool
    mode: str
    exit_anisotropic: bool
    followed: bool


@dataclass
class _Node:
    """A partial branch on the depth-first stack."""

    rays: AnisotropicRays
    power: Array  # per-ray power, 0 for dead rays
    alive: Array
    index: int  # the base surface index of the last step
    reverse: bool
    reflections: int
    steps: list[RawStep]
    views: list[SurfaceView]
    # (carried power, clipped power) of each child of a splitting step
    split: list[tuple[float, float]] = field(default_factory=list)


class _BranchSurfaceGroup(SequencedSurfaceGroup):
    """A sequenced surface group that traces ``AnisotropicRays``."""

    def trace(self, rays: Any, skip: int = 0) -> Any:
        if not isinstance(rays, AnisotropicRays):
            rays = AnisotropicRays.from_rays(rays)
        return super().trace(rays, skip=skip)


class BranchTracer:
    """Trace every branch of a sequential system with splitting surfaces.

    Args:
        optic: The optic. It needs a polarization state (the rays are
            ``AnisotropicRays``).
        ghosts: The surfaces whose reflected children are followed: an
            iterable of surface indices, or ``"all"`` (every surface between
            the object and the image). None or empty: transmitted children
            only.
        threshold: A branch whose power falls below ``threshold`` times the
            launch power is dropped (``pruned`` in the ledger). 0 keeps every
            branch with power.
        max_reflections: A branch is dropped before it takes reflection
            ``max_reflections + 1`` (2 is enough for the first-order ghosts
            that reach the image).

    Raises:
        ValueError: If the optic has no polarization state, or if a ghost
            surface is a mirror, carries a BSDF or a coating other than
            ``FresnelCoating``.
    """

    def __init__(
        self,
        optic: Optic,
        ghosts: Any = None,
        threshold: float = 1e-6,
        max_reflections: int = 2,
    ):
        if optic.polarization == "ignore":
            raise ValueError("Branch tracing needs a polarization state of the optic.")
        self.optic = optic
        self.threshold = threshold
        self.max_reflections = max_reflections
        surfaces = optic.surfaces.surfaces
        self._last = len(surfaces) - 1
        if ghosts is None:
            ghost_set: set[int] = set()
        elif isinstance(ghosts, str):
            if ghosts != "all":
                raise ValueError(f"ghosts must be indices or 'all', not {ghosts!r}.")
            ghost_set = set(range(1, self._last))
        else:
            ghost_set = {int(i) for i in ghosts}
        for index in ghost_set:
            if not 0 < index < self._last:
                raise ValueError(
                    f"Ghost surface {index} is not between the object and the "
                    "image surface."
                )
            model = surfaces[index].interaction_model
            if isinstance(model, AnisotropicInteractionModel):
                continue
            if getattr(model, "is_reflective", False):
                raise ValueError(f"Ghost surface {index} is a mirror.")
            if getattr(model, "bsdf", None) is not None:
                raise ValueError(f"Ghost surface {index} has a BSDF.")
            coating = getattr(model, "coating", None)
            if coating is not None and not isinstance(coating, FresnelCoating):
                raise ValueError(
                    f"Ghost surface {index} has a coating; only a bare "
                    "(Fresnel) interface can be split into R and T."
                )
        self.ghosts = ghost_set

    # -- structure -------------------------------------------------------------

    def _base(self, index: int) -> Any:
        return self.optic.surfaces.surfaces[index]

    def is_splitting(self, index: int) -> bool:
        """Return True if the surface adds entries to the branch key."""
        model = self._base(index).interaction_model
        return index in self.ghosts or isinstance(model, AnisotropicInteractionModel)

    def label(self, index: int) -> str:
        """Return the label of a surface in the branch key."""
        surface = self._base(index)
        model = surface.interaction_model
        label = getattr(model, "label", None)
        if label:
            return str(label)
        return surface.comment or f"s{index}"

    def _choices(self, index: int, reverse: bool) -> list[_Choice]:
        """Return the children of a splitting step, in branch order."""
        base = self._base(index)
        pre, post = resolve_view_materials(base, reverse, None)
        choices = []
        for reflect, medium in ((False, post), (True, pre)):
            followed = not reflect or index in self.ghosts
            if isinstance(medium, BaseTensorMaterial):
                modes = (
                    ("o", "e")
                    if isinstance(medium, UniaxialMaterial)
                    else ("slow", "fast")
                )
                choices += [_Choice(reflect, m, True, followed) for m in modes]
            else:
                mode = "R" if reflect else "T"
                choices.append(_Choice(reflect, mode, False, followed))
        return choices

    def _view(
        self, index: int, reverse: bool, choice: _Choice | None, previous: Any
    ) -> SurfaceView:
        """Return a view of a step with its model set for a choice."""
        base = self._base(index)
        override = "reflect" if choice is not None and choice.reflect else None
        view = SurfaceView(base, reverse, override, previous)
        if choice is not None:
            self._configure(view, index, choice)
        return view

    def _configure(self, view: SurfaceView, index: int, choice: _Choice) -> None:
        """Give a view of a splitting surface the model of a choice."""
        model: Any = view.interaction_model
        if not isinstance(model, AnisotropicInteractionModel):
            # A ghost surface between isotropic media: the bare interface.
            surface: Any = view  # a SurfaceView stands in for the Surface
            model = AnisotropicInteractionModel(surface)
            view.interaction_model = model
        model.is_reflective = choice.reflect
        model.mode = choice.mode
        model.label = self.label(index)

    # -- rays ------------------------------------------------------------------

    def generate_rays(
        self,
        Hx: Any = 0.0,
        Hy: Any = 0.0,
        wavelength: float | str = "primary",
        num_rays: int | None = 100,
        distribution: Any = "hexapolar",
    ) -> AnisotropicRays:
        """Return launch rays as ``Optic.trace`` makes them (``AnisotropicRays``)."""
        if wavelength == "primary":
            wavelength = self.optic.primary_wavelength
        tracer: Any = self.optic.ray_tracer
        tracer._validate_normalized_coordinates(Hx, Hy, "field")
        if isinstance(distribution, str):
            distribution = create_distribution(distribution)  # type: ignore[arg-type]
            distribution.generate_points(num_rays)
        px, py = distribution.x, distribution.y
        hx, hy = be.atleast_1d(Hx), be.atleast_1d(Hy)
        n_pupil = len(px)
        rays = tracer.ray_generator.generate_rays(
            be.repeat(hx, n_pupil),
            be.repeat(hy, n_pupil),
            be.tile(px, len(hx)),
            be.tile(py, len(hx)),
            wavelength,
        )
        if not isinstance(rays, AnisotropicRays):
            rays = AnisotropicRays.from_rays(rays)
        return rays

    # -- enumeration -------------------------------------------------------------

    def trace_all(
        self,
        Hx: Any = 0.0,
        Hy: Any = 0.0,
        wavelength: float | str = "primary",
        num_rays: int | None = 100,
        distribution: Any = "hexapolar",
        rays: AnisotropicRays | PolarizedRays | None = None,
    ) -> BranchResult:
        """Trace every branch of one launch.

        Args:
            Hx, Hy: The normalized field coordinates.
            wavelength: The wavelength in µm, or ``"primary"``.
            num_rays: The number of rays of the distribution.
            distribution: The pupil distribution (as ``Optic.trace``).
            rays: Launch rays to use instead (for example a hand-made fan);
                they are copied, then converted to ``AnisotropicRays``.

        Returns:
            BranchResult: ``{key: Branch}`` and the power ledger.
        """
        if rays is None:
            rays = self.generate_rays(Hx, Hy, wavelength, num_rays, distribution)
        elif not isinstance(rays, AnisotropicRays):
            rays = AnisotropicRays.from_rays(rays)
        else:
            rays = _copy_rays(rays)
        state = self.optic.polarization_state
        launch = ray_power(rays, state)
        alive = _alive(rays, launch)
        launch = be.where(alive, launch, 0.0)
        launch_power = _to_float(be.sum(launch))
        if launch_power <= 0:
            raise ValueError("The launch rays carry no power.")
        ledger = PowerLedger(launch=launch_power)
        result = BranchResult({}, ledger, {}, state)

        object_view = SurfaceView(self._base(0))
        _trace(object_view, rays)
        stack = [_Node(rays, launch, alive, 0, False, 0, [0], [object_view])]
        while stack:
            node = stack.pop()
            index = node.index - 1 if node.reverse else node.index + 1
            if node.reverse and index == 0:
                self._finish(result, node, "returned")
                continue
            if not self.is_splitting(index):
                view = self._view(index, node.reverse, None, node.views[-1])
                child = self._step(result, node, view, index)
                if index == self._last:
                    self._finish(result, child, "image")
                else:
                    stack.append(child)
                continue
            children: list[_Node] = []
            for choice in self._choices(index, node.reverse):
                view = self._view(index, node.reverse, choice, node.views[-1])
                child, power, lost = self._trace_child(node, view, index, choice)
                ledger.evanescent += lost
                if not choice.followed:
                    ledger.unfollowed += power
                    continue
                key = child.rays.branch_key
                if choice.reflect and node.reflections >= self.max_reflections:
                    self._prune(result, key, power, "reflections")
                elif power <= self.threshold * launch_power:
                    self._prune(result, key, power, "threshold")
                elif index == self._last:
                    self._finish(result, child, "image")
                else:
                    children.append(child)
            self._account(result, node)
            # Depth first, in the order of the choices.
            stack.extend(reversed(children))
        # Fractions of the launch power.
        for name in LEDGER_FIELDS:
            setattr(ledger, name, getattr(ledger, name) / launch_power)
        for branch in result.branches.values():
            branch.power = branch.power / launch_power
        return result

    def enumerate(self, *args: Any, **kwargs: Any) -> list[BranchKey]:
        """Return the keys of the kept branches (arguments of :meth:`trace_all`)."""
        return self.trace_all(*args, **kwargs).keys()

    # -- one step ------------------------------------------------------------------

    def _advance(
        self, node: _Node, view: SurfaceView, index: int, reverse: bool
    ) -> tuple[_Node, Array, Array, Array]:
        """Trace a copy of the node's rays through a view.

        Returns the child node, and per-ray: the power that stays alive, the
        power of rays that the step loses, the parent power of clipped rays.
        """
        rays = _copy_rays(node.rays)
        _trace(view, rays)
        power = ray_power(rays, self.optic.polarization_state)
        finite = _alive(rays, power)
        clipped = node.alive & (rays.i == 0)
        lost = node.alive & ~finite & ~clipped
        alive = node.alive & finite & ~clipped
        kept = be.where(alive, power, 0.0)
        lost_power = be.where(lost & be.isfinite(power), power, 0.0)
        clipped_power = be.where(clipped, node.power, 0.0)
        steps = [
            *node.steps,
            (index, "reflect") if view.interaction_override else index,
        ]
        child = _Node(
            rays,
            kept,
            alive,
            index,
            reverse,
            node.reflections,
            steps,
            [*node.views, view],
        )
        return child, kept, lost_power, clipped_power

    def _step(
        self, result: BranchResult, node: _Node, view: SurfaceView, index: int
    ) -> _Node:
        """Trace a step that does not split; book its losses."""
        child, kept, lost, clipped = self._advance(node, view, index, node.reverse)
        ledger = result.ledger
        lost_sum = _to_float(be.sum(lost))
        clipped_sum = _to_float(be.sum(clipped))
        ledger.evanescent += lost_sum
        ledger.clipped += clipped_sum
        ledger.absorbed += (
            _to_float(be.sum(node.power))
            - _to_float(be.sum(kept))
            - lost_sum
            - clipped_sum
        )
        return child

    def _trace_child(
        self, node: _Node, view: SurfaceView, index: int, choice: _Choice
    ) -> tuple[_Node, float, float]:
        """Trace one child of a splitting step; return it, its power, its loss."""
        reverse = node.reverse != choice.reflect
        child, kept, lost, clipped = self._advance(node, view, index, reverse)
        child.reflections = node.reflections + (1 if choice.reflect else 0)
        power = _to_float(be.sum(kept))
        lost_sum = _to_float(be.sum(lost))
        node.split.append((power + lost_sum, _to_float(be.sum(clipped))))
        return child, power, lost_sum

    def _account(self, result: BranchResult, node: _Node) -> None:
        """Book the clipped and the absorbed power of a splitting step."""
        entries = node.split
        if not entries:
            return
        clipped = entries[0][1]  # the aperture clips before the interaction
        carried = sum(e[0] for e in entries)
        result.ledger.clipped += clipped
        result.ledger.absorbed += _to_float(be.sum(node.power)) - carried - clipped

    @staticmethod
    def _prune(result: BranchResult, key: BranchKey, power: float, reason: str) -> None:
        result.ledger.pruned += power
        result.pruned[key] = (power / result.ledger.launch, reason)

    def _finish(self, result: BranchResult, node: _Node, terminal: str) -> None:
        """Close a branch: update the intensity and book its power."""
        rays = node.rays
        running = be.copy(rays.i)
        state: Any = self.optic.polarization_state  # None: unpolarized
        rays.update_intensity(state)
        power = _to_float(be.sum(node.power))
        if terminal == "image":
            result.ledger.kept += power
        else:
            result.ledger.returned += power
        key = rays.branch_key
        result.branches[key] = Branch(
            key, node.steps, rays, running, node.views, power, terminal
        )

    # -- per-branch sequences ------------------------------------------------------

    def sequence(self, key: BranchKey, name: str | None = None) -> SequencedOptic:
        """Return a ``SequencedOptic`` that traces one branch.

        Its views carry the choices of the key, and it traces
        ``AnisotropicRays``, so ``trace`` gives the rays of the branch and
        every analysis that takes a sequenced optic works on the branch.

        A mirror of the optic keeps the direction of the path, as in
        ``Optic``; a reflected child of a splitting surface reverses it.

        Args:
            key: The branch key.
            name: The name of the sequence (default: the key as text).
        """
        steps: list[SequenceRawStep] = [0]
        plan: list[tuple[int, _Choice | None]] = []
        index, reverse, entry = 0, False, 0
        while True:
            index = index - 1 if reverse else index + 1
            if reverse and index == 0:
                break
            if not self.is_splitting(index):
                steps.append(index)
                plan.append((index, None))
                if index == self._last:
                    break
                continue
            if entry >= len(key):
                raise ValueError(f"The key {key!r} ends before the path does.")
            choice = self._parse_entry(index, reverse, key[entry])
            entry += 1
            steps.append((index, "reflect") if choice.reflect else index)
            plan.append((index, choice))
            reverse = reverse != choice.reflect
        if entry != len(key):
            raise ValueError(f"The key {key!r} is longer than its path.")
        seq = SequencedOptic(self.optic, name or repr(key), steps)
        seq.surfaces = _BranchSurfaceGroup(self.optic.surfaces.surfaces, steps)
        for view, (step, step_choice) in zip(seq.surfaces[1:], plan, strict=True):
            if step_choice is not None:
                self._configure(view, step, step_choice)
        return seq

    def _parse_entry(
        self, index: int, reverse: bool, entry: tuple[str, ...]
    ) -> _Choice:
        """Return the choice of a key entry at a step, checking the label."""
        label = self.label(index)
        if entry[0] != label:
            raise ValueError(
                f"Key entry {entry!r} does not match surface {index} ({label!r})."
            )
        reflect = entry[1] == "R"
        mode = entry[2] if len(entry) > 2 else entry[1]
        for choice in self._choices(index, reverse):
            if choice.reflect == reflect and choice.mode == mode:
                return choice
        raise ValueError(f"Key entry {entry!r} is not a child of surface {index}.")
