from __future__ import annotations

from dataclasses import dataclass, field as dataclass_field
from typing import Optional

import torch

from .mesh import Mesh
from .field import Field
from .halo import Face, HaloFiller


@dataclass
class State:
    """The current set of named variables on a `Mesh`.

    Deliberately a dict of `Field`s, not one stacked tensor: variables can
    have arbitrary, independent shapes (a scalar temperature field next to
    a 3x3 stress field), and the variable set can grow during problem
    setup (`add()` as physics/kernels are composed) rather than being
    fixed up front. Each variable keeps its own BCs, since different
    variables on the same mesh often need different ones.

    The "stacked tensor" convenience from the doc's original sketch is
    still available (`stacked()`) for the common case where a subset of
    variables share a shape and you want the leading-axis block-op
    behavior -- but it's opt-in, not the general representation.
    """

    mesh: Mesh
    fields: dict[str, Field] = dataclass_field(default_factory=dict)

    def add(
        self,
        name: str,
        *extra_shape: int,
        fillers: Optional[dict[Face, HaloFiller]] = None,
    ) -> Field:
        """Register a new zero-initialized variable. Raises on duplicate
        names rather than silently overwriting."""
        if name in self.fields:
            raise ValueError(f"variable '{name}' already registered in this State")
        f = Field.zeros(self.mesh, *extra_shape, fillers=fillers)
        self.fields[name] = f
        return f

    def __getitem__(self, name: str) -> Field:
        return self.fields[name]

    def __contains__(self, name: str) -> bool:
        return name in self.fields

    def __iter__(self):
        return iter(self.fields)

    def names(self) -> tuple[str, ...]:
        return tuple(self.fields.keys())

    # ------------------------------------------------------------- views

    def interior(self) -> dict[str, torch.Tensor]:
        """Interior view of every variable, by name."""
        return {name: f.interior for name, f in self.fields.items()}

    def with_halos(self) -> dict[str, torch.Tensor]:
        """Ghost-filled tensor for every variable that has fillers
        assigned. Variables without fillers (e.g. postprocessing-only)
        are simply omitted rather than raising, since asking for "all
        halos" is a bulk convenience, not a per-field requirement."""
        return {
            name: f.with_halos()
            for name, f in self.fields.items()
            if f.fillers is not None
        }

    # ------------------------------------------------- export boundary

    def to_positional(self, names: tuple[str, ...]) -> tuple[torch.Tensor, ...]:
        """Fixed-order tuple of this State's raw (ghosted) tensors.

        This -- not a single stacked tensor -- is what should cross the
        torch.export boundary: positional tensors only, in a documented,
        frozen order, no dict/dataclass/State object itself crossing the
        boundary. Heterogeneous shapes are fine since each variable is
        its own positional argument rather than a slice of one stack.
        """
        return tuple(self.fields[n].data for n in names)

    def update_from_positional(self, names: tuple[str, ...], tensors: tuple[torch.Tensor, ...]) -> None:
        """Write exported-graph outputs back into this State's storage.
        Exported graphs are stateless (return new tensors); this is the
        Python-side bookkeeping step that makes the update visible via
        `state[name]` afterward.
        """
        if len(names) != len(tensors):
            raise ValueError("names and tensors must have the same length")
        for name, t in zip(names, tensors):
            self.fields[name].data = t

    # ------------------------------------------------ optional stacking

    def stacked(self, names: tuple[str, ...]) -> torch.Tensor:
        """Stack same-shaped variables on a new leading axis.

        Opt-in convenience for the homogeneous case (e.g. several scalar
        fields you want a block-Jacobi smoother to sweep over at once).
        Raises if the requested variables don't share a shape -- this is
        not the general representation, so a mismatch means the caller
        picked the wrong tool, not that stacking should silently pad/fail
        differently.
        """
        tensors = [self.fields[n].data for n in names]
        shapes = {t.shape for t in tensors}
        if len(shapes) > 1:
            raise ValueError(
                f"stacked() requires identical shapes across {names}, got {shapes}"
            )
        return torch.stack(tensors, dim=0)