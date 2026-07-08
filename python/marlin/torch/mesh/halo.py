"""Halo (ghost-cell) filling: Face labels, the HaloFiller strategy
protocol, and concrete physical BC implementations.

Deliberately independent of Mesh and Field:
* Face is a pure geometric label (which of the box's faces).
* HaloFiller.ghost_slab takes only what it needs (ghost width, cell
  spacing) rather than a whole Mesh, so fillers are trivially testable
  and don't create an import cycle with mesh.py.
* BCs are assigned per-Field (see field.py), not per-Mesh, since
  different variables on the same mesh can have different BCs.

Spatial axes convention (matches Field): a field tensor's first three
dims are always (x, y, z) -- i.e. axis 0/1/2 -- with any vector/tensor
component dims trailing. Axes beyond `dim` are size 1, un-ghosted.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Protocol

import torch


class Face(Enum):
    """The six faces of the box domain. axis 0/1/2 == x/y/z."""
    X_LO = ("x", 0, "lo")
    X_HI = ("x", 0, "hi")
    Y_LO = ("y", 1, "lo")
    Y_HI = ("y", 1, "hi")
    Z_LO = ("z", 2, "lo")
    Z_HI = ("z", 2, "hi")

    def __init__(self, axis_name: str, axis: int, side: str):
        self.axis_name = axis_name
        self.axis = axis
        self.side = side


class HaloFiller(Protocol):
    """Produces the ghost slab for one face, functionally.

    Given the interior tensor `u` (nx, ny, nz, ...), return a tensor of
    ghost values with the same shape except size `ghost` along `face.axis`.
    Must not mutate `u`.
    """

    def ghost_slab(
        self, u: torch.Tensor, face: Face, ghost: int, spacing: torch.Tensor
    ) -> torch.Tensor: ...


def _edge_slice(u: torch.Tensor, axis: int, side: str, width: int) -> torch.Tensor:
    """The `width` interior layers adjacent to a face."""
    if side == "lo":
        return u.narrow(axis, 0, width)
    return u.narrow(axis, u.shape[axis] - width, width)


def _flip(u: torch.Tensor, axis: int) -> torch.Tensor:
    return torch.flip(u, dims=(axis,))


@dataclass(frozen=True)
class Dirichlet:
    """u = value on the face (cell-centered linear extrapolation):
    ghost_k = 2*value - mirror_k.
    """
    value: float

    def ghost_slab(
        self, u: torch.Tensor, face: Face, ghost: int, spacing: torch.Tensor
    ) -> torch.Tensor:
        mirror = _flip(_edge_slice(u, face.axis, face.side, ghost), face.axis)
        return 2.0 * self.value - mirror


@dataclass(frozen=True)
class Neumann:
    """du/dn = flux on the face (outward normal). flux == 0.0 is the
    zero-flux mirror condition.
    """
    flux: float = 0.0

    def ghost_slab(
        self, u: torch.Tensor, face: Face, ghost: int, spacing: torch.Tensor
    ) -> torch.Tensor:
        mirror = _flip(_edge_slice(u, face.axis, face.side, ghost), face.axis)
        if self.flux == 0.0:
            return mirror
        k = torch.arange(ghost, device=u.device, dtype=u.dtype)
        dist = (2.0 * k + 1.0) * spacing
        if face.side == "lo":
            dist = torch.flip(dist, dims=(0,))
        shape = [1] * u.ndim
        shape[face.axis] = ghost
        dist = dist.view(shape)
        sign = 1.0 if face.side == "hi" else -1.0
        return mirror + sign * self.flux * dist


@dataclass(frozen=True)
class Periodic:
    """Ghosts copy the `ghost` interior layers from the opposite side."""

    def ghost_slab(
        self, u: torch.Tensor, face: Face, ghost: int, spacing: torch.Tensor
    ) -> torch.Tensor:
        opposite = "hi" if face.side == "lo" else "lo"
        return _edge_slice(u, face.axis, opposite, ghost)