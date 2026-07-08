from __future__ import annotations

from typing import Optional
from dataclasses import dataclass

import torch

from .halo import Face


@dataclass(frozen=True)
class GridConfig:
    """Static, structural description of the (local) domain.

    Everything here is hashable plain Python. torch.compile / torch.export
    specialize on these values; changing any of them means a new compiled
    artifact (by design -- one artifact per mesh level / resolution).

    nx, ny, nz are INTERIOR cell counts (ghost cells are not included).
    Coordinates are CELL CENTERS: dx = (xmax - xmin) / nx, and the first
    interior cell center sits at xmin + dx/2.
    """
    dim: int
    nx: int
    ny: int
    nz: int
    xmin: float = 0.0
    xmax: float = 1.0
    ymin: float = 0.0
    ymax: float = 1.0
    zmin: float = 0.0
    zmax: float = 1.0
    ghost: int = 1                      # max stencil radius over all kernels
    dtype: torch.dtype = torch.float64  # PETSc/MOOSE default; align policy
    device: str = "cpu"

    # --- domain-decomposition metadata (single-rank defaults) -------------
    # Global cell counts of the full problem; equal to local counts when
    # running single-rank.
    global_nx: Optional[int] = None
    global_ny: Optional[int] = None
    global_nz: Optional[int] = None
    # This subdomain's cell offset within the global index space.
    offset: tuple[int, int, int] = (0, 0, 0)

    def __post_init__(self):
        if self.dim not in (1, 2, 3):
            raise ValueError(f"dim must be 1, 2, or 3, got {self.dim}")
        if self.ghost < 1:
            raise ValueError(f"ghost must be >= 1, got {self.ghost}")
        dims_to_check = (self.nx, self.ny, self.nz)[: self.dim]
        for n in dims_to_check:
            if n < 2 * self.ghost:
                raise ValueError("cell count must be >= 2 * ghost width")

    @property
    def shape(self) -> tuple[int, int, int]:
        return (self.nx, self.ny, self.nz)

    @property
    def global_shape(self) -> tuple[int, int, int]:
        return (
            self.global_nx if self.global_nx is not None else self.nx,
            self.global_ny if self.global_ny is not None else self.ny,
            self.global_nz if self.global_nz is not None else self.nz,
        )

    @property
    def active_faces(self) -> tuple[Face, ...]:
        """Faces that geometrically exist given `dim`. Purely structural --
        this is what lets Field validate a filler dict against the mesh's
        dimensionality without Mesh knowing anything about BCs.
        """
        return tuple(f for f in Face if f.axis < self.dim)


class Mesh:
    """Structured, cell-centered box mesh with ghost machinery and a
    self-owned coarsening hierarchy.
    """

    def __init__(self, config: GridConfig):
        self.config = config

        # Numeric constants entering the math: 0-d device tensors.
        self.dx = torch.tensor(
            (config.xmax - config.xmin) / config.nx,
            device=config.device, dtype=config.dtype,
        )
        self.dy = torch.tensor(
            (config.ymax - config.ymin) / config.ny,
            device=config.device, dtype=config.dtype,
        )
        self.dz = torch.tensor(
            (config.zmax - config.zmin) / config.nz,
            device=config.device, dtype=config.dtype,
        )

        self._make_coords()

    def spacing(self, axis: int) -> torch.Tensor:
        """dx/dy/dz by axis index (0/1/2), for code that's axis-generic
        (e.g. HaloFiller implementations)."""
        return (self.dx, self.dy, self.dz)[axis]

    def _make_coords(self):
        """Interior cell-center coordinates only -- no ghost cells.

        Ghost cells are filled via stencil ops on interior data (padding,
        reflection, one-sided BC formulas), not by evaluating a function at
        a ghost cell's physical location, so there is deliberately no
        "ghosted coordinate" helper here. Field variables are stored
        ghosted; slice to the interior before combining with X/Y/Z:
            u[g:-g, g:-g, g:-g] = f(X, Y, Z)

        X/Y/Z are always created as (nx,1,1), (1,ny,1), (1,1,nz) shaped
        tensors, with size-1 placeholders for dims above `config.dim`, so
        downstream broadcasting code (X + Y + Z, etc.) doesn't need to
        branch on dim.
        """
        cfg = self.config
        kw = dict(device=cfg.device, dtype=cfg.dtype)

        def centers(n, xmin, dx):
            return xmin + (torch.arange(n, **kw) + 0.5) * dx

        x = centers(cfg.nx, cfg.xmin, self.dx)
        self.X = x.view(-1, 1, 1)

        if cfg.dim >= 2:
            y = centers(cfg.ny, cfg.ymin, self.dy)
            self.Y = y.view(1, -1, 1)
        else:
            self.Y = torch.zeros(1, 1, 1, **kw)

        if cfg.dim == 3:
            z = centers(cfg.nz, cfg.zmin, self.dz)
            self.Z = z.view(1, 1, -1)
        else:
            self.Z = torch.zeros(1, 1, 1, **kw)

    def zeros_field(self, *extra_shape: int) -> torch.Tensor:
        """Allocate a field variable at ghosted shape, zero-initialized.

        Axes beyond `config.dim` get size 1 with NO ghost padding, matching
        the convention used for X/Y/Z in `_make_coords` (a 2D mesh has no
        z-direction stencil, so there's nothing for z-ghosts to do).

        extra_shape: optional trailing dims for vector/tensor-valued fields
        living on each grid point, e.g.
            mesh.zeros_field()      -> (..., ...) scalar
            mesh.zeros_field(3)     -> (..., ..., 3) vector
            mesh.zeros_field(3, 3)  -> (..., ..., 3, 3) tensor
        """
        cfg = self.config
        g = cfg.ghost
        nx_shape = cfg.nx + 2*g
        ny_shape = cfg.ny + 2*g if cfg.dim >= 2 else 1
        nz_shape = cfg.nz + 2*g if cfg.dim == 3 else 1
        shape = (nx_shape, ny_shape, nz_shape, *extra_shape)
        return torch.zeros(shape, device=cfg.device, dtype=cfg.dtype)

    def interior(self, u: torch.Tensor) -> torch.Tensor:
        """View into the interior (non-ghost) region of a ghosted field.

        Works unchanged for vector/tensor fields -- slicing only the first
        three (spatial) dims leaves any trailing component dims untouched.
        Axes beyond `config.dim` are size 1 with no ghost padding, so the
        slice on those axes is a no-op ([0:1] via [:] below).
        """
        cfg = self.config
        g = cfg.ghost
        x_slice = slice(g, -g)
        y_slice = slice(g, -g) if cfg.dim >= 2 else slice(None)
        z_slice = slice(g, -g) if cfg.dim == 3 else slice(None)
        return u[x_slice, y_slice, z_slice]


if __name__ == "__main__":
    cfg = GridConfig(dim=3, nx=8, ny=8, nz=8)
    mesh = Mesh(config=cfg)
    print("dx:", mesh.dx.item())
    print("X shape:", mesh.X.shape, "first/last:", mesh.X.flatten()[0].item(), mesh.X.flatten()[-1].item())

    scalar = mesh.zeros_field()
    print("scalar field shape (ghosted):", scalar.shape)
    print("scalar field interior shape:", mesh.interior(scalar).shape)

    velocity = mesh.zeros_field(3)
    print("vector field shape (ghosted):", velocity.shape)
    print("vector field interior shape:", mesh.interior(velocity).shape)

    stress = mesh.zeros_field(3, 3)
    print("tensor field shape (ghosted):", stress.shape)
    print("tensor field interior shape:", mesh.interior(stress).shape)

    # interior assignment from coordinates, unaffected by trailing dims
    mesh.interior(scalar)[...] = mesh.X * mesh.Y * mesh.Z

    cfg1d = GridConfig(dim=1, nx=8, ny=1, nz=1, ghost=1)
    mesh1d = Mesh(config=cfg1d)
    print("1D X shape:", mesh1d.X.shape, "Y shape:", mesh1d.Y.shape, "Z shape:", mesh1d.Z.shape)