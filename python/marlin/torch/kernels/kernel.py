"""The Kernel interface: one physical term's contribution to F(u) = 0.

Kernels are pure and operate on already-padded (ghost-filled) tensors --
they never call `with_halos()` themselves. That's the assembler's job
(Physics, not yet built), so that when several kernels share a variable
(e.g. a diffusion term and a reaction term both touching `u`), the
ghost-fill happens exactly once per residual evaluation, not once per
kernel.
"""

from __future__ import annotations

from typing import Optional, Protocol

import torch

from ..mesh import Mesh


class Kernel(Protocol):
    """Residual contribution of one physical term.

    residual(): given the padded tensor(s) this kernel needs, return this
    term's contribution to F(u) on the INTERIOR cells only -- shape
    matches `mesh.interior(...)`, not the padded shape.

    diagonal(): optional contribution to the residual's Jacobian diagonal,
    for Jacobi/Chebyshev smoothers (Stage 2). Returning None means "not
    implemented for this kernel yet" -- callers must handle that, not
    assume every kernel provides one.
    """

    def residual(self, u_padded: torch.Tensor, mesh: Mesh) -> torch.Tensor: ...

    def diagonal(self, mesh: Mesh) -> Optional[torch.Tensor]:
        return None