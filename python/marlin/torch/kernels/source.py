"""Source/forcing term kernel: contributes -f to F(u) = 0, so that
F(u) = 0 (e.g. Laplacian + Source) encodes -div(k grad u) - f = 0,
i.e. the PDE -div(k grad u) = f.
"""

from __future__ import annotations

from dataclasses import dataclass, field as dataclass_field
from typing import Optional, Union

import torch

from ..mesh import Mesh


@dataclass
class Source:
    """A source/forcing term. `value` may be:

    * a torch.Tensor -- stored by REFERENCE, never cloned. If the caller
      later mutates it in place (e.g. a time-varying forcing term updated
      each step), this kernel picks up the change automatically with no
      need to rebuild it. Must broadcast against the residual's interior
      shape.
    * a float or int -- resolved once, lazily (on first `residual()`
      call, when `mesh` is available), into a 0-d tensor at the mesh's
      dtype/device. This matches the doc's "scalars entering the math are
      0-d tensors" contract, and is cached so repeated residual
      evaluations don't reallocate it.
    """

    value: Union[torch.Tensor, float, int]
    _resolved: Optional[torch.Tensor] = dataclass_field(
        default=None, init=False, repr=False, compare=False
    )

    def residual(self, u_padded: torch.Tensor, mesh: Mesh) -> torch.Tensor:
        return -self._as_tensor(mesh)

    def diagonal(self, mesh: Mesh) -> torch.Tensor:
        # A source term doesn't depend on u, so its Jacobian contribution
        # is exactly zero -- unlike Laplacian.diagonal() (still `None`,
        # genuinely deferred), this one is fully known and cheap, so
        # return the real answer rather than punting.
        return torch.zeros((), dtype=mesh.config.dtype, device=mesh.config.device)

    def _as_tensor(self, mesh: Mesh) -> torch.Tensor:
        if isinstance(self.value, torch.Tensor):
            return self.value  # reference, not a copy
        if self._resolved is None:
            self._resolved = torch.tensor(
                float(self.value), dtype=mesh.config.dtype, device=mesh.config.device
            )
        return self._resolved