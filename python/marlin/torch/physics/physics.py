"""Physics: assembles F(u) = sum of kernel residual contributions for one
named variable.

This is the single point where:
* `with_halos()` is called on the target variable -- exactly once per
  residual evaluation, regardless of how many kernels touch it. Kernels
  never pad themselves (see Kernel's docstring).
* Kernel output shapes get checked against the variable's interior shape.
  Kernels (e.g. Source, which may return a bare 0-d tensor) are allowed
  to return anything broadcastable; a genuinely incompatible shape is a
  configuration mistake and should fail here, with a message naming the
  offending kernel, rather than as an opaque broadcast error wherever the
  result is later consumed.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from ..mesh import Mesh, State
from ..kernels import Kernel


@dataclass
class Physics:
    """F(u) = 0 for one named variable, as the sum of its kernels'
    residual contributions.

    Matches the MOOSE mental model: a variable's residual is the sum of
    the Kernel objects registered against it, evaluated on the current
    iterate.
    """

    variable: str
    kernels: list[Kernel]

    def residual(self, state: State, mesh: Mesh) -> torch.Tensor:
        if not self.kernels:
            raise ValueError(f"Physics for '{self.variable}' has no kernels registered")

        target = state[self.variable]
        u_padded = target.with_halos()
        expected_shape = target.interior.shape

        total = None
        for kernel in self.kernels:
            contribution = kernel.residual(u_padded, mesh)
            self._check_broadcastable(contribution, expected_shape, kernel)
            total = contribution if total is None else total + contribution

        # Guards the edge case where every kernel returned something
        # smaller than the full interior shape (e.g. a lone 0-d Source
        # term) -- the assembled residual should always match the
        # variable's actual interior shape for downstream consumers
        # (norms, Newton updates, etc). This is a broadcast VIEW, not a
        # copy: fine for reads, not for in-place writes into it.
        return total.broadcast_to(expected_shape)

    @staticmethod
    def _check_broadcastable(
        contribution: torch.Tensor, expected_shape: torch.Size, kernel: Kernel
    ) -> None:
        try:
            torch.broadcast_shapes(contribution.shape, expected_shape)
        except RuntimeError as e:
            raise ValueError(
                f"kernel {type(kernel).__name__} produced a residual contribution "
                f"of shape {tuple(contribution.shape)}, which is not broadcastable "
                f"against the target variable's interior shape {tuple(expected_shape)}"
            ) from e