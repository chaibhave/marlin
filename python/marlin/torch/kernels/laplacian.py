"""-div(k grad u): variable-coefficient Laplacian, finite-volume
discretization on the cell-centered structured mesh.

Face conductivities use the harmonic mean of the two adjacent cell values
-- the flux-consistent choice for (possibly discontinuous) variable
coefficients: it reproduces the correct effective resistance of two
conductors in series, unlike an arithmetic mean, which over-estimates
flux across a low-conductivity cell next to a high-conductivity one.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from ..mesh import Mesh, Field


@dataclass
class VariableCoefficientLaplacian:
    """Kernel for -div(k grad u).

    `conductivity` is its own Field, ghost-padded independently of `u`'s
    physical BCs -- k is a material property, not a solved variable, so
    its ghost values should just be a zero-flux (Neumann(0.0)) extension
    on every active face, regardless of what BC u actually has there.
    Construct it with exactly that filler set; this kernel doesn't
    validate that choice for you.
    """

    conductivity: Field

    def residual(self, u_padded: torch.Tensor, mesh: Mesh) -> torch.Tensor:
        k_padded = self.conductivity.with_halos()
        cfg = mesh.config
        g = cfg.ghost
        shape = cfg.shape  # (nx, ny, nz), interior sizes

        total = None
        for axis in range(cfg.dim):
            n = shape[axis]
            h = mesh.spacing(axis)

            u_lo = u_padded.narrow(axis, g - 1, n)
            u_mid = u_padded.narrow(axis, g, n)
            u_hi = u_padded.narrow(axis, g + 1, n)

            k_lo = k_padded.narrow(axis, g - 1, n)
            k_mid = k_padded.narrow(axis, g, n)
            k_hi = k_padded.narrow(axis, g + 1, n)

            k_face_hi = 2.0 * k_mid * k_hi / (k_mid + k_hi)
            k_face_lo = 2.0 * k_lo * k_mid / (k_lo + k_mid)

            flux_div = (k_face_hi * (u_hi - u_mid) - k_face_lo * (u_mid - u_lo)) / (h * h)
            term = -flux_div  # -div(k grad u)

            total = term if total is None else total + term

        return total

    def diagonal(self, mesh: Mesh):
        # Jacobian diagonal for smoothers -- deferred to Stage 2, where
        # weighted-Jacobi/Chebyshev smoothers actually need it.
        return None