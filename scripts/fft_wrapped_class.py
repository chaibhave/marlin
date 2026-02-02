import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List

# class KKS(nn.Module):
#     def __init__(self):
#         super(KKS, self).__init__()

#         # Model constants

#         # Thermodynamics constants
#         self.E1 = nn.Parameter(torch.tensor(4.0))
#         self.kappa = nn.Parameter(torch.tensor(5.0))
#         self.w = nn.Parameter(torch.tensor(1.0))
#         self.c0_a = nn.Parameter(torch.tensor(0.3))
#         self.c0_b = nn.Parameter(torch.tensor(0.7))

#         # Kinetics constants
#         self.L = nn.Parameter(torch.tensor(5.0))
#         self.D1 = nn.Parameter(torch.tensor(1.0))
#         self.D2 = nn.Parameter(torch.tensor(5.0))

#     def CH(self, c_hat, eta_hat, k):
#         return torch.square(k).squeeze() * self.D1 * c_hat

#     def forward(self, x):
#         c_hat = x[..., 0]
#         eta_hat = x[..., 1]
#         k = x[..., 2:]
#         if not k.shape[-1] == 1:
#             raise ValueError("2D and 3D not yet implemented!")
#         dc_hat_dt = self.CH(c_hat, eta_hat, k)
#         return dc_hat_dt


class CH(nn.Module):
    def __init__(self):
        super(CH, self).__init__()

        # Model constants

        # Thermodynamics constants
        # self.E1 = nn.Parameter(torch.tensor(4.0))
        # self.kappa = nn.Parameter(torch.tensor(5.0))
        # self.w = nn.Parameter(torch.tensor(1.0))
        # self.c0_a = nn.Parameter(torch.tensor(0.3))
        # self.c0_b = nn.Parameter(torch.tensor(0.7))

        # Kinetics constants
        # self.L = nn.Parameter(torch.tensor(5.0))
        self.D1 = 1.0  # nn.Parameter(torch.tensor(1.0))
        self.D2 = 5.0  # nn.Parameter(torch.tensor(5.0))

    def forward(self,
                c_hat: torch.Tensor,
                local_reciprocal_axis: torch.Tensor,
                real_domain_shape: List[int]) -> torch.Tensor:
        """
        c_hat: complex, shape (..., *kshape) where last spatial axis is rfft-reduced
            e.g. 3D: (..., Nx, Ny, Nz//2+1)
        local_reciprocal_axis: real, shape (..., *kshape, ndim) with components (kx,ky,kz)
        real_domain_shape: List[int] real-space spatial sizes [Nx,Ny,Nz] (or [Nx] / [Nx,Ny])
        Returns: dc_hat_dt, complex, same shape as c_hat
        """

        # Spatial dims are the last ndim axes of c_hat
        ndim = local_reciprocal_axis.shape[-1]

        # spatial_dims = tuple([i for i in range(-ndim, 0)])
        # grad_spatial_dims = tuple([i for i in range(-ndim-1, -1)])

        c = torch.fft.irfftn(c_hat)        # (..., *realshape)
        # (..., *realshape)
        D = self.D1 + c * (self.D2 - self.D1)

        # Build (i*k) and grad in spectral space, vectorized over components
        # local_reciprocal_axis shape: (..., *kshape, ndim)
        # complex, same shape
        ik = 1j * local_reciprocal_axis


        return c_hat #dc_hat_dt

if __name__ == '__main__':
    ch_runner = torch.jit.script(CH())
    torch.jit.save(
        ch_runner, '/Users/bhavcv/projects/marlin/data/torch_script_pfm/ch_runner.pt')
