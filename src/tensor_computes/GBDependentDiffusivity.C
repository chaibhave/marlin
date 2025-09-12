
/**********************************************************************/
/*                    DO NOT MODIFY THIS HEADER                       */
/*             Swift, a Fourier spectral solver for MOOSE             */
/*                                                                    */
/*            Copyright 2024 Battelle Energy Alliance, LLC            */
/*                        ALL RIGHTS RESERVED                         */
/**********************************************************************/

#include "GBDependentDiffusivity.h"
#include "SwiftUtils.h"
#include <torch/torch.h>

registerMooseObject("SwiftApp", GBDependentDiffusivity);

InputParameters
GBDependentDiffusivity::validParams()
{
  InputParameters params = TensorOperator<>::validParams();
  params.addClassDescription("Calculates the GB-dependent rank-2 tensor diffusivity.");
  params.addRequiredParam<TensorInputBufferName>("D_b", "Bulk diffusivity tensor name");
  params.addRequiredParam<TensorInputBufferName>("D_gb", "GB diffusivity tensor name");
  params.addRequiredParam<TensorInputBufferName>("GB", "GB tensor name");
  return params;
}

GBDependentDiffusivity::GBDependentDiffusivity(const InputParameters & parameters)
  : TensorOperator<>(parameters),
    _D_b(getInputBuffer("D_b")),
    _D_gb(getInputBuffer("D_gb")),
    _gb(getInputBuffer("GB")),
    _imag(torch::tensor(c10::complex<double>(0.0, 1.0), MooseTensor::complexFloatTensorOptions()))
{
}

void
GBDependentDiffusivity::computeBuffer()
{
  auto dim = _domain.getDim();
  if(dim==1)
    mooseError("Anisotropic diffusivity is not implemented for 1D systems!");
  auto grad_x_gb = _domain.ifft(_i * _domain.fft(_gb) * _imag);
  auto grad_y_gb = _domain.ifft(_j * _domain.fft(_gb) * _imag);
  auto grad_z_gb = _domain.ifft(_k * _domain.fft(_gb) * _imag);

  auto grad_gb = torch::stack({grad_x_gb, grad_y_gb, grad_z_gb}, -1);

  auto ein_operation = dim > 2 ? "ijkl,ijkm->ijklm" : "ijk,ijl->ijkl";
  auto outer_product = torch::einsum(ein_operation, {grad_gb, grad_gb});

  auto temp = torch::square(grad_gb);

  auto square_grad_gb = torch::sum(temp, /*dim=*/-1).unsqueeze(-1).unsqueeze(-1);

  auto normal_tensor = torch::where(_gb.unsqueeze(-1).unsqueeze(-1) > 0, outer_product / square_grad_gb, 0.0 );

  auto D = torch::zeros_like(normal_tensor);

  // Set the diagonal elements to D_b
  for (auto i = 0; i < dim; ++i)
  {
    D.index_put_({"...", i, i}, _D_b);
  }

  // Add the normal_tensor * D_gb contribution
  D += normal_tensor * (_gb * _D_gb).unsqueeze(-1).unsqueeze(-1);

  _u = D;
}
