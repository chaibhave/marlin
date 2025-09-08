
/**********************************************************************/
/*                    DO NOT MODIFY THIS HEADER                       */
/*             Swift, a Fourier spectral solver for MOOSE             */
/*                                                                    */
/*            Copyright 2024 Battelle Energy Alliance, LLC            */
/*                        ALL RIGHTS RESERVED                         */
/**********************************************************************/

#include "ReciprocalRobinBC.h"
#include "SwiftUtils.h"
#include <torch/torch.h>

registerMooseObject("SwiftApp", ReciprocalRobinBC);

InputParameters
ReciprocalRobinBC::validParams()
{
  InputParameters params = TensorOperator::validParams();
  params.addClassDescription("Calculates a RobinBC term for the Cahn-Hilliard diffusion equation.");
  params.addRequiredParam<TensorInputBufferName>("chemical_potential",
                                                 "Chemical potential buffer name");
  params.addRequiredParam<TensorInputBufferName>("mobility", "Mobility buffer name");
  params.addParam<TensorInputBufferName>(
      "W_N", "Tensor that holds weights for Neumann-Dirichlet split (0 <= W_N <= 1)");
  params.addParam<TensorInputBufferName>("B_N", "NeumannBC value");
  params.addParam<TensorInputBufferName>("B_D", "Dirichlet BC value");
  params.addParam<TensorInputBufferName>("psi", "Variable to impose Neuamnn BC.");
  params.addParam<bool>("always_update_psi", false, "Set to true if the BC changes .");
  return params;
}

ReciprocalRobinBC::ReciprocalRobinBC(const InputParameters & parameters)
  : TensorOperator(parameters),
    _chem_pot(getInputBuffer("chemical_potential")),
    _M(getInputBuffer("mobility")),
    _W_N(getInputBuffer("W_N")),
    _B_N(getInputBuffer("B_N")),
    _B_D(getInputBuffer("B_D")),
    _imag(torch::tensor(c10::complex<double>(0.0, 1.0), MooseTensor::complexFloatTensorOptions())),
    _psi(getInputBuffer("psi")),
    _update_psi(true),
    _always_update_psi(getParam<bool>("always_update_psi"))
{
}

void
ReciprocalRobinBC::computeBuffer()
{
  if (_update_psi || _always_update_psi)
  {
    _psi_thresh = _psi > 0.0;

    auto grad_psi_x = _domain.ifft(_i * _domain.fft(_psi) * _imag);
    auto grad_psi_y = _domain.ifft(_j * _domain.fft(_psi) * _imag);
    auto grad_psi_z = _domain.ifft(_k * _domain.fft(_psi) * _imag);
    auto psi_sq = torch::square(_psi);

    _grad_psi_by_psi =
        torch::where(_psi_thresh,
                     torch::sqrt(torch::square(grad_psi_x) + torch::square(grad_psi_y) +
                                 torch::square(grad_psi_z)) /
                         _psi,
                     0);
    _grad_psi_x_by_psi_sq = torch::where(_psi_thresh, grad_psi_x / psi_sq, 0.0);
    _grad_psi_y_by_psi_sq = torch::where(_psi_thresh, grad_psi_y / psi_sq, 0.0);
    _grad_psi_z_by_psi_sq = torch::where(_psi_thresh, grad_psi_z / psi_sq, 0.0);

    _update_psi = false;
  }

  // Neumann terms
  auto robin_bc = _W_N * _M * _B_N * _grad_psi_by_psi;

  // Dirichlet terms
  {

    auto J_x = _M * _domain.ifft(_i * _domain.fft(_psi * _chem_pot) * _imag);
    auto J_y = _M * _domain.ifft(_j * _domain.fft(_psi * _chem_pot) * _imag);
    auto J_z = _M * _domain.ifft(_z * _domain.fft(_psi * _chem_pot) * _imag);
    robin_bc -= (1 - _W_N)*( _grad_psi_x_by_psi_sq * J_x + _grad_psi_y_by_psi_sq * J_y + _grad_psi_z_by_psi_sq * J_z
        +  _B_D*torch::square(_grad_psi_by_psi) );
  }

  _u = robin_bc;
}
