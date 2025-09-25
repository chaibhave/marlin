
/**********************************************************************/
/*                    DO NOT MODIFY THIS HEADER                       */
/*             Swift, a Fourier spectral solver for MOOSE             */
/*                                                                    */
/*            Copyright 2024 Battelle Energy Alliance, LLC            */
/*                        ALL RIGHTS RESERVED                         */
/**********************************************************************/

#include "ReciprocalMatDiffusion.h"
#include "SwiftUtils.h"
#include <torch/torch.h>

registerMooseObject("SwiftApp", ReciprocalMatDiffusion);

InputParameters
ReciprocalMatDiffusion::validParams()
{
  InputParameters params = TensorOperator::validParams();
  params.addClassDescription(
      "Calculates the divergence of flux for a variable mobility in reciprocal space.");
  params.addRequiredParam<TensorInputBufferName>("chemical_potential",
                                                 "Chemical potential buffer name");
  params.addRequiredParam<TensorInputBufferName>("mobility", "Mobility buffer name");
  params.addParam<TensorInputBufferName>("psi", "Variable to impose Neuamnn BC.");
  params.addParam<bool>("always_update_psi", false, "Set to true if the BC changes .");
  return params;
}

ReciprocalMatDiffusion::ReciprocalMatDiffusion(const InputParameters & parameters)
  : TensorOperator(parameters),
    _chem_pot(getInputBuffer("chemical_potential")),
    _M(getInputBuffer("mobility")),
    _imag(torch::tensor(c10::complex<double>(0.0, 1.0), MooseTensor::complexFloatTensorOptions())),
    _psi(getInputBuffer("psi")),
    _update_psi(true),
    _always_update_psi(getParam<bool>("always_update_psi"))
{
}

void
ReciprocalMatDiffusion::computeBuffer()
{
  using namespace torch::indexing;

  if (_update_psi || _always_update_psi)
  {
    _psi_thresh = _psi > 0.0;
    _grad_psi_x_by_psi =
        torch::where(_psi_thresh, _domain.ifft(_i * _domain.fft(_psi) * _imag) / _psi, 0.0);
    _grad_psi_y_by_psi =
        torch::where(_psi_thresh, _domain.ifft(_j * _domain.fft(_psi) * _imag) / _psi, 0.0);
    _grad_psi_z_by_psi =
        torch::where(_psi_thresh, _domain.ifft(_k * _domain.fft(_psi) * _imag) / _psi, 0.0);
    _update_psi = false;
  }

  auto psi_M = _M.dim() > _psi_thresh.dim() ? _M * _psi_thresh.unsqueeze(-1).unsqueeze(-1)
                                            : _M * _psi_thresh;

  // Handle both scalar and tensor diffusivity
  auto grad_chem_pot = torch::stack({_domain.ifft(_i * _domain.fft(_chem_pot) * _imag),
                                     _domain.ifft(_j * _domain.fft(_chem_pot) * _imag),
                                     _domain.ifft(_k * _domain.fft(_chem_pot) * _imag)},
                                    -1);

  // Use einsum to perform scalar or tensor diffusivity based flux calculation
  auto J = psi_M.dim() > grad_chem_pot.dim()
               ? torch::einsum("...ij,...j->...i", {psi_M, grad_chem_pot})
               : torch::einsum("ij...,j...->i...", {psi_M, grad_chem_pot});

  auto div_J_hat =
      _imag * (_i * _domain.fft(J.index({Ellipsis, 0})) + _j * _domain.fft(J.index({Ellipsis, 1})) +
               _k * _domain.fft(J.index({Ellipsis, 2})));

  auto no_flux_hat = _domain.fft(_grad_psi_x_by_psi * J.index({Ellipsis, 0}) +
                                 _grad_psi_y_by_psi * J.index({Ellipsis, 1}) +
                                 _grad_psi_z_by_psi * J.index({Ellipsis, 2}));
  _u = div_J_hat + no_flux_hat;
}
