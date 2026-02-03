/**********************************************************************/
/*                     DO NOT MODIFY THIS HEADER                      */
/*            Marlin, a Fourier spectral solver for MOOSE             */
/*                                                                    */
/*            Copyright 2024 Battelle Energy Alliance, LLC            */
/*                        ALL RIGHTS RESERVED                         */
/**********************************************************************/

#include "FCCPFCEnergy.h"

#include <cmath>

registerMooseObject("MarlinApp", FCCPFCEnergy);

InputParameters
FCCPFCEnergy::validParams()
{
  InputParameters params = TensorOperator<>::validParams();
  params.addClassDescription(
      "Computes the total free energy for the two-mode FCC phase-field crystal model.");
  params.addRequiredParam<TensorInputBufferName>("psi", "Phase field variable buffer.");
  params.addParam<Real>("eps", -0.5, "Undercooling parameter.");
  params.addParam<Real>("Q1", std::sqrt(4.0 / 3.0), "First mode wave number.");
  params.addParam<Real>("R1", 0.0, "Second-mode strength (>= 0).");
  return params;
}

FCCPFCEnergy::FCCPFCEnergy(const InputParameters & parameters)
  : TensorOperator<>(parameters),
    _psi(getInputBuffer("psi")),
    _eps(getParam<Real>("eps")),
    _q1(getParam<Real>("Q1")),
    _r1(getParam<Real>("R1")),
    _k2(_domain.getKSquare())
{
  if (_r1 < 0.0)
    paramError("R1", "R1 must be non-negative.");
}

void
FCCPFCEnergy::computeBuffer()
{
  // Compute the quartic local term: (1/4) * integral(psi^4) dx
  auto psi4 = _psi * _psi * _psi * _psi;
  auto local_energy = 0.25 * psi4.sum();

  // Transform psi to reciprocal space
  auto psi_hat = _domain.fft(_psi);

  // Compute the linear operator L̂(k) = -eps + (1-k²)²((Q1²-k²)² + R1)
  const auto one_minus_k2 = 1.0 - _k2;
  const auto q1_sq_minus_k2 = _q1 * _q1 - _k2;
  const auto lhat = -_eps + one_minus_k2 * one_minus_k2 * (q1_sq_minus_k2 * q1_sq_minus_k2 + _r1);

  // Compute the gradient/quadratic term: (1/2) * integral(psi_hat* L̂ psi_hat) dk
  // For complex tensors: psi_hat.conj() * lhat * psi_hat gives the pointwise product
  // Real part extracts the physical energy (imaginary part should be ~0 for real psi)
  auto gradient_energy = 0.5 * torch::real(psi_hat.conj() * lhat * psi_hat).sum();

  // Total free energy
  auto total_energy = local_energy + gradient_energy;

  // Store as a constant field for postprocessor access
  _u = torch::full(_domain.getShape(), total_energy.item<Real>(), MooseTensor::floatTensorOptions());
}
