/**********************************************************************/
/*                    DO NOT MODIFY THIS HEADER                       */
/*             Swift, a Fourier spectral solver for MOOSE             */
/*                                                                    */
/*            Copyright 2024 Battelle Energy Alliance, LLC            */
/*                        ALL RIGHTS RESERVED                         */
/**********************************************************************/

#pragma once

#include "TensorOperator.h"

/**
 * ReciprocalRobinBC object
 */

class ReciprocalRobinBC : public TensorOperator<>
{
public:
  static InputParameters validParams();

  ReciprocalRobinBC(const InputParameters & parameters);

  virtual void computeBuffer() override;

protected:
  const torch::Tensor & _chem_pot;
  const torch::Tensor & _M;
  const torch::Tensor & _W_N;
  const torch::Tensor & _B_N;
  const torch::Tensor & _B_D;
  /// imaginary unit i
  const torch::Tensor _imag;
  const torch::Tensor & _psi;

  bool _update_psi;
  const bool _always_update_psi;

  torch::Tensor _psi_thresh;
  torch::Tensor _grad_psi_by_psi;
  torch::Tensor _grad_psi_x_by_psi_sq;
  torch::Tensor _grad_psi_y_by_psi_sq;
  torch::Tensor _grad_psi_z_by_psi_sq;
};
