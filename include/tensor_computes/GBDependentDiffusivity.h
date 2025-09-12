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
 * GBDependentDiffusivity object -> assigns anistropic diffusivity along GBs for phase-field models
 */

class GBDependentDiffusivity : public TensorOperator<>
{
public:
  static InputParameters validParams();

  GBDependentDiffusivity(const InputParameters & parameters);

  virtual void computeBuffer() override;

protected:
  const torch::Tensor & _D_b;
  const torch::Tensor & _D_gb;
  const torch::Tensor & _gb;

  /// imaginary unit i
  const torch::Tensor _imag;
};
