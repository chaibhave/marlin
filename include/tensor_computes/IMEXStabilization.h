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
 * Computes the required stabilized terms from a given non-linear term for an implicit-explicit
 * stabilized time integration.
 */

 class IMEXStabilization : public TensorOperator<>
{
public:
  static InputParameters validParams();

  IMEXStabilization(const InputParameters & parameters);

  virtual void computeBuffer() override;

  const torch::Tensor & _NL;
  const torch::Tensor & _v_bar;
  const torch::Tensor & _k2;
};
