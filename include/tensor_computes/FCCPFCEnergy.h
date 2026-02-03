/**********************************************************************/
/*                     DO NOT MODIFY THIS HEADER                      */
/*            Marlin, a Fourier spectral solver for MOOSE             */
/*                                                                    */
/*            Copyright 2024 Battelle Energy Alliance, LLC            */
/*                        ALL RIGHTS RESERVED                         */
/**********************************************************************/

#pragma once

#include "TensorOperator.h"

/**
 * Free energy for the two-mode FCC phase-field crystal model.
 */
class FCCPFCEnergy : public TensorOperator<>
{
public:
  static InputParameters validParams();

  FCCPFCEnergy(const InputParameters & parameters);

  virtual void computeBuffer() override;

protected:
  const torch::Tensor & _psi;
  const Real _eps;
  const Real _q1;
  const Real _r1;
  const torch::Tensor & _k2;
};
