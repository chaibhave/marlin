/**********************************************************************/
/*                     DO NOT MODIFY THIS HEADER                      */
/*            Marlin, a Fourier spectral solver for MOOSE             */
/*                                                                    */
/*            Copyright 2024 Battelle Energy Alliance, LLC            */
/*                        ALL RIGHTS RESERVED                         */
/**********************************************************************/

#pragma once

#include "TensorOperator.h"
#include "LinearFDOperator.h"

/**
 * Computes the Laplacian using finite differences with ghost layer support.
 */
class FiniteDifferenceLaplacian : public TensorOperator<torch::Tensor>, public LinearFDOperator
{
public:
  static InputParameters validParams();

  FiniteDifferenceLaplacian(const InputParameters & parameters);

  virtual void realSpaceComputeBuffer() override;

  virtual void computeBuffer() override;

  virtual std::vector<FDStencilEntry> stencilAt() const override;

  /**
   * Builds the isotropic 3-pt or 5-pt Laplacian stencil for a grid of the
   * given dimension and spacing. This is the single source of truth for the
   * stencil coefficients: both computeBuffer() (matrix-free convolution)
   * and stencilAt() build their result from this table, so the two cannot
   * drift apart. No MOOSE/TensorProblem dependency, so it is directly
   * unit-testable.
   */
  static std::vector<FDStencilEntry>
  buildStencil(unsigned int dim, bool wide, const std::vector<Real> & grid_spacing, Real factor);

protected:
  const unsigned int _radius;
  /// input buffer
  const torch::Tensor & _u_in;
  /// prefactor
  const Real _factor;
};
