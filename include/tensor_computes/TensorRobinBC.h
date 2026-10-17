/**********************************************************************/
/*                     DO NOT MODIFY THIS HEADER                      */
/*            Marlin, a Fourier spectral solver for MOOSE             */
/*                                                                    */
/*            Copyright 2024 Battelle Energy Alliance, LLC            */
/*                        ALL RIGHTS RESERVED                         */
/**********************************************************************/

#pragma once

#include "TensorBoundaryCondition.h"

/**
 * General Robin boundary condition: a*u + b*du/dn = c on the given face,
 * where n is the outward normal. Both the explicit ghost-fill path
 * (fillGhostCells) and the implicit matrix-row/RHS path (stencilAt, from
 * LinearFDOperator, unused until a later PETSc PR) are derived from the same
 * (a, b, c) triple, by construction, so the two cannot drift apart.
 */
class TensorRobinBC : public TensorBoundaryCondition
{
public:
  static InputParameters validParams();

  TensorRobinBC(const InputParameters & parameters);

  virtual void fillGhostCells(torch::Tensor & buffer,
                              unsigned int ghost_layers,
                              int64_t halo,
                              int64_t owned_extent,
                              Real dx) const override;

  virtual std::pair<std::vector<FDStencilEntry>, Real>
  stencilContribution(unsigned int ghost_layers, Real dx) const override;

protected:
  /// Robin coefficients: a*u + b*du/dn = c
  const Real _a;
  const Real _b;
  const Real _c;

  /**
   * Solve a*ubar + b*(u0-ughost)/dx_signed = c for ughost, where ubar is the
   * boundary-face value approximation (u0+ughost)/2 and dx_signed carries the
   * outward-normal sign convention (+dx on the upper face, -dx on the lower
   * face, consistent between fillGhostCells and stencilAt). Shared by both,
   * so a sign or algebra mistake affects both paths identically and the R4
   * dual-path test exercises the real formula, not two independent ones.
   */
  Real ghostCoefficientOnU0(Real dx_signed) const;
  Real ghostConstantTerm(Real dx_signed) const;
};
