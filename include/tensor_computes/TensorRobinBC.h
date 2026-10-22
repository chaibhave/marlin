/**********************************************************************/
/*                     DO NOT MODIFY THIS HEADER                      */
/*            Marlin, a Fourier spectral solver for MOOSE             */
/*                                                                    */
/*            Copyright 2024 Battelle Energy Alliance, LLC            */
/*                        ALL RIGHTS RESERVED                         */
/**********************************************************************/

#pragma once

#include "TensorBoundaryCondition.h"
#include "FunctionInterface.h"

/**
 * General Robin boundary condition: a*u + b*du/dn = c on the given face,
 * where n is the outward normal. Both the explicit ghost-fill path
 * (fillGhostCells) and the implicit matrix-row/RHS path (stencilContribution)
 * are derived from the same (a, b, c) triple, so the two cannot drift apart.
 * c may vary along the face (c_function or c_buffer); a and b are constants.
 */
class TensorRobinBC : public TensorBoundaryCondition, public FunctionInterface
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

  /// exchange the halo of c_buffer along the face
  virtual void prepareFaceData() override;

  /**
   * Solve a*(u0 + ughost)/2 + b*(ughost - u0)/dx = c for ughost = coefficient*u0 +
   * constant, with dx > 0 the unsigned spacing. The ghost cell lies on the outward
   * side of either face, so the same expression holds on the lower and upper face.
   * Both fillGhostCellsStatic() and stencilContributionStatic() build from these.
   */
  static Real ghostCoefficientOnU0(Real a, Real b, Real dx);
  static Real ghostConstantTerm(Real a, Real b, Real c, Real dx);

  /// Object-independent core of fillGhostCells(), used by the instance method and the unit test.
  /// c is a scalar tensor or has the shape of a ghost slice (size 1 along face_dim).
  static void fillGhostCellsStatic(torch::Tensor & buffer,
                                   bool is_upper_face,
                                   int64_t face_dim,
                                   unsigned int ghost_layers,
                                   int64_t halo,
                                   int64_t owned_extent,
                                   Real a,
                                   Real b,
                                   const torch::Tensor & c,
                                   Real dx);

  /// Object-independent core of stencilContribution(), used by the instance method and the unit test.
  static std::pair<std::vector<FDStencilEntry>, Real>
  stencilContributionStatic(unsigned int dim, Real a, Real b, Real c, Real dx);

protected:
  /// c on this rank's face, shaped like a ghost slice (or a scalar tensor for a constant c)
  torch::Tensor faceValue() const;

  /// c_function at the face position and the tangential cell centres of the ghost slice
  torch::Tensor functionFaceValue(Real t) const;

  /// Robin coefficients: a*u + b*du/dn = c
  const Real _a;
  const Real _b;
  const Real _c;

  /// constant c as a scalar tensor
  const torch::Tensor _c_scalar;

  /// c varying along the face, from a function or from a buffer's first owned layer at the face
  const Function * const _c_function;
  const torch::Tensor * const _c_buffer;

  /// c_function on the face, cached for the time it was evaluated at
  mutable torch::Tensor _c_function_face;
  mutable Real _c_function_time;
};
