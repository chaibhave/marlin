/**********************************************************************/
/*                     DO NOT MODIFY THIS HEADER                      */
/*            Marlin, a Fourier spectral solver for MOOSE             */
/*                                                                    */
/*            Copyright 2024 Battelle Energy Alliance, LLC            */
/*                        ALL RIGHTS RESERVED                         */
/**********************************************************************/

#pragma once

#include "SplitOperatorBase.h"

/**
 * Exponential Time Differencing Runge-Kutta solver (fourth order).
 */
class ETDRK4Solver : public SplitOperatorBase
{
public:
  static InputParameters validParams();

  ETDRK4Solver(const InputParameters & parameters);

  virtual void computeBuffer() override;

protected:
  /// Number of substeps per time step
  const unsigned int _substeps;

  /// Treat off-diagonal entries (i,j) as (j,i) if not provided
  const bool _assume_symmetric;

  /// off-diagonal specification (i -> row, j -> col)
  std::vector<std::pair<unsigned int, unsigned int>> _L_offdiag_indices;
  std::vector<TensorInputBufferName> _L_offdiag_names;
  std::vector<const torch::Tensor *> _L_offdiag_buffer;

  /// references to the substep dt/time managed by the TensorProblem
  Real & _sub_dt;
  Real & _sub_time;
};
