/**********************************************************************/
/*                    DO NOT MODIFY THIS HEADER                       */
/*             Swift, a Fourier spectral solver for MOOSE             */
/*                                                                    */
/*            Copyright 2024 Battelle Energy Alliance, LLC            */
/*                        ALL RIGHTS RESERVED                         */
/**********************************************************************/

#pragma once

#include "SplitOperatorBase.h"

/**
 * Exponential time differencing fourth-order Runge--Kutta solver
 */
class ETDRK4Solver : public SplitOperatorBase
{
public:
  static InputParameters validParams();

  ETDRK4Solver(const InputParameters & parameters);

  virtual void computeBuffer() override;

protected:
  /// number of semi-implicit substeps per time step
  unsigned int _substeps;
  /// subcycle time step size (dt/substeps)
  Real & _sub_dt;
  /// subcycle time
  Real & _sub_time;
};
