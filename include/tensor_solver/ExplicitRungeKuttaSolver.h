/**********************************************************************/
/*                    DO NOT MODIFY THIS HEADER                       */
/*             Swift, a Fourier spectral solver for MOOSE             */
/*                                                                    */
/*            Copyright 2024 Battelle Energy Alliance, LLC            */
/*                        ALL RIGHTS RESERVED                         */
/**********************************************************************/

#pragma once

#include "ExplicitSolverBase.h"
#include "SplitOperatorBase.h"

/**
 * ExplicitRungeKuttaSolver object
 */
class ExplicitRungeKuttaSolver : public ExplicitSolverBase
{
public:
  static InputParameters validParams();

  ExplicitRungeKuttaSolver(const InputParameters & parameters);

  virtual void computeBuffer() override;

protected:
  unsigned int _substeps;
  unsigned int _order;
  Real & _sub_dt;
  Real & _sub_time;
};
