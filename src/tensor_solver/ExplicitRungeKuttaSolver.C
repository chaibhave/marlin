/**********************************************************************/
/*                    DO NOT MODIFY THIS HEADER                       */
/*             Swift, a Fourier spectral solver for MOOSE             */
/*                                                                    */
/*            Copyright 2024 Battelle Energy Alliance, LLC            */
/*                        ALL RIGHTS RESERVED                         */
/**********************************************************************/

#include "ExplicitRungeKuttaSolver.h"
#include "TensorProblem.h"
#include "DomainAction.h"

registerMooseObject("SwiftApp", ExplicitRungeKuttaSolver);

InputParameters
ExplicitRungeKuttaSolver::validParams()
{
  InputParameters params = ExplicitSolverBase::validParams();
  params.addClassDescription("Semi-implicit time integration solver.");
  params.addParam<unsigned int>("substeps", 1, "Explicit substeps per time step.");
  params.addParam<unsigned int>("order",4,"number of stages in Runge-Kutta time integration.");
  return params;
}

ExplicitRungeKuttaSolver::ExplicitRungeKuttaSolver(const InputParameters & parameters)
  : ExplicitSolverBase(parameters),
    _substeps(getParam<unsigned int>("substeps")),
    _order(getParam<unsigned int>("order")),
    _sub_dt(_tensor_problem.subDt()),
    _sub_time(_tensor_problem.subTime())
{
}

void
ExplicitRungeKuttaSolver::computeBuffer()
{
  std::vector<torch::Tensor> y_current_hat(_variables.size()), y_next_hat(_variables.size());
  _sub_dt = _dt / _substeps;

  // subcycles
  for (const auto substep : make_range(_substeps))
{
  _compute->computeBuffer();
  // Stage 1
  std::size_t i = 0;
  for (auto& [u, reciprocal_buffer, time_derivative_reciprocal] : _variables)
  {
      y_current_hat[i] = reciprocal_buffer;
      y_next_hat[i] = reciprocal_buffer +  time_derivative_reciprocal * _sub_dt / 6;
      u = _domain.ifft(reciprocal_buffer + time_derivative_reciprocal * _sub_dt / 2);
      ++i;
  }
  _compute->computeBuffer();

  // Stage 2
  i = 0;
  for (auto& [u, reciprocal_buffer, time_derivative_reciprocal] : _variables)
  {
      y_next_hat[i] += time_derivative_reciprocal * _sub_dt / 3;
      u = _domain.ifft(y_current_hat[i] + time_derivative_reciprocal * _sub_dt / 2);
      ++i;
  }
  _compute->computeBuffer();

  // Stage 3
  i = 0;
  for (auto& [u, reciprocal_buffer, time_derivative_reciprocal] : _variables)
  {
      y_next_hat[i] += time_derivative_reciprocal * _sub_dt / 3;
      u = _domain.ifft(y_current_hat[i] + time_derivative_reciprocal * _sub_dt);
      ++i;
  }
  _compute->computeBuffer();

  // Stage 4
  i = 0;
  for (auto& [u, reciprocal_buffer, time_derivative_reciprocal] : _variables)
  {
      u = _domain.ifft(y_next_hat[i] + time_derivative_reciprocal * _sub_dt / 6);
      ++i;
  }
  _compute->computeBuffer();

  if (substep < _substeps - 1)
    _tensor_problem.advanceState();

  // increment substep time
  _sub_time += _sub_dt;
}
}
