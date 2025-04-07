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
  params.addParam<unsigned int>("order",4,"number of stages in Runge-Kutta time integration.")
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

// void
// ExplicitRungeKuttaSolver::computeBuffer()
// {
//   torch::Tensor ubar;
//   _sub_dt = _dt / _substeps;

//   // subcycles
//   for (const auto substep : make_range(_substeps))
//   {
//     // re-evaluate the solve compute
//     _compute->computeBuffer();
//     // integrate all variables
//     for (auto & [u, reciprocal_buffer, time_derivative_reciprocal] : _variables)
//     {
//       ubar = reciprocal_buffer + _sub_dt * time_derivative_reciprocal;
//       u = _domain.ifft(ubar);
//     }

//     if (substep < _substeps - 1)
//       _tensor_problem.advanceState();

//     // increment substep time
//     _sub_time += _sub_dt;
//   }
// }

void
ExplicitRungeKuttaSolver::computeBuffer()
{
  torch::Tensor k1, k2, k3, k4;
  std::vector<torch::Tensor> u0, u_temp;
  _sub_dt = _dt / _substeps;

  // subcycles
  for (const auto substep : make_range(_substeps))
  {
    // Save the initial state for all variables
    u0.clear();
    for (auto & [u, reciprocal_buffer, time_derivative_reciprocal] : _variables)
    {
      u0.push_back(u.clone());
    }

    // Stage 1
    _compute->computeBuffer();
    std::vector<torch::Tensor> k1_list;
    for (auto & [u, reciprocal_buffer, time_derivative_reciprocal] : _variables)
    {
      k1_list.push_back(_sub_dt * time_derivative_reciprocal);
    }

    // Stage 2
    for (size_t i = 0; i < _variables.size(); ++i)
    {
      auto & [u, reciprocal_buffer, time_derivative_reciprocal] = _variables[i];
      u_temp.push_back(_domain.ifft(reciprocal_buffer + 0.5 * k1_list[i]));
      u = u_temp.back();
    }
    _compute->computeBuffer();
    std::vector<torch::Tensor> k2_list;
    for (auto & [u, reciprocal_buffer, time_derivative_reciprocal] : _variables)
    {
      k2_list.push_back(_sub_dt * time_derivative_reciprocal);
    }

    // Stage 3
    for (size_t i = 0; i < _variables.size(); ++i)
    {
      auto & [u, reciprocal_buffer, time_derivative_reciprocal] = _variables[i];
      u_temp[i] = _domain.ifft(reciprocal_buffer + 0.5 * k2_list[i]);
      u = u_temp[i];
    }
    _compute->computeBuffer();
    std::vector<torch::Tensor> k3_list;
    for (auto & [u, reciprocal_buffer, time_derivative_reciprocal] : _variables)
    {
      k3_list.push_back(_sub_dt * time_derivative_reciprocal);
    }

    // Stage 4
    for (size_t i = 0; i < _variables.size(); ++i)
    {
      auto & [u, reciprocal_buffer, time_derivative_reciprocal] = _variables[i];
      u_temp[i] = _domain.ifft(reciprocal_buffer + k3_list[i]);
      u = u_temp[i];
    }
    _compute->computeBuffer();
    std::vector<torch::Tensor> k4_list;
    for (auto & [u, reciprocal_buffer, time_derivative_reciprocal] : _variables)
    {
      k4_list.push_back(_sub_dt * time_derivative_reciprocal);
    }

    // Combine stages
    for (size_t i = 0; i < _variables.size(); ++i)
    {
      auto & [u, reciprocal_buffer, time_derivative_reciprocal] = _variables[i];
      ubar = reciprocal_buffer + (k1_list[i] + 2 * k2_list[i] + 2 * k3_list[i] + k4_list[i]) / 6.0;
      u = _domain.ifft(ubar);
    }

    if (substep < _substeps - 1)
      _tensor_problem.advanceState();

    // increment substep time
    _sub_time += _sub_dt;
  }
}
