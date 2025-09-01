/**********************************************************************/
/*                    DO NOT MODIFY THIS HEADER                       */
/*             Swift, a Fourier spectral solver for MOOSE             */
/*                                                                    */
/*            Copyright 2024 Battelle Energy Alliance, LLC            */
/*                        ALL RIGHTS RESERVED                         */
/**********************************************************************/

#include "ETDRK4Solver.h"
#include "TensorProblem.h"
#include "DomainAction.h"

registerMooseObject("SwiftApp", ETDRK4Solver);

InputParameters
ETDRK4Solver::validParams()
{
  InputParameters params = SplitOperatorBase::validParams();
  params.addClassDescription(
      "Exponential time differencing fourth-order Runge--Kutta semi-implicit solver.");
  params.addParam<unsigned int>("substeps", 1, "semi-implicit substeps per time step.");
  return params;
}

ETDRK4Solver::ETDRK4Solver(const InputParameters & parameters)
  : SplitOperatorBase(parameters),
    _substeps(getParam<unsigned int>("substeps")),
    _sub_dt(_tensor_problem.subDt()),
    _sub_time(_tensor_problem.subTime())
{
  getVariables(0);
}

void
ETDRK4Solver::computeBuffer()
{
  const auto n = _variables.size();

  std::vector<torch::Tensor> ubar_n(n), a_bar(n), N1(n), Na(n), Nb(n), E(n), E2(n), phi1(n),
      phi2(n), phi3(n), phi1h(n);

  _sub_dt = _dt / _substeps;

  for (const auto substep : make_range(_substeps))
  {
    // stage N1
    _compute->computeBuffer();
    forwardBuffers();

    for (const auto i : index_range(_variables))
    {
      auto & var = _variables[i];
      const auto & ubar = var._reciprocal_buffer;
      const auto * Lp = var._linear_reciprocal;
      const auto & N = var._nonlinear_reciprocal;

      ubar_n[i] = ubar.clone();
      N1[i] = N.clone();

      torch::Tensor L = Lp ? *Lp : torch::zeros_like(N);
      auto Ldt = _sub_dt * L;
      E[i] = torch::exp(Ldt);
      E2[i] = torch::exp(Ldt / 2.0);
      auto eps = 1e-12;
      phi1[i] = torch::where(torch::abs(Ldt) < eps, torch::ones_like(Ldt), (E[i] - 1.0) / Ldt);
      phi2[i] = torch::where(
          torch::abs(Ldt) < eps, 0.5 * torch::ones_like(Ldt), (E[i] - 1.0 - Ldt) / (Ldt * Ldt));
      phi3[i] = torch::where(torch::abs(Ldt) < eps,
                             (1.0 / 6.0) * torch::ones_like(Ldt),
                             (E[i] - 1.0 - Ldt - 0.5 * Ldt * Ldt) / (Ldt * Ldt * Ldt));
      phi1h[i] = torch::where(
          torch::abs(Ldt / 2.0) < eps, torch::ones_like(Ldt), (E2[i] - 1.0) / (Ldt / 2.0));

      a_bar[i] = E2[i] * ubar_n[i] + _sub_dt * phi1h[i] * N1[i];
      var._buffer = _domain.ifft(a_bar[i]);
    }

    // stage Na
    _compute->computeBuffer();
    forwardBuffers();
    for (const auto i : index_range(_variables))
    {
      auto & var = _variables[i];
      Na[i] = var._nonlinear_reciprocal.clone();
      auto b_bar = E2[i] * ubar_n[i] + _sub_dt * phi1h[i] * Na[i];
      var._buffer = _domain.ifft(b_bar);
    }

    // stage Nb
    _compute->computeBuffer();
    forwardBuffers();
    for (const auto i : index_range(_variables))
    {
      auto & var = _variables[i];
      Nb[i] = var._nonlinear_reciprocal.clone();
      auto c_bar = E2[i] * a_bar[i] + _sub_dt * phi1h[i] * (2.0 * Nb[i] - N1[i]);
      var._buffer = _domain.ifft(c_bar);
    }

    // stage Nc and final update
    _compute->computeBuffer();
    forwardBuffers();
    for (const auto i : index_range(_variables))
    {
      auto & var = _variables[i];
      const auto Nc = var._nonlinear_reciprocal;
      auto ubar_new = E[i] * ubar_n[i] +
                      _sub_dt * (N1[i] * phi1[i] + 2.0 * (Na[i] + Nb[i]) * phi2[i] + Nc * phi3[i]);
      var._buffer = _domain.ifft(ubar_new);
    }

    if (substep < _substeps - 1)
      _tensor_problem.advanceState();

    _sub_time += _sub_dt;
  }
}
