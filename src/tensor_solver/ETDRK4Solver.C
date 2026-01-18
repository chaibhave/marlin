/**********************************************************************/
/*                     DO NOT MODIFY THIS HEADER                      */
/*            Marlin, a Fourier spectral solver for MOOSE             */
/*                                                                    */
/*            Copyright 2024 Battelle Energy Alliance, LLC            */
/*                        ALL RIGHTS RESERVED                         */
/**********************************************************************/

#include "ETDRK4Solver.h"
#include "TensorProblem.h"
#include "DomainAction.h"

registerMooseObject("MarlinApp", ETDRK4Solver);

InputParameters
ETDRK4Solver::validParams()
{
  InputParameters params = SplitOperatorBase::validParams();
  params.addClassDescription("Fourth-order exponential time differencing solver.");
  params.addParam<unsigned int>("substeps", 1, "Substeps per time step.");

  // Off-diagonal linear operator specification (optional)
  params.addParam<std::vector<unsigned int>>("linear_offdiag_rows", {}, "Row indices for L_ij.");
  params.addParam<std::vector<unsigned int>>("linear_offdiag_cols", {}, "Column indices for L_ij.");
  params.addParam<std::vector<TensorInputBufferName>>(
      "linear_offdiag", {}, "Off-diagonal linear operator buffers.");
  params.addParam<bool>("assume_symmetric",
                        false,
                        "Mirror off-diagonal entries (i,j) into (j,i) if not explicitly provided.");
  return params;
}

ETDRK4Solver::ETDRK4Solver(const InputParameters & parameters)
  : SplitOperatorBase(parameters),
    _substeps(getParam<unsigned int>("substeps")),
    _assume_symmetric(getParam<bool>("assume_symmetric")),
    _L_offdiag_indices(getParam<unsigned int, unsigned int>("linear_offdiag_rows",
                                                            "linear_offdiag_cols")),
    _L_offdiag_names(getParam<std::vector<TensorInputBufferName>>("linear_offdiag")),
    _sub_dt(_tensor_problem.subDt()),
    _sub_time(_tensor_problem.subTime())
{
  getVariables(1);

  if (_L_offdiag_indices.size() != _L_offdiag_names.size())
    paramError("linear_offdiag",
               "'linear_offdiag_rows', 'linear_offdiag_cols', and 'linear_offdiag' must all have "
               "the same length.");

  const auto N = _variables.size();
  for (const auto & [i, j] : _L_offdiag_indices)
  {
    if (i >= N)
      paramError("linear_offdiag_rows", "Off-diagonal indices out of range.");
    if (j >= N)
      paramError("linear_offdiag_cols", "Off-diagonal indices out of range.");
  }

  for (const auto & name : _L_offdiag_names)
    _L_offdiag_buffer.push_back(&getInputBufferByName(name));
}

void
ETDRK4Solver::computeBuffer()
{
  _sub_dt = _dt / _substeps;

  const bool has_offdiag = !_L_offdiag_buffer.empty();

  auto evaluate_nonlinear = [&](const std::vector<torch::Tensor> & ubar_stage)
  {
    for (const auto i : index_range(_variables))
      _variables[i]._buffer = _domain.ifft(ubar_stage[i]);

    _compute->computeBuffer();
    forwardBuffers();

    std::vector<torch::Tensor> nonlinear(_variables.size());
    for (const auto i : index_range(_variables))
      nonlinear[i] = _variables[i]._nonlinear_reciprocal;

    return nonlinear;
  };
  std::cout << "Starting to sub-cycle" << std::endl;
  // subcycles
  for (const auto substep : make_range(_substeps))
  {
    _compute->computeBuffer();
    forwardBuffers();

    std::vector<torch::Tensor> ubar_n(_variables.size());
    std::vector<torch::Tensor> linear(_variables.size());
    std::vector<torch::Tensor> nonlinear1(_variables.size());

    for (const auto i : index_range(_variables))
    {
      ubar_n[i] = _variables[i]._reciprocal_buffer;
      nonlinear1[i] = _variables[i]._nonlinear_reciprocal;

      if (_variables[i]._linear_reciprocal)
        linear[i] = *_variables[i]._linear_reciprocal;
      else
        linear[i] = torch::zeros_like(ubar_n[i]);
    }

    if (!has_offdiag)
    {
      std::vector<torch::Tensor> ubar_b(_variables.size());
      std::vector<torch::Tensor> ubar_c(_variables.size());
      std::vector<torch::Tensor> ubar_d(_variables.size());
      std::vector<torch::Tensor> expLdt(_variables.size());
      std::vector<torch::Tensor> expHalfLdt(_variables.size());
      std::vector<torch::Tensor> phi1(_variables.size());
      std::vector<torch::Tensor> phi2(_variables.size());
      std::vector<torch::Tensor> phi3(_variables.size());

      for (const auto i : index_range(_variables))
      {
        const auto Ldt = linear[i] * _sub_dt;
        expLdt[i] = torch::exp(Ldt);
        expHalfLdt[i] = torch::exp(Ldt / 2.0);

        const auto denom = Ldt * Ldt * Ldt;
        phi1[i] = _sub_dt * (-4.0 - 3.0 * Ldt + expLdt[i] * (4.0 - Ldt)) / denom;
        phi2[i] = _sub_dt * (2.0 + Ldt + expLdt[i] * (-2.0 + Ldt)) / denom;
        phi3[i] = _sub_dt * (-4.0 - 3.0 * Ldt - Ldt * Ldt + expLdt[i] * (4.0 - Ldt)) / denom;

        const auto zero_mask = Ldt == 0.0;
        if (zero_mask.any().item<bool>())
        {
          const auto dt_tensor = torch::full_like(Ldt, _sub_dt);
          phi1[i] = torch::where(zero_mask, dt_tensor, phi1[i]);
          phi2[i] = torch::where(zero_mask, dt_tensor * dt_tensor / 2.0, phi2[i]);
          phi3[i] = torch::where(zero_mask, dt_tensor * dt_tensor / 6.0, phi3[i]);
        }

        ubar_b[i] = expHalfLdt[i] * ubar_n[i] + 0.5 * _sub_dt * nonlinear1[i];
      }

      const auto nonlinear2 = evaluate_nonlinear(ubar_b);

      for (const auto i : index_range(_variables))
        ubar_c[i] = expHalfLdt[i] * ubar_n[i] + 0.5 * _sub_dt * nonlinear2[i];

      const auto nonlinear3 = evaluate_nonlinear(ubar_c);

      for (const auto i : index_range(_variables))
        ubar_d[i] = expLdt[i] * ubar_n[i] + _sub_dt * nonlinear3[i];

      const auto nonlinear4 = evaluate_nonlinear(ubar_d);

      for (const auto i : index_range(_variables))
      {
        auto ubar = expLdt[i] * ubar_n[i] + phi1[i] * nonlinear1[i] +
                    2.0 * phi2[i] * (nonlinear2[i] + nonlinear3[i]) + phi3[i] * nonlinear4[i];

        _variables[i]._buffer = _domain.ifft(ubar);
      }
    }
    else
    {
      std::cout << "Doing a matrix exponential" << std::endl;
      const auto N = _variables.size();
      auto base_opts = ubar_n[0].options();
      auto base_dtype = ubar_n[0].scalar_type();

      const auto zeros_like_grid = torch::zeros_like(ubar_n[0]);
      std::vector<const torch::Tensor *> Lptr(N * N, &zeros_like_grid);

      for (const auto i : make_range(N))
        Lptr[i * N + i] = &linear[i];

      for (const auto k : index_range(_L_offdiag_buffer))
      {
        const auto & [i, j] = _L_offdiag_indices[k];
        Lptr[i * N + j] = _L_offdiag_buffer[k];
        if (_assume_symmetric && i != j && Lptr[j * N + i] == &zeros_like_grid)
          Lptr[j * N + i] = _L_offdiag_buffer[k];
      }

      using torch::stack;
      std::vector<torch::Tensor> rows;
      rows.reserve(N);
      for (const auto i : make_range(N))
      {
        std::vector<torch::Tensor> cols;
        cols.reserve(N);
        for (const auto j : make_range(N))
          cols.push_back(*Lptr[i * N + j]);
        rows.push_back(stack(cols, -1));
      }

      auto L = stack(rows, -1).to(base_dtype);
      auto Ldt = L * _sub_dt;

      auto expLdt = torch::matrix_exp(Ldt);
      auto expHalfLdt = torch::matrix_exp(Ldt / 2.0);

      const auto Ldt_sizes = Ldt.sizes();
      std::vector<int64_t> eye_shape(Ldt_sizes.begin(), Ldt_sizes.end());
      auto eye = torch::eye(N, base_opts);
      for (const auto d : make_range(Ldt.dim() - 2))
        eye = eye.unsqueeze(0);
      auto I = eye.expand(eye_shape);

      auto Ldt_absmax = Ldt.abs().amax({-2, -1});
      const auto small_mask = Ldt_absmax < 1e-10;
      auto mask_matrix = small_mask.unsqueeze(-1).unsqueeze(-1);

      auto Ldt2 = torch::matmul(Ldt, Ldt);
      auto Ldt3 = torch::matmul(Ldt2, Ldt);

      auto phi1_series = I + 0.5 * Ldt + (1.0 / 6.0) * Ldt2 + (1.0 / 24.0) * Ldt3;
      auto phi2_series = 0.5 * I + (1.0 / 6.0) * Ldt + (1.0 / 24.0) * Ldt2 + (1.0 / 120.0) * Ldt3;
      auto phi3_series = (1.0 / 6.0) * I + (1.0 / 24.0) * Ldt + (1.0 / 120.0) * Ldt2 + (1.0 / 720.0) * Ldt3;

      auto pinvLdt = torch::linalg_pinv(Ldt.to(base_dtype));
      auto phi1_dense = torch::matmul(pinvLdt, expLdt - I);
      auto phi2_dense = torch::matmul(pinvLdt, phi1_dense - I);
      auto phi3_dense = torch::matmul(pinvLdt, phi2_dense - 0.5 * I);

      auto phi1 = torch::where(mask_matrix, phi1_series, phi1_dense);
      auto phi2 = torch::where(mask_matrix, phi2_series, phi2_dense);
      auto phi3 = torch::where(mask_matrix, phi3_series, phi3_dense);

      auto ubar_n_stack = stack(ubar_n, -1).to(base_dtype);
      auto nonlinear1_stack = stack(nonlinear1, -1).to(base_dtype);

      auto ubar_b_stack = torch::matmul(expHalfLdt, ubar_n_stack) + 0.5 * _sub_dt * nonlinear1_stack;
      auto ubar_b = torch::unbind(ubar_b_stack, -1);
      const auto nonlinear2 = evaluate_nonlinear(ubar_b);

      auto nonlinear2_stack = stack(nonlinear2, -1).to(base_dtype);
      auto ubar_c_stack = torch::matmul(expHalfLdt, ubar_n_stack) + 0.5 * _sub_dt * nonlinear2_stack;
      auto ubar_c = torch::unbind(ubar_c_stack, -1);

      const auto nonlinear3 = evaluate_nonlinear(ubar_c);
      auto nonlinear3_stack = stack(nonlinear3, -1).to(base_dtype);
      auto ubar_d_stack = torch::matmul(expLdt, ubar_n_stack) + _sub_dt * nonlinear3_stack;
      auto ubar_d = torch::unbind(ubar_d_stack, -1);

      const auto nonlinear4 = evaluate_nonlinear(ubar_d);
      auto nonlinear4_stack = stack(nonlinear4, -1).to(base_dtype);

      auto nonlinear23 = nonlinear2_stack + nonlinear3_stack;

      auto ubar_all = torch::matmul(expLdt, ubar_n_stack) + torch::matmul(phi1, nonlinear1_stack) +
                      2.0 * torch::matmul(phi2, nonlinear23) + torch::matmul(phi3, nonlinear4_stack);

      auto ubar_solutions = torch::unbind(ubar_all, -1);
      for (const auto i : make_range(N))
        _variables[i]._buffer = _domain.ifft(ubar_solutions[i]);
    }

    if (substep < _substeps - 1)
      _tensor_problem.advanceState();

    _sub_time += _sub_dt;
  }

  _compute->computeBuffer();
  forwardBuffers();
}
