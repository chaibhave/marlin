/**********************************************************************/
/*                     DO NOT MODIFY THIS HEADER                      */
/*            Marlin, a Fourier spectral solver for MOOSE             */
/*                                                                    */
/*            Copyright 2024 Battelle Energy Alliance, LLC            */
/*                        ALL RIGHTS RESERVED                         */
/**********************************************************************/

#include "PolycrystalAnisotropicGBEnergy.h"

#include "DomainAction.h"
#include "GrainBoundaryHullBuilder.h"
#include "MarlinUtils.h"

#include <ATen/TensorIndexing.h>

#include <cmath>
#include <limits>

registerMooseObject("MarlinApp", PolycrystalAnisotropicGBEnergy);

InputParameters
PolycrystalAnisotropicGBEnergy::validParams()
{
  InputParameters params = TensorOperator<>::validParams();

  params.addClassDescription(
      "O(N) anisotropic grain-boundary energy and driving force for N order parameters. "
      "Builds all N(N-1)/2 pairwise convex-hull GB-energy surrogates once (from a shared "
      "GB5DOF TorchScript model and Qhull), then on every call accumulates the nonlinear "
      "update for each order parameter and the total (bulk + gradient) free energy density "
      "directly, without storing any per-pair field. The gradient term uses kappa, which (like "
      "sigma_max) is only known once the hulls are built, so it is computed here rather than by "
      "a separately-parameterized GradientEnergyDensity object. The 'buffer' parameter (from "
      "TensorOperator) receives this total free energy density.");

  params.addRequiredParam<std::vector<TensorInputBufferName>>(
      "op_buffers", "Order parameter buffers, one per grain.");
  params.addRequiredParam<std::vector<TensorInputBufferName>>(
      "grad_buffers",
      "Gradient vector buffers (e.g. from GradientVector), one per grain, matching op_buffers.");
  params.addRequiredParam<std::vector<TensorInputBufferName>>(
      "reciprocal_buffers",
      "Reciprocal-space order parameter buffers (e.g. from ForwardFFT), one per grain, matching "
      "op_buffers.");
  params.addRequiredParam<TensorInputBufferName>(
      "orientations",
      "Grain orientation buffer ([N,3,3] rotation matrices), e.g. from "
      "GrainOrientationsFromInput or GrainOrientationsFromFile.");
  params.addParam<TensorInputBufferName>(
      "dealiasing_buffer",
      "Optional de-aliasing filter (e.g. from DeAliasingTensor) applied to every NL output "
      "before it is returned, so the input file does not need a separate multiply per grain.");

  params.addRequiredParam<std::vector<TensorOutputBufferName>>(
      "NL", "Nonlinear update buffers (reciprocal space), one per grain, matching op_buffers.");
  params.addRequiredParam<TensorOutputBufferName>(
      "linear_coefficient",
      "Shared linear reciprocal-space coefficient (-k^2 * L0 * kappa), one value for every "
      "order parameter.");
  params.addRequiredParam<TensorOutputBufferName>("sigma", "Interpolated local GB energy.");

  params.addRequiredParam<DataFileName>("libtorch_model_file",
                                        "Path to the shared GB5DOF TorchScript model.");

  params.addRequiredRangeCheckedParam<Real>(
      "interface_width", "interface_width > 0", "Coarsest interface width in the model.");
  params.addRequiredParam<Real>("L0", "Linear interface mobility.");

  params.addRangeCheckedParam<unsigned int>(
      "num_samples", 100000, "num_samples > 0", "GB5DOF samples used to build each pair's hull.");
  params.addRangeCheckedParam<unsigned int>(
      "hull_sample_chunk_size",
      1000,
      "hull_sample_chunk_size > 0",
      "Maximum number of samples evaluated by GB5DOF at once while building a hull.");
  params.addRangeCheckedParam<unsigned int>(
      "chunk_size",
      65536,
      "chunk_size > 0",
      "Maximum number of spatial points processed at once per pair, every step.");
  params.addRangeCheckedParam<Real>(
      "beta", 100.0, "beta > 0", "Softmax sharpness of the hull surrogate.");
  params.addRangeCheckedParam<Real>("qhull_merge_tolerance",
                                    1.0e-3,
                                    "qhull_merge_tolerance > 0",
                                    "Qhull coplanar facet-merge tolerance.");
  params.addRangeCheckedParam<Real>(
      "eps", 1.0e-8, "eps > 0", "Facet-validity threshold for the hull surrogate.");

  return params;
}

PolycrystalAnisotropicGBEnergy::PolycrystalAnisotropicGBEnergy(const InputParameters & parameters)
  : TensorOperator<>(parameters),
    _n_grains(getParam<std::vector<TensorInputBufferName>>("op_buffers").size()),
    _orientations(getInputBuffer<torch::Tensor>("orientations")),
    _dealiasing(isParamValid("dealiasing_buffer")
                    ? &getInputBuffer<torch::Tensor>("dealiasing_buffer")
                    : nullptr),
    _linear_coefficient(getOutputBuffer<torch::Tensor>("linear_coefficient")),
    _sigma(getOutputBuffer<torch::Tensor>("sigma")),
    _file_path(Moose::DataFileUtils::getPath(getParam<DataFileName>("libtorch_model_file"))),
    _surrogate(std::make_unique<torch::jit::script::Module>(torch::jit::load(_file_path.path))),
    _interface_width(getParam<Real>("interface_width")),
    _L0(getParam<Real>("L0")),
    _num_samples(getParam<unsigned int>("num_samples")),
    _hull_sample_chunk_size(getParam<unsigned int>("hull_sample_chunk_size")),
    _chunk_size(getParam<unsigned int>("chunk_size")),
    _beta(getParam<Real>("beta")),
    _qhull_merge_tolerance(getParam<Real>("qhull_merge_tolerance")),
    _eps(getParam<Real>("eps")),
    _imaginary_unit(
        torch::tensor(c10::complex<double>(0.0, 1.0), MooseTensor::complexFloatTensorOptions())),
    _hulls_built(false),
    _sigma_max(0),
    _kappa(0),
    _mu(0),
    _accum_batch_size(-1)
{
  const auto & op_names = getParam<std::vector<TensorInputBufferName>>("op_buffers");
  const auto & grad_names = getParam<std::vector<TensorInputBufferName>>("grad_buffers");
  const auto & reciprocal_names =
      getParam<std::vector<TensorInputBufferName>>("reciprocal_buffers");
  const auto & NL_names = getParam<std::vector<TensorOutputBufferName>>("NL");

  if (grad_names.size() != _n_grains || reciprocal_names.size() != _n_grains ||
      NL_names.size() != _n_grains)
    paramError("grad_buffers",
               "op_buffers, grad_buffers, reciprocal_buffers, and NL must all list the same "
               "number of grains.");

  if (_n_grains < 2)
    paramError("op_buffers", "At least two order parameters are required.");

  for (const auto & name : op_names)
    _op.push_back(&getInputBufferByName(name));
  for (const auto & name : grad_names)
    _grad.push_back(&getInputBufferByName(name));
  for (const auto & name : reciprocal_names)
    _reciprocal.push_back(&getInputBufferByName(name));
  for (const auto & name : NL_names)
    _NL.push_back(&getOutputBufferByName(name));

  // _surrogate is only ever used for hull construction (buildHulls(), always
  // on CPU/float64 for Qhull robustness - see there), so it is kept exactly
  // as exported by export_gb5dof.py (CPU/float64) rather than moved to the
  // simulation device/precision.
  _surrogate->eval();

  // See AnisotropicGBEnergy/PairwiseAnisotropicGBEnergy for the derivation of
  // this threshold from the tanh interface profile.
  _gradient_threshold = 2 / (cosh(4) * cosh(4)) / _interface_width;

  for (unsigned int i = 0; i < _n_grains; ++i)
    for (unsigned int j = i + 1; j < _n_grains; ++j)
      _pair_index.emplace_back(i, j);
}

void
PolycrystalAnisotropicGBEnergy::buildHulls()
{
  // Grain orientations are produced by an [Initialize]-stage compute and are
  // not populated until EXEC_INITIAL, after this object's constructor and
  // init() have already run. Building the hulls lazily, here, on the first
  // call, is the earliest point at which real orientation data is available.
  if (static_cast<unsigned int>(_orientations.size(0)) != _n_grains || _orientations.size(1) != 3 ||
      _orientations.size(2) != 3)
    mooseError("orientations buffer must have shape [n_grains, 3, 3]; got ", _orientations.sizes());

  // Hull construction runs on CPU/float64 for Qhull robustness and
  // reproducibility, independent of the simulation's device/precision.
  // _surrogate is already CPU/float64 (see constructor).
  const auto hull_options = torch::TensorOptions().dtype(torch::kFloat64).device(torch::kCPU);
  auto & gb5dof_cpu = *_surrogate;

  // Two-step conversion: MPS does not support float64 at all, even as a
  // transient step within a single combined device+dtype .to() call, so the
  // device move and the dtype cast cannot be combined when the source
  // tensor lives on MPS.
  const auto orientations_cpu = _orientations.to(torch::kCPU).to(hull_options);

  MarlinGBHull::HullBuildOptions options;
  options.num_samples = _num_samples;
  options.chunk_size = _hull_sample_chunk_size;
  options.qhull_merge_tolerance = _qhull_merge_tolerance;

  _hull_equations.resize(_pair_index.size());
  _hull_g.resize(_pair_index.size());
  _sigma_max = 0;

  const auto compute_options = MooseTensor::floatTensorOptions();

  for (std::size_t p = 0; p < _pair_index.size(); ++p)
  {
    const auto i = _pair_index[p].first;
    const auto j = _pair_index[p].second;

    const auto R_i = orientations_cpu.select(0, i);
    const auto R_j = orientations_cpu.select(0, j);
    const auto result = MarlinGBHull::buildHullEquations(gb5dof_cpu, R_i, R_j, options);

    _sigma_max = std::max(_sigma_max, result.sigma_max);

    const auto equations = result.equations.to(compute_options);
    const auto coeffs = equations.slice(/*dim=*/1, 0, 3);
    const auto offsets = equations.select(/*dim=*/1, 3);

    _hull_equations[p] = equations;
    _hull_g[p] = coeffs / (-offsets).unsqueeze(1);

    _console << "PolycrystalAnisotropicGBEnergy: built hull for pair (" << i << ", " << j
             << ") with " << equations.size(0) << " facets" << std::endl;
  }

  // kappa/mu/inv_sqrt_kappa_mu all divide by sigma_max (directly or via it), so a degenerate
  // hull set (every pair's hull passing through, or arbitrarily close to, the origin) would
  // otherwise silently propagate Inf/NaN through the whole energy and nonlinear update instead
  // of failing clearly here, at the one point sigma_max is known.
  if (_sigma_max <= 0)
    mooseError("PolycrystalAnisotropicGBEnergy: sigma_max is non-positive (",
               _sigma_max,
               "); every pairwise hull is degenerate (passes through, or arbitrarily close to, "
               "the origin). Check the GB5DOF model and grain orientations.");

  const Real g_gamma0 = std::sqrt(2.0) / 3.0; // g(gamma=1.5)
  const Real f0_gamma0 = 1.0 / 8.0;

  _kappa = _sigma_max * _interface_width * std::sqrt(f0_gamma0) / g_gamma0;
  _mu = _sigma_max / (g_gamma0 * _interface_width * std::sqrt(f0_gamma0));

  _kappa_laplacian_factor = -_domain.getKSquare() * _kappa;
  _linear_coefficient_const = _L0 * _kappa_laplacian_factor;

  _hulls_built = true;
}

void
PolycrystalAnisotropicGBEnergy::computeBuffer()
{
  if (!_hulls_built)
    buildHulls();

  _linear_coefficient = _linear_coefficient_const;

  const auto & grad0 = *_grad[0];
  if (grad0.dim() < 1 || grad0.size(-1) != 3)
    mooseError("grad buffers must have trailing component dimension of size 3.");

  const int64_t batch_size = grad0.numel() / 3;
  const std::vector<int64_t> output_shape(grad0.sizes().begin(), grad0.sizes().end() - 1);

  // Every grain's op/grad/reciprocal buffer must describe the same grid, or every per-pair
  // elementwise combination below silently mixes up points between grains instead of failing
  // clearly (as the old PairwiseAnisotropicGBEnergy/GradientEnergyDensity/AnisotropyTorque each
  // checked explicitly for their own buffer pairs).
  for (unsigned int g = 1; g < _n_grains; ++g)
    if (_grad[g]->sizes() != grad0.sizes())
      mooseError("grad_buffers must all have identical shapes; grad_buffers[",
                 g,
                 "] has shape ",
                 _grad[g]->sizes(),
                 ", expected ",
                 grad0.sizes(),
                 ".");
  for (unsigned int g = 0; g < _n_grains; ++g)
    if (_op[g]->sizes() != at::IntArrayRef(output_shape))
      mooseError("op_buffers[",
                 g,
                 "] has shape ",
                 _op[g]->sizes(),
                 ", expected ",
                 at::IntArrayRef(output_shape),
                 " (matching grad_buffers' leading dimensions).");
  for (unsigned int g = 1; g < _n_grains; ++g)
    if (_reciprocal[g]->sizes() != _reciprocal[0]->sizes())
      mooseError("reciprocal_buffers must all have identical shapes; reciprocal_buffers[",
                 g,
                 "] has shape ",
                 _reciprocal[g]->sizes(),
                 ", expected ",
                 _reciprocal[0]->sizes(),
                 ".");

  const auto opts = grad0.options();
  const Real gradient_threshold_sq = _gradient_threshold * _gradient_threshold;
  const Real inv_sqrt_kappa_mu = 1.0 / std::sqrt(_kappa * _mu);

  std::vector<torch::Tensor> grad_flat(_n_grains), op_flat(_n_grains);
  for (unsigned int g = 0; g < _n_grains; ++g)
  {
    grad_flat[g] = _grad[g]->reshape({batch_size, 3});
    op_flat[g] = _op[g]->reshape({batch_size});
  }

  // Persistent accumulators (see header): preallocated once and reused across steps instead of
  // torch::zeros()-allocated fresh every call. Reallocated only if the grid/device/dtype changes.
  const bool need_alloc = _bulk_sum.empty() || _accum_batch_size != batch_size ||
                          _bulk_sum[0].options().device() != opts.device() ||
                          _bulk_sum[0].options().dtype() != opts.dtype();
  if (need_alloc)
  {
    _bulk_sum.assign(_n_grains, torch::Tensor());
    _torque_vec.assign(_n_grains, torch::Tensor());
    for (unsigned int g = 0; g < _n_grains; ++g)
    {
      _bulk_sum[g] = torch::zeros({batch_size}, opts);
      _torque_vec[g] = torch::zeros({batch_size, 3}, opts);
    }
    _sigma_num = torch::zeros({batch_size}, opts);
    _sigma_den = torch::zeros({batch_size}, opts);
    _f0_cross = torch::zeros({batch_size}, opts);
    _accum_batch_size = batch_size;
  }
  else
  {
    for (unsigned int g = 0; g < _n_grains; ++g)
    {
      _bulk_sum[g].zero_();
      _torque_vec[g].zero_();
    }
    _sigma_num.zero_();
    _sigma_den.zero_();
    _f0_cross.zero_();
  }
  auto & bulk_sum = _bulk_sum;
  auto & torque_vec = _torque_vec;
  auto & sigma_num = _sigma_num;
  auto & sigma_den = _sigma_den;
  auto & f0_cross = _f0_cross;

  // Per-grain gradient magnitude and interface-threshold mask, computed once per grain rather
  // than once per pair (every pair used to recompute both grains' magnitudes from scratch).
  // grad_mag_sq is reused below for grad_energy_sum.
  std::vector<torch::Tensor> grad_mag_sq(_n_grains), grad_mask(_n_grains);
  for (unsigned int g = 0; g < _n_grains; ++g)
  {
    grad_mag_sq[g] = (grad_flat[g] * grad_flat[g]).sum(/*dim=*/1);
    grad_mask[g] = grad_mag_sq[g] >= gradient_threshold_sq;
  }

  const int64_t chunk_size = static_cast<int64_t>(_chunk_size);
  const int64_t n_chunks = (batch_size + chunk_size - 1) / chunk_size;

  // Per-grain, per-chunk occupancy (does this grain clear the gradient threshold ANYWHERE in
  // this chunk?). A pair/chunk can only contain an interface point if both grains are occupied
  // there, so this lets whole pairs or chunks be skipped on the host without ever evaluating the
  // hull surrogate on them. Built on-device, then copied to the host in one transfer.
  auto occ = torch::zeros({static_cast<int64_t>(_n_grains), n_chunks},
                          torch::TensorOptions().dtype(torch::kBool).device(opts.device()));
  for (unsigned int g = 0; g < _n_grains; ++g)
    for (int64_t c = 0; c < n_chunks; ++c)
    {
      const int64_t chunk_start = c * chunk_size;
      const int64_t chunk_length = std::min<int64_t>(chunk_size, batch_size - chunk_start);
      occ.index_put_({static_cast<int64_t>(g), c},
                     grad_mask[g].narrow(/*dim=*/0, chunk_start, chunk_length).any());
    }
  const auto occ_cpu = occ.cpu();
  const auto occ_acc = occ_cpu.accessor<bool, 2>();

  // Single chunked pass over all pairs. Nothing per-pair survives past a
  // single chunk iteration; every pair scatter-accumulates directly into the
  // per-grain and global accumulators above.
  for (std::size_t p = 0; p < _pair_index.size(); ++p)
  {
    const auto i = _pair_index[p].first;
    const auto j = _pair_index[p].second;

    bool pair_has_overlap = false;
    for (int64_t c = 0; c < n_chunks; ++c)
      if (occ_acc[i][c] && occ_acc[j][c])
      {
        pair_has_overlap = true;
        break;
      }
    if (!pair_has_overlap)
      continue;

    const auto & grad_i_full = grad_flat[i];
    const auto & grad_j_full = grad_flat[j];
    const auto & op_i_full = op_flat[i];
    const auto & op_j_full = op_flat[j];
    const auto & coeffs = _hull_equations[p].slice(/*dim=*/1, 0, 3);
    const auto & offsets = _hull_equations[p].select(/*dim=*/1, 3);
    const auto & g_facets = _hull_g[p];

    for (int64_t c = 0; c < n_chunks; ++c)
    {
      if (!(occ_acc[i][c] && occ_acc[j][c]))
        continue;

      const int64_t chunk_start = c * chunk_size;
      const int64_t chunk_length = std::min<int64_t>(chunk_size, batch_size - chunk_start);

      const auto grad_i_chunk = grad_i_full.narrow(/*dim=*/0, chunk_start, chunk_length);
      const auto grad_j_chunk = grad_j_full.narrow(/*dim=*/0, chunk_start, chunk_length);
      const auto delta_chunk = grad_i_chunk - grad_j_chunk;
      const auto delta_mag_sq = (delta_chunk * delta_chunk).sum(/*dim=*/1);

      torch::Tensor valid_idx;
      {
        const auto m_i_chunk = grad_mask[i].narrow(/*dim=*/0, chunk_start, chunk_length);
        const auto m_j_chunk = grad_mask[j].narrow(/*dim=*/0, chunk_start, chunk_length);
        const auto valid_mask = m_i_chunk & m_j_chunk & (delta_mag_sq >= gradient_threshold_sq);
        valid_idx = torch::where(valid_mask)[0];
      }

      const int64_t n_valid = valid_idx.size(0);
      if (n_valid == 0)
        continue;

      const auto op_i_valid =
          op_i_full.narrow(/*dim=*/0, chunk_start, chunk_length).index_select(0, valid_idx);
      const auto op_j_valid =
          op_j_full.narrow(/*dim=*/0, chunk_start, chunk_length).index_select(0, valid_idx);

      const auto delta = delta_chunk.index_select(0, valid_idx);
      const auto delta_mag = torch::sqrt(delta_mag_sq.index_select(0, valid_idx)).unsqueeze(1);
      const auto n_hat = delta / delta_mag;

      // Softmax-hull energy and its closed-form analytical gradient w.r.t.
      // n_hat (replacing autograd; see class/plan documentation for the
      // derivation). Each facet's candidate energy E_k(n) = (n.c_k)/(-d_k) is
      // linear in n, so its gradient g_k = c_k/(-d_k) is constant and was
      // precomputed once in buildHulls() as g_facets.
      const auto denom = torch::matmul(n_hat, coeffs.t());
      auto E_candidate = denom / (-offsets).unsqueeze(0);
      E_candidate = E_candidate.masked_fill(denom <= _eps, -1.0e6);
      const auto w = torch::softmax(_beta * E_candidate, /*dim=*/1);
      const auto sigma_valid = (w * E_candidate).sum(/*dim=*/1);

      const auto wE = w * E_candidate;
      const auto gbar = torch::matmul(w, g_facets);
      const auto cross = torch::matmul(wE, g_facets);
      const auto dE_dn = gbar + _beta * (cross - gbar * sigma_valid.unsqueeze(1));

      // Chain rule through n_hat = (grad_i - grad_j)/|grad_i - grad_j|:
      // d(n_hat)/d(grad_i) = (I - n_hat n_hat^T)/|delta|, and the same with a
      // minus sign for grad_j.
      const auto proj = dE_dn - (dE_dn * n_hat).sum(/*dim=*/1, /*keepdim=*/true) * n_hat;
      const auto dsigma_dgrad_i_valid = proj / delta_mag;
      const auto dsigma_dgrad_j_valid = -dsigma_dgrad_i_valid;

      // gamma_ij(sigma_ij) and its derivative w.r.t. sigma_ij: the same
      // closed-form map used identically across the existing anisotropic
      // input files (g_to_gamma_func / dgamma_dsigma), reproduced here.
      const auto g_norm = sigma_valid * inv_sqrt_kappa_mu;
      const auto g2 = g_norm * g_norm;
      const auto poly =
          -3.0944 * g2.pow(4) - 1.8169 * g2.pow(3) + 10.323 * g2.pow(2) - 8.1819 * g2 + 2.0033;
      const auto gamma_ij = 1.0 / poly;
      const auto dgamma_dsigma =
          g_norm * (24.7552 * g2.pow(3) + 10.9014 * g2.pow(2) - 41.292 * g2 + 16.3638) /
          (poly * poly) * inv_sqrt_kappa_mu;

      const auto op_i_sq = op_i_valid * op_i_valid;
      const auto op_j_sq = op_j_valid * op_j_valid;
      const auto coeff_ij = dgamma_dsigma * op_i_sq * op_j_sq;

      bulk_sum[i]
          .narrow(/*dim=*/0, chunk_start, chunk_length)
          .index_add_(0, valid_idx, gamma_ij * op_j_sq);
      bulk_sum[j]
          .narrow(/*dim=*/0, chunk_start, chunk_length)
          .index_add_(0, valid_idx, gamma_ij * op_i_sq);

      torque_vec[i]
          .narrow(/*dim=*/0, chunk_start, chunk_length)
          .index_add_(0, valid_idx, coeff_ij.unsqueeze(1) * dsigma_dgrad_i_valid);
      torque_vec[j]
          .narrow(/*dim=*/0, chunk_start, chunk_length)
          .index_add_(0, valid_idx, coeff_ij.unsqueeze(1) * dsigma_dgrad_j_valid);

      const auto op_ij_sq = op_i_sq * op_j_sq;
      sigma_num.narrow(/*dim=*/0, chunk_start, chunk_length)
          .index_add_(0, valid_idx, op_ij_sq * sigma_valid);
      sigma_den.narrow(/*dim=*/0, chunk_start, chunk_length).index_add_(0, valid_idx, op_ij_sq);
      f0_cross.narrow(/*dim=*/0, chunk_start, chunk_length)
          .index_add_(0, valid_idx, gamma_ij * op_ij_sq);
    }
  }

  // sigma_den is only index_add_'d at interface points (see the pair loop above), so it is
  // exactly 0 wherever no pair touched this point at all. Left as 0/(0+eps) = 0 there, sigma
  // (and so L_NL = L0*sigma/sigma_max) would be 0 away from every interface, making the
  // (L_NL - L0)*kappa_laplacian_i correction term below equal -L0*kappa_laplacian_i instead of
  // vanishing -- a spurious bulk forcing proportional to the order parameter's Laplacian,
  // dragging the interface. Filling these points with sigma_max (matching
  // PairwiseAnisotropicGBEnergy's analogous NaN-fill with the max measured GBE) instead gives
  // L_NL = L0 there, so the correction term is exactly 0, as it should be away from any GB.
  const auto bulk_mask = sigma_den <= 0;
  auto sigma_ratio = sigma_num / (sigma_den + 1.0e-4);
  sigma_ratio = torch::where(bulk_mask, torch::full_like(sigma_ratio, _sigma_max), sigma_ratio);

  _sigma = sigma_ratio.reshape(output_shape);
  const auto L_NL_flat = _L0 * sigma_ratio / _sigma_max;

  auto f0_self_sum = torch::zeros({batch_size}, opts);

  for (unsigned int g = 0; g < _n_grains; ++g)
  {
    const auto & op_g = op_flat[g];
    f0_self_sum = f0_self_sum + op_g.pow(4) / 4.0 - op_g.pow(2) / 2.0;

    // Divergence of this grain's accumulated torque vector field. Linearity
    // of the FFT-based divergence means every pair's contribution can be
    // summed first and the divergence taken once per grain, rather than
    // once per pair (dim forward FFTs + 1 inverse FFT per grain, instead of
    // 2*dim per pair).
    auto torque_i = torch::zeros(output_shape, opts);
    for (const auto d : make_range(_dim))
    {
      const auto component = torque_vec[g].select(/*dim=*/1, d).reshape(output_shape);
      torque_i = torque_i + _domain.ifft(_domain.fft(component) * _domain.getReciprocalAxis(d) *
                                         _imaginary_unit);
    }

    const auto kappa_laplacian_i = _domain.ifft(*_reciprocal[g] * _kappa_laplacian_factor);

    const auto op_g_spatial = op_g.reshape(output_shape);
    const auto bulk_sum_g_spatial = bulk_sum[g].reshape(output_shape);
    const auto L_NL = L_NL_flat.reshape(output_shape);

    const auto bulk_term = -L_NL * _mu *
                               (op_g_spatial.pow(3) - op_g_spatial +
                                2.0 * op_g_spatial * bulk_sum_g_spatial - torque_i) +
                           (L_NL - _L0) * kappa_laplacian_i;

    auto NL_g = _domain.fft(bulk_term);
    if (_dealiasing)
      NL_g = NL_g * *_dealiasing;

    *_NL[g] = NL_g;
  }

  // Normalization constant so that f0 = 0 for a pure grain (one gr_i = 1, the
  // rest 0): the self term there is 1/4 - 1/2 = -1/4 and all cross terms
  // vanish, so this additive constant is always 1/4, independent of N.
  const auto f0 = f0_self_sum + f0_cross + 0.25;

  // Gradient energy density 0.5 * kappa * sum_i |grad gr_i|^2, matching
  // GradientEnergyDensity's formula. kappa is only known once the hulls are
  // built (it depends on sigma_max), so it cannot be supplied as the
  // compile-time Real parameter that object expects; the term is computed
  // here instead, using the same per-grain gradients already gathered above.
  auto grad_energy_sum = torch::zeros({batch_size}, opts);
  for (unsigned int g = 0; g < _n_grains; ++g)
    grad_energy_sum = grad_energy_sum + grad_mag_sq[g];

  _u = (_mu * f0).reshape(output_shape) + 0.5 * _kappa * grad_energy_sum.reshape(output_shape);
}
