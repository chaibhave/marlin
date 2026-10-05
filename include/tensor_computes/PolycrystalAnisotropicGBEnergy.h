/**********************************************************************/
/*                     DO NOT MODIFY THIS HEADER                      */
/*            Marlin, a Fourier spectral solver for MOOSE             */
/*                                                                    */
/*            Copyright 2024 Battelle Energy Alliance, LLC            */
/*                        ALL RIGHTS RESERVED                         */
/**********************************************************************/

#pragma once

#include "DataFileUtils.h"
#include "TensorOperator.h"

#include <torch/script.h>

/**
 * O(N) replacement for the pairwise anisotropic grain-boundary energy path
 * (PairwiseAnisotropicGBEnergy + AnisotropyTorque + per-pair gamma/coeff
 * ParsedComputes), for an arbitrary number of order parameters N.
 *
 * At construction, nothing device/orientation-dependent is available yet
 * (grain orientations are themselves produced by an [Initialize]-stage
 * compute and are not populated until EXEC_INITIAL). The one-time hull
 * construction described in the class description therefore happens lazily,
 * on the first call to computeBuffer(), guarded by _hulls_built: for each of
 * the N(N-1)/2 grain pairs, a merged convex-hull surrogate of the reciprocal
 * GB-energy surface is built with GrainBoundaryHullBuilder (sampling the
 * shared GB5DOF TorchScript module at libtorch_model_file and running Qhull),
 * and sigma_max, kappa, mu, and the shared linear reciprocal-space
 * coefficient are derived from it once and cached. None of this is repeated
 * on subsequent calls.
 *
 * Every call to computeBuffer() does one chunked pass over all pairs,
 * evaluating each pair's hull surrogate (and its closed-form analytical
 * gradient, replacing autograd) only where that pair's order parameters
 * actually form an interface, and scatter-accumulating directly into
 * per-grain outputs. No per-pair field is ever stored.
 *
 * The inherited 'buffer' output (see TensorOperator) is the *total*
 * (bulk + gradient) free energy density, not just the bulk part: kappa is
 * only known once the hulls are built, so the gradient term
 * 0.5*kappa*sum_i|grad gr_i|^2 is computed here directly from the same
 * per-grain gradients used for the nonlinear update, rather than by a
 * separately-parameterized GradientEnergyDensity object (which expects
 * kappa as a compile-time Real).
 */
class PolycrystalAnisotropicGBEnergy : public TensorOperator<>
{
public:
  static InputParameters validParams();

  PolycrystalAnisotropicGBEnergy(const InputParameters & parameters);

  virtual void computeBuffer() override;

protected:
  /// Builds all pairwise hull surrogates and the derived constants. Runs once,
  /// on the first computeBuffer() call (see class description).
  void buildHulls();

  const unsigned int _n_grains;

  std::vector<const torch::Tensor *> _op;
  std::vector<const torch::Tensor *> _grad;
  std::vector<const torch::Tensor *> _reciprocal;
  const torch::Tensor & _orientations;
  const torch::Tensor * _dealiasing;

  std::vector<torch::Tensor *> _NL;
  torch::Tensor & _linear_coefficient;
  torch::Tensor & _sigma;

  Moose::DataFileUtils::Path _file_path;
  // forward() is not const-qualified
  std::unique_ptr<torch::jit::script::Module> _surrogate;

  const Real _interface_width;
  const Real _L0;
  /// Number of GB5DOF samples used to build each pair's hull (construction only).
  const int64_t _num_samples;
  /// Max samples evaluated by GB5DOF at once while building a hull (construction only).
  const unsigned int _hull_sample_chunk_size;
  /// Max number of spatial grid points processed at once per pair, every step.
  const unsigned int _chunk_size;
  const Real _beta;
  const Real _qhull_merge_tolerance;
  const Real _eps;
  /// Imaginary unit, matching the reciprocal-space buffers' complex dtype (as in AnisotropyTorque).
  const torch::Tensor _imaginary_unit;

  bool _hulls_built;
  Real _gradient_threshold;

  /// Pair index i<j in the same order as _hull_equations/_hull_g.
  std::vector<std::pair<unsigned int, unsigned int>> _pair_index;
  /// Per-pair normalized hull facet equations ([F,dim+1], moved to the simulation
  /// device/precision after being built on CPU/float64).
  std::vector<torch::Tensor> _hull_equations;
  /// Per-pair constant facet gradient g_k = c_k / (-d_k), shape [F,dim].
  std::vector<torch::Tensor> _hull_g;

  Real _sigma_max;
  Real _kappa;
  Real _mu;
  /// -k^2 * kappa, shared by every grain's Laplacian correction term.
  torch::Tensor _kappa_laplacian_factor;
  /// -k^2 * L0 * kappa, cached once and copied into the linear_coefficient
  /// output buffer every step (never recomputed).
  torch::Tensor _linear_coefficient_const;

  /// Per-step accumulators, preallocated on first use (or on a grid/device/dtype change) and
  /// zeroed in place every call thereafter, rather than torch::zeros()-allocated fresh each step.
  std::vector<torch::Tensor> _bulk_sum;
  std::vector<torch::Tensor> _torque_vec;
  torch::Tensor _sigma_num;
  torch::Tensor _sigma_den;
  torch::Tensor _f0_cross;
  /// Batch size the accumulators above are currently sized for; -1 before the first call.
  int64_t _accum_batch_size;
};
