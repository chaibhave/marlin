/**********************************************************************/
/*                     DO NOT MODIFY THIS HEADER                      */
/*            Marlin, a Fourier spectral solver for MOOSE             */
/*                                                                    */
/*            Copyright 2024 Battelle Energy Alliance, LLC            */
/*                        ALL RIGHTS RESERVED                         */
/**********************************************************************/

#pragma once

#include <cstdint>

#include <torch/script.h>

/**
 * Builds smooth, differentiable convex-hull surrogates of the reciprocal
 * grain-boundary energy surface, following the same construction as
 * scripts/gb_energy/generate_gb_hulls.py and GrainBoundaryEnergyHull.py:
 * sample x = n / gamma(n) over the sphere using a GB5DOF-type surrogate,
 * take the convex hull of the samples with Qhull, and keep the merged
 * facet plane equations as the hull surrogate.
 *
 * Unlike the offline Python script, hulls are built here at MOOSE object
 * construction time, directly from a single shared GB5DOF TorchScript
 * module and whatever grain orientations the input file supplies, so no
 * per-pair .pt file needs to be pre-generated.
 */
namespace MarlinGBHull
{

/// Options controlling hull sampling and construction. Mirrors the
/// configuration constants at the top of generate_gb_hulls.py.
struct HullBuildOptions
{
  /// Number of GB5DOF evaluations used to sample the reciprocal-energy surface.
  int64_t num_samples = 100000;
  /// Maximum number of samples evaluated by the GB5DOF module at once.
  int64_t chunk_size = 1000;
  /// Qhull facet-merge tolerance (absolute distance in reciprocal-energy space).
  double qhull_merge_tolerance = 1.0e-3;
};

/// A single pair's hull surrogate: normalized facet plane equations
/// ([F, dim+1], unit normal in columns [0,dim), offset <= 0 in column dim)
/// and the sharp (unsmoothed) maximum GB energy on this hull, sigma_max = max_k 1/|offset_k|.
struct HullResult
{
  torch::Tensor equations;
  double sigma_max;
};

/// Deterministic, approximately equal-area sampling of n points on the unit sphere.
torch::Tensor fibonacciSphere(int64_t n, const torch::TensorOptions & options);

/// Orthonormal lab frames [n, t1, t2] (shape [M,3,3]) built from unit normal vectors ([M,3]).
torch::Tensor labRotationsFromNormals(const torch::Tensor & normals);

/// Normalizes convex-hull plane equations ([F, dim+1]) to a unit normal with offset <= 0.
torch::Tensor normalizeEquations(const torch::Tensor & equations, int64_t dim);

/**
 * Builds the merged convex-hull surrogate for one grain pair.
 *
 * @param gb5dof    A GB5DOF-compatible TorchScript module (forward(P, Q) -> energies),
 *                  already moved to CPU/float64 for sampling and Qhull robustness.
 * @param R_i, R_j  Orientation matrices for the two grains ([3,3] each).
 */
HullResult buildHullEquations(torch::jit::script::Module & gb5dof,
                              const torch::Tensor & R_i,
                              const torch::Tensor & R_j,
                              const HullBuildOptions & options);

} // namespace MarlinGBHull
