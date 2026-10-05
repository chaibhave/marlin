/**********************************************************************/
/*                     DO NOT MODIFY THIS HEADER                      */
/*            Marlin, a Fourier spectral solver for MOOSE             */
/*                                                                    */
/*            Copyright 2024 Battelle Energy Alliance, LLC            */
/*                        ALL RIGHTS RESERVED                         */
/**********************************************************************/

#include "GrainBoundaryHullBuilder.h"

#include <libqhullcpp/Qhull.h>
#include <libqhullcpp/QhullFacet.h>
#include <libqhullcpp/QhullFacetList.h>
#include <libqhullcpp/QhullHyperplane.h>

#include <cmath>
#include <iomanip>
#include <sstream>
#include <string>
#include <vector>

namespace MarlinGBHull
{

torch::Tensor
fibonacciSphere(int64_t n, const torch::TensorOptions & options)
{
  const auto i = torch::arange(n, options);
  const double golden_ratio = (1.0 + std::sqrt(5.0)) / 2.0;

  const auto z = 1.0 - 2.0 * (i + 0.5) / static_cast<double>(n);
  const auto r = torch::sqrt(torch::clamp(1.0 - z * z, /*min=*/0.0));
  const auto phi = 2.0 * M_PI * i / golden_ratio;

  const auto x = r * torch::cos(phi);
  const auto y = r * torch::sin(phi);

  return torch::stack({x, y, z}, /*dim=*/1);
}

torch::Tensor
labRotationsFromNormals(const torch::Tensor & normals)
{
  const auto & n = normals;
  const auto opts = n.options();

  auto ref = torch::tensor({0.0, 0.0, 1.0}, opts).expand_as(n).clone();
  const auto use_y_ref = ((n * ref).sum(/*dim=*/1)).abs() > 0.99;
  ref.index_put_({use_y_ref}, torch::tensor({0.0, 1.0, 0.0}, opts));

  auto t1 = ref - (ref * n).sum(/*dim=*/1, /*keepdim=*/true) * n;
  t1 = t1 / torch::linalg_norm(t1, c10::nullopt, {1}, /*keepdim=*/true);

  auto t2 = torch::cross(n, t1, /*dim=*/1);
  t2 = t2 / torch::linalg_norm(t2, c10::nullopt, {1}, /*keepdim=*/true);

  return torch::stack({n, t1, t2}, /*dim=*/1);
}

torch::Tensor
normalizeEquations(const torch::Tensor & equations, int64_t dim)
{
  auto coeffs = equations.slice(/*dim=*/1, 0, dim).clone();
  auto offsets = equations.select(/*dim=*/1, dim).clone();

  const auto norm = torch::linalg_norm(coeffs, c10::nullopt, {1}, /*keepdim=*/true);
  coeffs = coeffs / norm;
  offsets = offsets / norm.select(/*dim=*/1, 0);

  // For a hull containing the origin, Qhull normally gives offset < 0.
  // Enforce that convention.
  const auto flip = offsets > 0;
  coeffs.index_put_({flip}, coeffs.index({flip}) * -1.0);
  offsets.index_put_({flip}, offsets.index({flip}) * -1.0);

  return torch::cat({coeffs, offsets.unsqueeze(1)}, /*dim=*/1);
}

namespace
{

/// Samples x = n / gamma(n), the reciprocal GB-energy surface, in chunks.
torch::Tensor
sampleInverseEnergySurface(torch::jit::script::Module & gb5dof,
                           const torch::Tensor & R_i,
                           const torch::Tensor & R_j,
                           const torch::Tensor & normals,
                           int64_t chunk_size)
{
  const int64_t num_samples = normals.size(0);
  const int64_t dim = normals.size(1);

  auto pts = torch::empty({num_samples, dim}, normals.options());

  torch::NoGradGuard no_grad;

  for (int64_t start = 0; start < num_samples; start += chunk_size)
  {
    const int64_t length = std::min<int64_t>(chunk_size, num_samples - start);
    const auto normals_chunk = normals.narrow(/*dim=*/0, start, length);

    const auto L_chunk = labRotationsFromNormals(normals_chunk);
    const auto P_chunk = torch::matmul(L_chunk, R_i);
    const auto Q_chunk = torch::matmul(L_chunk, R_j);

    const auto gbe_chunk = gb5dof.forward({P_chunk, Q_chunk}).toTensor().reshape({length});

    pts.narrow(/*dim=*/0, start, length).copy_(normals_chunk / gbe_chunk.unsqueeze(1));
  }

  return pts;
}

/// Runs Qhull on CPU/float64 points and returns the raw (unnormalized, unmerged
/// to a fixed count but already Qhull-coplanar-merged) facet plane equations
/// as [F, dim+1].
torch::Tensor
runQhull(const torch::Tensor & points, double merge_tolerance)
{
  const auto pts_cpu = points.to(torch::kCPU, torch::kFloat64).contiguous();
  const int64_t npoints = pts_cpu.size(0);
  const int64_t dim = pts_cpu.size(1);

  // "C-<tol> Qc": merge facets within <tol> of coplanar (bounding the relative
  // energy error at roughly that tolerance), and keep coplanar points. Mirrors
  // scripts/gb_energy/generate_gb_hulls.py's qhull_options.
  // orgQhull::Qhull's constructor already prepends "qhull" itself before
  // running the command; passing a leading "qhull " word here duplicates it
  // (observed as "qhull qhull Qt" in the resulting QhullError) and the
  // option parser then throws on it. The command here must be options only.
  // std::to_string(double) only emits 6 digits after the decimal point, which
  // silently truncates merge_tolerance to 0 for any value below 1e-6 (its
  // param range check only requires > 0); format explicitly instead.
  std::ostringstream tol_stream;
  tol_stream << std::scientific << std::setprecision(3) << merge_tolerance;
  const std::string qhull_command = "C-" + tol_stream.str() + " Qc";

  orgQhull::Qhull qhull("",
                        static_cast<int>(dim),
                        static_cast<int>(npoints),
                        pts_cpu.data_ptr<double>(),
                        qhull_command.c_str());

  const auto facets = qhull.facetList();

  std::vector<double> equations_data;
  equations_data.reserve(facets.count() * (dim + 1));

  int64_t num_facets = 0;
  for (const auto & facet : facets)
  {
    const auto hyperplane = facet.hyperplane();
    const auto * coords = hyperplane.coordinates();
    for (int64_t d = 0; d < dim; ++d)
      equations_data.push_back(coords[d]);
    equations_data.push_back(hyperplane.offset());
    ++num_facets;
  }

  auto result =
      torch::from_blob(equations_data.data(), {num_facets, dim + 1}, torch::kFloat64).clone();
  return result;
}

} // namespace

HullResult
buildHullEquations(torch::jit::script::Module & gb5dof,
                   const torch::Tensor & R_i,
                   const torch::Tensor & R_j,
                   const HullBuildOptions & options)
{
  const auto sampling_options = torch::TensorOptions().dtype(torch::kFloat64).device(torch::kCPU);

  const auto normals = fibonacciSphere(options.num_samples, sampling_options);
  const auto pts = sampleInverseEnergySurface(
      gb5dof, R_i.to(sampling_options), R_j.to(sampling_options), normals, options.chunk_size);

  const int64_t dim = pts.size(1);
  const auto raw_equations = runQhull(pts, options.qhull_merge_tolerance);
  const auto equations = normalizeEquations(raw_equations, dim);

  // Sharp (unsmoothed) maximum GB energy on this hull: max_k 1/|offset_k|.
  const auto offsets = equations.select(/*dim=*/1, dim);
  const double sigma_max = (1.0 / offsets.abs()).max().item<double>();

  return {equations, sigma_max};
}

} // namespace MarlinGBHull
