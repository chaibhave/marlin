/**********************************************************************/
/*                     DO NOT MODIFY THIS HEADER                      */
/*            Marlin, a Fourier spectral solver for MOOSE             */
/*                                                                    */
/*            Copyright 2024 Battelle Energy Alliance, LLC            */
/*                        ALL RIGHTS RESERVED                         */
/**********************************************************************/

// Verifies the hand-derived closed-form analytical gradient of the
// softmax-weighted convex-hull GB-energy surrogate used by
// PolycrystalAnisotropicGBEnergy::computeBuffer(), replacing
// torch::autograd::grad. The forward evaluation and the gradient formula
// below are reproduced exactly from that method so a mismatch here would
// mean the production code is wrong too.

#include "MarlinUtils.h"
#include "MooseError.h"
#include "MooseTypes.h"
#include "gtest/gtest.h"

#include <torch/torch.h>

namespace
{

/// Softmax-hull energy, autograd-differentiable, independent reimplementation
/// of the forward half of PolycrystalAnisotropicGBEnergy::computeBuffer().
torch::Tensor
hullEnergy(const torch::Tensor & n_hat,
           const torch::Tensor & coeffs,
           const torch::Tensor & offsets,
           Real beta,
           Real eps)
{
  const auto denom = torch::matmul(n_hat, coeffs.t());
  auto E_candidate = denom / (-offsets).unsqueeze(0);
  E_candidate = E_candidate.masked_fill(denom <= eps, -1.0e6);
  const auto w = torch::softmax(beta * E_candidate, /*dim=*/1);
  return (w * E_candidate).sum(/*dim=*/1);
}

/// Closed-form analytical gradient of hullEnergy() w.r.t. n_hat, reproduced
/// exactly from PolycrystalAnisotropicGBEnergy::computeBuffer().
torch::Tensor
hullEnergyGradAnalytical(const torch::Tensor & n_hat,
                         const torch::Tensor & coeffs,
                         const torch::Tensor & offsets,
                         Real beta,
                         Real eps)
{
  const auto g_facets = coeffs / (-offsets).unsqueeze(1);

  const auto denom = torch::matmul(n_hat, coeffs.t());
  auto E_candidate = denom / (-offsets).unsqueeze(0);
  E_candidate = E_candidate.masked_fill(denom <= eps, -1.0e6);
  const auto w = torch::softmax(beta * E_candidate, /*dim=*/1);
  const auto sigma = (w * E_candidate).sum(/*dim=*/1);

  const auto wE = w * E_candidate;
  const auto gbar = torch::matmul(w, g_facets);
  const auto cross = torch::matmul(wE, g_facets);
  return gbar + beta * (cross - gbar * sigma.unsqueeze(1));
}

/// Builds the facet equations (outward unit normal c_k, offset d_k <= 0) of
/// a regular octahedron, a simple synthetic hull with F=8 facets in 3D.
torch::Tensor
octahedronEquations(const torch::TensorOptions & options)
{
  // clang-format off
  const std::vector<double> signs = {
    1, 1, 1,   1, 1, -1,   1, -1, 1,   1, -1, -1,
   -1, 1, 1,  -1, 1, -1,  -1, -1, 1,  -1, -1, -1,
  };
  // clang-format on
  auto normals =
      torch::from_blob(const_cast<double *>(signs.data()), {8, 3}, torch::kFloat64).clone();
  normals = normals / normals.norm(2, /*dim=*/1, /*keepdim=*/true);
  const auto offsets = -torch::ones({8}, torch::kFloat64) / std::sqrt(3.0);
  return torch::cat({normals, offsets.unsqueeze(1)}, /*dim=*/1).to(options);
}

} // namespace

TEST(PolycrystalAnisotropicGBEnergyTest, AnalyticalGradientMatchesAutograd)
{
  const auto options = MooseTensor::floatTensorOptions();
  const auto equations = octahedronEquations(options);
  const auto coeffs = equations.slice(/*dim=*/1, 0, 3);
  const auto offsets = equations.select(/*dim=*/1, 3);

  const Real beta = 100.0;
  const Real eps = 1.0e-8;
  const Real epsilon =
      MooseTensor::floatTensorOptions().dtype() == torch::kFloat64 ? 1.0e-9 : 1.0e-4;

  // A batch of normal directions spanning several octants and facet interiors.
  auto n_raw = torch::tensor({{1.0, 0.3, -0.2},
                              {0.1, 1.0, 0.4},
                              {-0.6, -0.3, 0.8},
                              {0.33, 0.33, 0.34},
                              {-1.0, 0.05, 0.02},
                              {0.2, -0.9, 0.1}},
                             options);
  auto n_hat = (n_raw / n_raw.norm(2, /*dim=*/1, /*keepdim=*/true)).detach();
  n_hat.requires_grad_(true);

  const auto sigma = hullEnergy(n_hat, coeffs, offsets, beta, eps);
  const auto grad_autograd =
      torch::autograd::grad({sigma.sum()}, {n_hat}, /*grad_outputs=*/{}, /*retain_graph=*/false)[0];

  const auto grad_analytical = hullEnergyGradAnalytical(n_hat.detach(), coeffs, offsets, beta, eps);

  const auto abs_diff = (grad_autograd - grad_analytical).abs().max().template item<double>();
  EXPECT_NEAR(abs_diff, 0.0, epsilon)
      << "Analytical gradient of the softmax-hull energy does not match autograd.";
}

TEST(PolycrystalAnisotropicGBEnergyTest, AnalyticalGradientMatchesFiniteDifference)
{
  const auto options = MooseTensor::floatTensorOptions();
  const auto equations = octahedronEquations(options);
  const auto coeffs = equations.slice(/*dim=*/1, 0, 3);
  const auto offsets = equations.select(/*dim=*/1, 3);

  const Real beta = 100.0;
  const Real eps = 1.0e-8;
  const Real h = MooseTensor::floatTensorOptions().dtype() == torch::kFloat64 ? 1.0e-6 : 1.0e-3;
  const Real rel_tolerance =
      MooseTensor::floatTensorOptions().dtype() == torch::kFloat64 ? 1.0e-4 : 1.0e-2;

  auto n_raw = torch::tensor({{0.5, 0.5, 0.1}, {-0.2, 0.7, -0.4}, {0.9, -0.1, -0.3}}, options);
  const auto n_hat = n_raw / n_raw.norm(2, /*dim=*/1, /*keepdim=*/true);

  const auto grad_analytical = hullEnergyGradAnalytical(n_hat, coeffs, offsets, beta, eps);

  // Central finite difference, one component at a time, un-normalized (the
  // analytical formula is the gradient w.r.t. the (already unit) n_hat
  // argument itself, not w.r.t. an unconstrained direction).
  for (int64_t c = 0; c < 3; ++c)
  {
    auto perturbation = torch::zeros_like(n_hat);
    perturbation.select(1, c).fill_(h);

    const auto sigma_plus = hullEnergy(n_hat + perturbation, coeffs, offsets, beta, eps);
    const auto sigma_minus = hullEnergy(n_hat - perturbation, coeffs, offsets, beta, eps);
    const auto fd = (sigma_plus - sigma_minus) / (2.0 * h);

    const auto analytical_component = grad_analytical.select(1, c);
    const auto abs_diff = (fd - analytical_component).abs();
    const auto rel_error = (abs_diff / (fd.abs() + h)).max().template item<double>();

    EXPECT_TRUE(rel_error < rel_tolerance)
        << "Analytical gradient component " << c
        << " disagrees with finite difference, relative error " << rel_error;
  }
}
