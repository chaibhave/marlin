/**********************************************************************/
/*                     DO NOT MODIFY THIS HEADER                      */
/*            Marlin, a Fourier spectral solver for MOOSE             */
/*                                                                    */
/*            Copyright 2024 Battelle Energy Alliance, LLC            */
/*                        ALL RIGHTS RESERVED                         */
/**********************************************************************/

#include "FiniteDifferenceLaplacian.h"
#include "gtest/gtest.h"

#include <map>
#include <vector>

namespace
{
// Sum of stencilAt()'s coefficients applied to u(x) = x^2 (per axis, isotropic)
// against the known analytic Laplacian 2*dim, as an independent sanity check
// that stencilAt() produces a consistent discrete Laplacian: for u = sum_axis
// x_axis^2 sampled at offsets {-2,-1,0,1,2}*dx from a point, each axis'
// stencil row applied to x^2 recovers exactly 2 (to machine precision for the
// 3-pt stencil, and to O(dx^4) truncation, i.e. exactly for a pure quadratic,
// for the 5-pt stencil), independent of the convolution-kernel code path in
// computeBuffer().
Real
applyStencilToQuadratic(const std::vector<FDStencilEntry> & stencil, const std::vector<Real> & dx)
{
  Real total = 0.0;
  for (const auto & entry : stencil)
  {
    Real x = 0.0;
    for (std::size_t axis = 0; axis < entry.offset.size(); ++axis)
      x += entry.offset[axis] * dx[axis];
    total += entry.coefficient * x * x;
  }
  return total;
}
}

TEST(LinearFDOperatorTest, Stencil3pt1D)
{
  const std::vector<Real> dx{0.1};
  const auto stencil = FiniteDifferenceLaplacian::buildStencil(1, false, dx, 1.0);

  ASSERT_EQ(stencil.size(), 3u);
  EXPECT_NEAR(applyStencilToQuadratic(stencil, dx), 2.0, 1e-10);
}

TEST(LinearFDOperatorTest, Stencil5pt1D)
{
  const std::vector<Real> dx{0.1};
  const auto stencil = FiniteDifferenceLaplacian::buildStencil(1, true, dx, 1.0);

  ASSERT_EQ(stencil.size(), 5u);
  EXPECT_NEAR(applyStencilToQuadratic(stencil, dx), 2.0, 1e-10);
}

TEST(LinearFDOperatorTest, Stencil3pt2D)
{
  const std::vector<Real> dx{0.2, 0.3};
  const auto stencil = FiniteDifferenceLaplacian::buildStencil(2, false, dx, 1.0);

  ASSERT_EQ(stencil.size(), 6u);
  EXPECT_NEAR(applyStencilToQuadratic(stencil, dx), 4.0, 1e-10);
}

TEST(LinearFDOperatorTest, Stencil5pt3D)
{
  const std::vector<Real> dx{0.15, 0.25, 0.35};
  const auto stencil = FiniteDifferenceLaplacian::buildStencil(3, true, dx, 1.0);

  ASSERT_EQ(stencil.size(), 15u);
  EXPECT_NEAR(applyStencilToQuadratic(stencil, dx), 6.0, 1e-10);
}

TEST(LinearFDOperatorTest, StencilAppliesFactor)
{
  const std::vector<Real> dx{0.1};
  const auto stencil = FiniteDifferenceLaplacian::buildStencil(1, false, dx, 2.5);

  EXPECT_NEAR(applyStencilToQuadratic(stencil, dx), 2.0 * 2.5, 1e-10);
}

// Direct coefficient-for-coefficient check against the literal values
// computeBuffer() uses to build its convolution kernel (see
// FiniteDifferenceLaplacian.C), so the two cannot silently diverge.
TEST(LinearFDOperatorTest, Stencil3ptMatchesConvKernelCoefficients)
{
  const std::vector<Real> dx{1.0};
  const auto stencil = FiniteDifferenceLaplacian::buildStencil(1, false, dx, 1.0);

  std::map<int, Real> by_offset;
  for (const auto & entry : stencil)
    by_offset[entry.offset[0]] = entry.coefficient;

  EXPECT_NEAR(by_offset.at(0), -2.0, 1e-12);
  EXPECT_NEAR(by_offset.at(1), 1.0, 1e-12);
  EXPECT_NEAR(by_offset.at(-1), 1.0, 1e-12);
}

TEST(LinearFDOperatorTest, Stencil5ptMatchesConvKernelCoefficients)
{
  const std::vector<Real> dx{1.0};
  const auto stencil = FiniteDifferenceLaplacian::buildStencil(1, true, dx, 1.0);

  std::map<int, Real> by_offset;
  for (const auto & entry : stencil)
    by_offset[entry.offset[0]] = entry.coefficient;

  EXPECT_NEAR(by_offset.at(0), -2.5, 1e-12);
  EXPECT_NEAR(by_offset.at(1), 4.0 / 3.0, 1e-12);
  EXPECT_NEAR(by_offset.at(-1), 4.0 / 3.0, 1e-12);
  EXPECT_NEAR(by_offset.at(2), -1.0 / 12.0, 1e-12);
  EXPECT_NEAR(by_offset.at(-2), -1.0 / 12.0, 1e-12);
}
