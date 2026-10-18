/**********************************************************************/
/*                     DO NOT MODIFY THIS HEADER                      */
/*            Marlin, a Fourier spectral solver for MOOSE             */
/*                                                                    */
/*            Copyright 2024 Battelle Energy Alliance, LLC            */
/*                        ALL RIGHTS RESERVED                         */
/**********************************************************************/

// SPIKE (not a maintained solver): feasibility check for a torch-native
// geometric multigrid preconditioner whose levels each carry their own ghost
// rows and halo exchange, as a distributed implementation would need.
//
// The "ranks" are emulated: the domain is split into P slabs along dimension 0
// inside one process, each slab stored as [owned + 2 ghost rows, n]. The ghost
// exchange (exchange()) copies neighbouring slabs' edge rows with periodic
// wrap, the same pattern as TensorProblem::exchangeGhostLayers for a 1D rank
// partition. No MPI, no TensorProblem buffers and no timing are exercised.
//
// Problem: periodic 2D  A u = -Laplacian(u) + sigma u,  h = 1/n, 5-point
// stencil, cell-centered, n a power of two. V-cycle: red-black Gauss-Seidel
// (red,black before; black,red after, so the cycle is symmetric), bilinear
// prolongation P, restriction P^T/4, rediscretized coarse operators (spacing
// doubles), dense inverse at n=8 (stands in for an allgather plus redundant
// coarse solve). Outer solver: the existing MooseTensor::conjugateGradientSolve.
//
// DISABLED_CudaVcycleTimingSweep times the same V-cycle on one CUDA device
// (single slab, no MPI) against Jacobi-CG, to see whether the many small
// kernel launches on the coarse levels dominate. Run it on a GPU machine with
//   --gtest_also_run_disabled_tests --gtest_filter='*DISABLED_CudaVcycleTimingSweep'
// (optionally under nsys; each phase carries an NVTX range).

#include "MarlinUtils.h"
#include "MooseError.h"
#include "gtest/gtest.h"

#include <torch/cuda.h>

#include <array>
#include <chrono>
#include <string>
#include <vector>

#if defined(__has_include)
#if __has_include(<nvtx3/nvToolsExt.h>)
#include <nvtx3/nvToolsExt.h>
#define MARLIN_SPIKE_HAS_NVTX 1
#endif
#endif

namespace
{
constexpr double sigma = 1.0;
constexpr int64_t coarsest_n = 8;
constexpr double rel_tol = 1e-10;

/// One tensor per emulated rank: [owned rows + 2 ghost rows, n].
using Slabs = std::vector<torch::Tensor>;

class SlabMultigrid
{
public:
  SlabMultigrid(int64_t n, int64_t slabs, torch::Device device = torch::kCPU) : _slabs(slabs)
  {
    for (int64_t level_n = n; level_n >= coarsest_n; level_n /= 2)
    {
      EXPECT_EQ(level_n % slabs, 0);
      _n.push_back(level_n);
      _h2.push_back(1.0 / (static_cast<double>(level_n) * level_n));
      if (level_n == coarsest_n)
        break;
    }

    // Dense coarsest operator, built by applying the periodic stencil to unit
    // vectors, then inverted once.
    const int64_t nc = coarsest_n;
    const double h2 = _h2.back();
    const auto dopts = torch::TensorOptions().dtype(torch::kDouble).device(device);
    auto columns = torch::zeros({nc * nc, nc * nc}, dopts);
    for (int64_t k = 0; k < nc * nc; ++k)
    {
      auto e = torch::zeros({nc, nc}, dopts);
      e[k / nc][k % nc] = 1.0;
      auto Ae = (4.0 * e - torch::roll(e, 1, 0) - torch::roll(e, -1, 0) - torch::roll(e, 1, 1) -
                 torch::roll(e, -1, 1)) /
                    h2 +
                sigma * e;
      columns.select(1, k).copy_(Ae.reshape({-1}));
    }
    _coarse_inverse = torch::inverse(columns);

    // Red/black masks depend only on level, slab and colour: build them once so
    // the smoother launches no index-arithmetic kernels.
    const auto iopts = torch::TensorOptions().dtype(torch::kInt64).device(device);
    for (const int64_t level_n : _n)
    {
      const int64_t m = level_n / slabs;
      const auto cols = torch::arange(level_n, iopts).unsqueeze(0);
      std::vector<std::array<torch::Tensor, 2>> level_masks;
      for (int64_t p = 0; p < slabs; ++p)
      {
        const auto global_rows = (torch::arange(m, iopts) + p * m).unsqueeze(1);
        const auto parity = (global_rows + cols) % 2;
        level_masks.push_back({parity == 0, parity == 1});
      }
      _masks.push_back(level_masks);
    }
  }

  /// A x on the finest level; x and the result are flat length-n^2 vectors.
  torch::Tensor apply(const torch::Tensor & x) const
  {
    auto u = split(x, 0);
    exchange(u);
    Slabs out;
    for (const auto & slab : u)
      out.push_back(applySlab(slab, 0));
    return join(out, 0);
  }

  /// One symmetric V-cycle started from zero: the preconditioner M(r).
  torch::Tensor precondition(const torch::Tensor & r) const
  {
    auto f = split(r, 0);
    exchange(f);
    Slabs u;
    for (const auto & slab : f)
      u.push_back(torch::zeros_like(slab));
    vcycle(0, u, f);
    return join(u, 0);
  }

private:
  int64_t rows(int64_t level) const { return _n[level] / _slabs; }

  /// Flat vector -> slabs with ghost rows allocated (not yet filled).
  Slabs split(const torch::Tensor & flat, int64_t level) const
  {
    const int64_t n = _n[level], m = rows(level);
    auto grid = flat.view({n, n});
    Slabs slabs;
    for (int64_t p = 0; p < _slabs; ++p)
    {
      auto slab = torch::zeros({m + 2, n}, flat.options());
      slab.narrow(0, 1, m).copy_(grid.narrow(0, p * m, m));
      slabs.push_back(slab);
    }
    return slabs;
  }

  torch::Tensor join(const Slabs & slabs, int64_t level) const
  {
    const int64_t m = rows(level);
    std::vector<torch::Tensor> owned;
    for (const auto & slab : slabs)
      owned.push_back(slab.narrow(0, 1, m));
    return torch::cat(owned, 0).reshape({-1});
  }

  /// Periodic ghost-row exchange between neighbouring slabs (the emulated
  /// MPI halo exchange). Reads owned rows, writes ghost rows only.
  void exchange(Slabs & u) const
  {
    const int64_t P = static_cast<int64_t>(u.size());
    const int64_t m = u[0].size(0) - 2;
    for (int64_t p = 0; p < P; ++p)
    {
      u[p][0].copy_(u[(p + P - 1) % P][m]);
      u[p][m + 1].copy_(u[(p + 1) % P][1]);
    }
  }

  /// Owned rows of A u for one slab with fresh ghosts; ghost rows of the
  /// result are zero.
  torch::Tensor applySlab(const torch::Tensor & u, int64_t level) const
  {
    const int64_t m = u.size(0) - 2;
    auto owned = u.narrow(0, 1, m);
    auto out = torch::zeros_like(u);
    out.narrow(0, 1, m).copy_((4.0 * owned - u.narrow(0, 0, m) - u.narrow(0, 2, m) -
                               torch::roll(owned, 1, 1) - torch::roll(owned, -1, 1)) /
                                  _h2[level] +
                              sigma * owned);
    return out;
  }

  /// One red (color 0) or black (color 1) Gauss-Seidel half-sweep on all
  /// slabs, followed by the ghost exchange it needs.
  void smooth(Slabs & u, const Slabs & f, int64_t level, int64_t color) const
  {
    const int64_t m = rows(level);
    for (int64_t p = 0; p < _slabs; ++p)
    {
      auto owned = u[p].narrow(0, 1, m);
      const auto & mask = _masks[level][p][color];
      auto neighbours = u[p].narrow(0, 0, m) + u[p].narrow(0, 2, m) + torch::roll(owned, 1, 1) +
                        torch::roll(owned, -1, 1);
      auto updated = (f[p].narrow(0, 1, m) + neighbours / _h2[level]) / (4.0 / _h2[level] + sigma);
      owned.copy_(torch::where(mask, updated, owned));
    }
    exchange(u);
  }

  /// Fine residual slab (fresh ghosts) -> owned rows of the coarse right-hand
  /// side, R = P^T / 4 with P the bilinear cell-centered prolongation.
  torch::Tensor restrictSlab(const torch::Tensor & fine, int64_t m, int64_t n) const
  {
    const int64_t nc = n / 2, mc = m / 2;
    auto pairs = fine.reshape({m + 2, nc, 2});
    auto even = pairs.select(2, 0), odd = pairs.select(2, 1);
    auto t = 0.75 * (even + odd) + 0.25 * (torch::roll(odd, 1, 1) + torch::roll(even, -1, 1));

    auto own = t.narrow(0, 1, m).reshape({mc, 2, nc});
    auto below = t.narrow(0, 0, m).reshape({mc, 2, nc}).select(1, 0);
    auto above = t.narrow(0, 2, m).reshape({mc, 2, nc}).select(1, 1);
    return (0.75 * (own.select(1, 0) + own.select(1, 1)) + 0.25 * (below + above)) / 4.0;
  }

  /// Coarse correction slab (fresh ghosts) -> owned fine rows, bilinear P.
  torch::Tensor prolongSlab(const torch::Tensor & coarse, int64_t m, int64_t n) const
  {
    const int64_t mc = m / 2;
    auto even1 = 0.75 * coarse + 0.25 * torch::roll(coarse, 1, 1);
    auto odd1 = 0.75 * coarse + 0.25 * torch::roll(coarse, -1, 1);
    auto t = torch::stack({even1, odd1}, 2).reshape({mc + 2, n});

    auto own = t.narrow(0, 1, mc);
    auto even0 = 0.75 * own + 0.25 * t.narrow(0, 0, mc);
    auto odd0 = 0.75 * own + 0.25 * t.narrow(0, 2, mc);
    return torch::stack({even0, odd0}, 1).reshape({m, n});
  }

  void vcycle(int64_t level, Slabs & u, const Slabs & f) const
  {
    const int64_t m = rows(level), n = _n[level];

    if (level + 1 == static_cast<int64_t>(_n.size()))
    {
      // Coarsest level: gather, solve densely, scatter.
      const auto rhs = join(f, level);
      const auto sol = _coarse_inverse.matmul(rhs).reshape({n, n});
      for (int64_t p = 0; p < _slabs; ++p)
        u[p].narrow(0, 1, m).copy_(sol.narrow(0, p * m, m));
      exchange(u);
      return;
    }

    smooth(u, f, level, 0);
    smooth(u, f, level, 1);

    Slabs residual;
    for (int64_t p = 0; p < _slabs; ++p)
      residual.push_back(f[p] - applySlab(u[p], level));
    // residual ghost rows are not valid after the subtraction: refresh them.
    exchange(residual);

    const int64_t mc = m / 2, nc = n / 2;
    Slabs coarse_f, coarse_u;
    for (int64_t p = 0; p < _slabs; ++p)
    {
      auto slab = torch::zeros({mc + 2, nc}, u[p].options());
      slab.narrow(0, 1, mc).copy_(restrictSlab(residual[p], m, n));
      coarse_f.push_back(slab);
      coarse_u.push_back(torch::zeros_like(slab));
    }
    exchange(coarse_f);

    vcycle(level + 1, coarse_u, coarse_f);

    for (int64_t p = 0; p < _slabs; ++p)
      u[p].narrow(0, 1, m).add_(prolongSlab(coarse_u[p], m, n));
    exchange(u);

    smooth(u, f, level, 1);
    smooth(u, f, level, 0);
  }

  int64_t _slabs;
  std::vector<int64_t> _n;
  std::vector<double> _h2;
  torch::Tensor _coarse_inverse;
  std::vector<std::vector<std::array<torch::Tensor, 2>>> _masks; // [level][slab][colour]
};

struct SolveResult
{
  torch::Tensor x;
  unsigned int its;
  double relative_residual;
};

SolveResult
solveMG(int64_t n, int64_t slabs, const torch::Tensor & b)
{
  SlabMultigrid mg(n, slabs);
  auto A = [&](const torch::Tensor & x) { return mg.apply(x); };
  auto M = [&](const torch::Tensor & r) { return mg.precondition(r); };
  const auto [x, its, norm] =
      MooseTensor::conjugateGradientSolve(A, b, torch::Tensor(), rel_tol, 0, M);
  (void)norm;
  return {x, its, (mg.apply(x) - b).norm().item<double>() / b.norm().item<double>()};
}

unsigned int
jacobiCGIterations(int64_t n, const torch::Tensor & b)
{
  SlabMultigrid mg(n, 1);
  const double inv_diag = 1.0 / (4.0 * n * n + sigma);
  auto A = [&](const torch::Tensor & x) { return mg.apply(x); };
  auto M = [&](const torch::Tensor & r) { return r * inv_diag; };
  return std::get<1>(MooseTensor::conjugateGradientSolve(A, b, torch::Tensor(), rel_tol, 0, M));
}
} // namespace

// MG-preconditioned CG converges in a roughly n-independent number of
// iterations, while Jacobi-CG on the same operator grows with n.
TEST(TorchMultigridSpikeTest, IterationCountIsGridIndependent)
{
  torch::manual_seed(0);
  std::vector<unsigned int> mg_its;
  for (const int64_t n : {64, 128, 256, 512})
  {
    const auto b = torch::randn({n * n}, torch::kDouble);
    const auto result = solveMG(n, 1, b);
    mg_its.push_back(result.its);
    Moose::out << "n=" << n << " MG-PCG: " << result.its << " its, relative residual "
               << result.relative_residual << "\n";
    EXPECT_LT(result.relative_residual, 1e-9);
  }
  EXPECT_LE(mg_its.back(), mg_its.front() + 2);

  std::vector<unsigned int> jacobi_its;
  for (const int64_t n : {32, 64, 128})
  {
    const auto b = torch::randn({n * n}, torch::kDouble);
    jacobi_its.push_back(jacobiCGIterations(n, b));
    Moose::out << "n=" << n << " Jacobi-CG: " << jacobi_its.back() << " its\n";
  }
  EXPECT_GT(jacobi_its.back(), 2 * jacobi_its.front());
}

// The emulated slab decomposition must reproduce the single-slab result:
// any error in a ghost row on any level shows up as a difference.
TEST(TorchMultigridSpikeTest, SlabCountDoesNotChangeTheSolution)
{
  torch::manual_seed(1);
  const int64_t n = 128;
  const auto b = torch::randn({n * n}, torch::kDouble);
  const auto reference = solveMG(n, 1, b);

  for (const int64_t slabs : {2, 4})
  {
    const auto result = solveMG(n, slabs, b);
    const double diff = (result.x - reference.x).abs().max().item<double>();
    Moose::out << slabs << " slabs: " << result.its << " its (1 slab: " << reference.its
               << "), max |x - x_1slab| = " << diff << "\n";
    EXPECT_EQ(result.its, reference.its);
    EXPECT_LT(diff, 1e-12 * reference.x.abs().max().item<double>());
  }
}

namespace
{
/// NVTX range around a timed phase (no-op if the NVTX headers are absent).
class Range
{
public:
  explicit Range(const std::string & name)
  {
#ifdef MARLIN_SPIKE_HAS_NVTX
    nvtxRangePushA(name.c_str());
#else
    (void)name;
#endif
  }
  ~Range()
  {
#ifdef MARLIN_SPIKE_HAS_NVTX
    nvtxRangePop();
#endif
  }
};

/// Wall time of fn() in ms, bracketed by device synchronization.
template <typename F>
double
timedMs(const std::string & label, F fn)
{
  torch::cuda::synchronize();
  Range range(label);
  const auto t0 = std::chrono::steady_clock::now();
  fn();
  torch::cuda::synchronize();
  return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
}
} // namespace

// Timing only: no pass/fail on speed. Each solve runs twice (first, repeat) as
// in the PETSc benchmark, since first calls carry one-off costs.
TEST(TorchMultigridSpikeTest, DISABLED_CudaVcycleTimingSweep)
{
  ASSERT_TRUE(torch::cuda::is_available()) << "needs a CUDA-enabled libtorch and a GPU";
  const torch::Device device(torch::kCUDA, 0);
  const auto dopts = torch::TensorOptions().dtype(torch::kDouble).device(device);
  constexpr int vcycle_reps = 20;
  constexpr int64_t jacobi_maxiter = 20000;

  torch::manual_seed(0);
  for (const int64_t n : {256, 512, 1024, 2048, 4096, 8192})
  {
    const std::string tag = " n=" + std::to_string(n);
    SlabMultigrid mg(n, 1, device);
    const auto b = torch::randn({n * n}, dopts);
    const double inv_diag = 1.0 / (4.0 * n * n + sigma);
    auto A = [&](const torch::Tensor & x) { return mg.apply(x); };
    auto M_mg = [&](const torch::Tensor & r) { return mg.precondition(r); };
    auto M_jac = [&](const torch::Tensor & r) { return r * inv_diag; };

    // Warm-up so allocator/cuBLAS setup is out of the isolated kernels.
    (void)mg.precondition(b);
    (void)mg.apply(b);

    const double vcycle_ms = timedMs("vcycle x20" + tag,
                                     [&]
                                     {
                                       for (int i = 0; i < vcycle_reps; ++i)
                                         (void)mg.precondition(b);
                                     }) /
                             vcycle_reps;
    const double matvec_ms = timedMs("matvec x20" + tag,
                                     [&]
                                     {
                                       for (int i = 0; i < vcycle_reps; ++i)
                                         (void)mg.apply(b);
                                     }) /
                             vcycle_reps;

    unsigned int mg_its = 0, jac_its = 0;
    double mg_first = timedMs("MG-PCG first" + tag,
                              [&]
                              {
                                mg_its = std::get<1>(MooseTensor::conjugateGradientSolve(
                                    A, b, torch::Tensor(), rel_tol, 0, M_mg));
                              });
    double mg_repeat = timedMs("MG-PCG repeat" + tag,
                               [&]
                               {
                                 mg_its = std::get<1>(MooseTensor::conjugateGradientSolve(
                                     A, b, torch::Tensor(), rel_tol, 0, M_mg));
                               });
    double jac_first = timedMs("Jacobi-CG first" + tag,
                               [&]
                               {
                                 jac_its = std::get<1>(MooseTensor::conjugateGradientSolve(
                                     A, b, torch::Tensor(), rel_tol, jacobi_maxiter, M_jac));
                               });
    double jac_repeat = timedMs("Jacobi-CG repeat" + tag,
                                [&]
                                {
                                  jac_its = std::get<1>(MooseTensor::conjugateGradientSolve(
                                      A, b, torch::Tensor(), rel_tol, jacobi_maxiter, M_jac));
                                });

    printf("n=%ld V-cycle %.3f ms, matvec %.3f ms (ratio %.1f) | MG-PCG: %u its, %.1f ms, repeat "
           "%.1f ms (%.3f ms/it) | Jacobi-CG: %u its%s, %.1f ms, repeat %.1f ms (%.3f ms/it)\n",
           long(n),
           vcycle_ms,
           matvec_ms,
           vcycle_ms / matvec_ms,
           mg_its,
           mg_first,
           mg_repeat,
           mg_repeat / mg_its,
           jac_its,
           jac_its >= jacobi_maxiter ? " (hit maxiter)" : "",
           jac_first,
           jac_repeat,
           jac_repeat / jac_its);
  }
}
