/**********************************************************************/
/*                     DO NOT MODIFY THIS HEADER                      */
/*            Marlin, a Fourier spectral solver for MOOSE             */
/*                                                                    */
/*            Copyright 2024 Battelle Energy Alliance, LLC            */
/*                        ALL RIGHTS RESERVED                         */
/**********************************************************************/

// SPIKE (not a maintained solver): the TorchMultigridSpikeTest V-cycle on real
// MPI ranks. Partitions, rank index, communicator, owned extents and the
// allreduce come from a REAL_SPACE DomainAction. Every level keeps one ghost
// layer and has its own owned extent (fine owned / 2^level, same rank grid);
// the halo exchange is a free function with the same communication pattern as
// TensorProblem::exchangeGhostLayers (periodic, one dimension at a time over
// the full extent so corners are filled), because that member is tied to a
// named buffer and to the fine-level owned size in _domain.
//
// 2D periodic  A u = -Laplacian(u) + sigma u, h = 1/n, cell-centered, CPU.
// Run:  mpiexec -n {1,2,4} ./marlin-unit-opt --gtest_filter='TorchMultigridMPISpikeTest.*'
// and compare the printed iteration count and solution checksums across rank
// counts. n and the rank grid must satisfy coarsest_n % parts == 0.

#include "ActionFactory.h"
#include "AppFactory.h"
#include "DomainAction.h"
#include "MarlinUtils.h"
#include "MooseApp.h"
#include "MooseError.h"
#include "gtest/gtest.h"

#include <array>
#include <cmath>
#include <memory>
#include <string>
#include <vector>

namespace
{
constexpr double sigma = 1.0;
constexpr int64_t coarsest_n = 8;
constexpr double rel_tol = 1e-10;

struct MarlinDomain
{
  std::unique_ptr<MooseApp> app;
  std::shared_ptr<DomainAction> domain;
};

MarlinDomain
buildDomain(int64_t n)
{
  MarlinDomain md;
  md.app = AppFactory::create("MarlinApp");
  auto & af = md.app->getActionFactory();
  auto params = af.getValidParams("DomainAction");
  params.set<MooseEnum>("dim") = "2";
  params.set<unsigned int>("nx") = n;
  params.set<unsigned int>("ny") = n;
  params.set<unsigned int>("nz") = 1;
  params.set<MooseEnum>("parallel_mode") = "REAL_SPACE";
  params.set<MultiMooseEnum>("periodic_directions") = "X Y";
  params.set<std::string>("task") = "meta_action";
  md.domain = std::dynamic_pointer_cast<DomainAction>(af.create("DomainAction", "Domain", params));
  if (!md.domain)
    throw std::runtime_error("DomainAction construction failed");
  return md;
}

/// Rank layout needed by the halo exchange.
struct Layout
{
  std::array<unsigned int, 2> parts;
  std::array<unsigned int, 2> index;
  std::array<int64_t, 2> begin; // global index of this rank's first owned fine cell
  MPI_Comm comm;
  int toRank(unsigned int ix, unsigned int iy) const
  {
    return static_cast<int>(ix + parts[0] * iy);
  }
};

/// Fill the one-cell ghost frame of t = [mx + 2, my + 2] from the periodic
/// neighbours, dimension 0 then dimension 1 over the full extent.
void
exchange(torch::Tensor & t, const Layout & L)
{
  for (unsigned int d = 0; d < 2; ++d)
  {
    const int64_t m = t.size(d) - 2;
    if (L.parts[d] == 1)
    {
      t.narrow(d, 0, 1).copy_(t.narrow(d, m, 1).clone());
      t.narrow(d, m + 1, 1).copy_(t.narrow(d, 1, 1).clone());
      continue;
    }

    auto neighbor = [&](int shift)
    {
      auto idx = L.index;
      idx[d] = (idx[d] + L.parts[d] + shift) % L.parts[d];
      return L.toRank(idx[0], idx[1]);
    };
    const int lower = neighbor(-1), upper = neighbor(+1);

    // send first owned slice down / receive upper ghost; send last owned slice
    // up / receive lower ghost.
    auto send_down = t.narrow(d, 1, 1).contiguous();
    auto send_up = t.narrow(d, m, 1).contiguous();
    auto recv_up = torch::empty_like(send_down);
    auto recv_down = torch::empty_like(send_up);
    const int count = static_cast<int>(send_down.numel());
    MPI_Sendrecv(send_down.data_ptr<double>(),
                 count,
                 MPI_DOUBLE,
                 lower,
                 1,
                 recv_up.data_ptr<double>(),
                 count,
                 MPI_DOUBLE,
                 upper,
                 1,
                 L.comm,
                 MPI_STATUS_IGNORE);
    MPI_Sendrecv(send_up.data_ptr<double>(),
                 count,
                 MPI_DOUBLE,
                 upper,
                 2,
                 recv_down.data_ptr<double>(),
                 count,
                 MPI_DOUBLE,
                 lower,
                 2,
                 L.comm,
                 MPI_STATUS_IGNORE);
    t.narrow(d, m + 1, 1).copy_(recv_up);
    t.narrow(d, 0, 1).copy_(recv_down);
  }
}

/// Owned block of a ghosted tensor.
torch::Tensor
owned(const torch::Tensor & g)
{
  return g.narrow(0, 1, g.size(0) - 2).narrow(1, 1, g.size(1) - 2);
}

/// Owned data -> ghosted tensor with fresh ghosts.
torch::Tensor
withGhosts(const torch::Tensor & o, const Layout & L)
{
  auto g = torch::zeros({o.size(0) + 2, o.size(1) + 2}, o.options());
  owned(g).copy_(o);
  exchange(g, L);
  return g;
}

/// Full-weighting along one axis, P^T for cell-centered linear interpolation
/// (without the 1/2 per-axis scaling). Input has one ghost cell on each side
/// along `axis`; output has m / 2 owned cells.
torch::Tensor
restrictAxis(const torch::Tensor & t, int64_t axis)
{
  const int64_t m = t.size(axis) - 2;
  return 0.75 * (t.slice(axis, 1, m + 1, 2) + t.slice(axis, 2, m + 2, 2)) +
         0.25 * (t.slice(axis, 0, m, 2) + t.slice(axis, 3, m + 2, 2));
}

/// Linear interpolation along one axis. Input has one ghost cell on each side
/// along `axis` and mc owned cells; output has 2 mc owned cells.
torch::Tensor
prolongAxis(const torch::Tensor & t, int64_t axis)
{
  const int64_t mc = t.size(axis) - 2;
  auto own = t.narrow(axis, 1, mc);
  auto even = 0.75 * own + 0.25 * t.narrow(axis, 0, mc);
  auto odd = 0.75 * own + 0.25 * t.narrow(axis, 2, mc);
  return torch::stack({even, odd}, axis + 1).flatten(axis, axis + 1);
}

struct Level
{
  int64_t n, mx, my, begin0, begin1;
  double h2;
};

class DistMultigrid
{
public:
  DistMultigrid(const DomainAction & domain, int64_t n)
  {
    _L.parts = {domain.getRealSpacePartitions()[0], domain.getRealSpacePartitions()[1]};
    _L.index = {domain.getRealSpaceIndex()[0], domain.getRealSpaceIndex()[1]};
    _L.comm = domain.getMPIComm();
    int rank;
    MPI_Comm_rank(_L.comm, &rank);
    std::array<int64_t, 3> begin, end;
    domain.getLocalBounds(rank, begin, end);
    _L.begin = {begin[0], begin[1]};
    _domain = &domain;

    int64_t mx = domain.getLocalGridSize()[0], my = domain.getLocalGridSize()[1];
    int64_t b0 = _L.begin[0], b1 = _L.begin[1];
    for (int64_t level_n = n;; level_n /= 2, mx /= 2, my /= 2, b0 /= 2, b1 /= 2)
    {
      _levels.push_back({level_n, mx, my, b0, b1, 1.0 / (double(level_n) * level_n)});
      if (level_n == coarsest_n)
        break;
      if (mx % 2 || my % 2)
        throw std::runtime_error("owned extent must stay even down to the coarsest level");
    }

    const int64_t nc = coarsest_n;
    auto columns = torch::zeros({nc * nc, nc * nc}, torch::kDouble);
    for (int64_t k = 0; k < nc * nc; ++k)
    {
      auto e = torch::zeros({nc, nc}, torch::kDouble);
      e[k / nc][k % nc] = 1.0;
      auto Ae = (4.0 * e - torch::roll(e, 1, 0) - torch::roll(e, -1, 0) - torch::roll(e, 1, 1) -
                 torch::roll(e, -1, 1)) /
                    _levels.back().h2 +
                sigma * e;
      columns.select(1, k).copy_(Ae.reshape({-1}));
    }
    _coarse_inverse = torch::inverse(columns);
  }

  const Layout & layout() const { return _L; }
  const Level & fine() const { return _levels.front(); }

  /// A x for owned x of shape [mx, my].
  torch::Tensor apply(const torch::Tensor & x) const { return applyGhosted(withGhosts(x, _L), 0); }

  /// One symmetric V-cycle from a zero guess.
  torch::Tensor precondition(const torch::Tensor & r) const
  {
    auto u = torch::zeros({fine().mx + 2, fine().my + 2}, r.options());
    vcycle(0, u, r);
    return owned(u).clone();
  }

private:
  torch::Tensor applyGhosted(const torch::Tensor & g, int64_t level) const
  {
    const auto & lv = _levels[level];
    auto c = owned(g);
    auto lo0 = g.narrow(0, 0, lv.mx).narrow(1, 1, lv.my);
    auto up0 = g.narrow(0, 2, lv.mx).narrow(1, 1, lv.my);
    auto lo1 = g.narrow(0, 1, lv.mx).narrow(1, 0, lv.my);
    auto up1 = g.narrow(0, 1, lv.mx).narrow(1, 2, lv.my);
    return (4.0 * c - lo0 - up0 - lo1 - up1) / lv.h2 + sigma * c;
  }

  void smooth(torch::Tensor & u, const torch::Tensor & f, int64_t level, int64_t color) const
  {
    const auto & lv = _levels[level];
    auto c = owned(u);
    auto nb = u.narrow(0, 0, lv.mx).narrow(1, 1, lv.my) +
              u.narrow(0, 2, lv.mx).narrow(1, 1, lv.my) +
              u.narrow(0, 1, lv.mx).narrow(1, 0, lv.my) + u.narrow(0, 1, lv.mx).narrow(1, 2, lv.my);
    auto updated = (f + nb / lv.h2) / (4.0 / lv.h2 + sigma);
    const auto rows = (torch::arange(lv.mx) + lv.begin0).unsqueeze(1);
    const auto cols = (torch::arange(lv.my) + lv.begin1).unsqueeze(0);
    const auto mask = ((rows + cols) % 2) == color;
    c.copy_(torch::where(mask, updated, c));
    exchange(u, _L);
  }

  void vcycle(int64_t level, torch::Tensor & u, const torch::Tensor & f) const
  {
    const auto & lv = _levels[level];

    if (level + 1 == static_cast<int64_t>(_levels.size()))
    {
      // Coarsest level: assemble the global right-hand side with an
      // allreduce, solve densely and redundantly on every rank.
      auto g = torch::zeros({lv.n, lv.n}, f.options());
      g.narrow(0, lv.begin0, lv.mx).narrow(1, lv.begin1, lv.my).copy_(f);
      const auto global_f = _domain->allreduceSum(g);
      const auto sol = _coarse_inverse.matmul(global_f.reshape({-1})).reshape({lv.n, lv.n});
      owned(u).copy_(sol.narrow(0, lv.begin0, lv.mx).narrow(1, lv.begin1, lv.my));
      exchange(u, _L);
      return;
    }

    smooth(u, f, level, 0);
    smooth(u, f, level, 1);

    const auto residual = f - applyGhosted(u, level);
    const auto residual_g = withGhosts(residual, _L);
    const auto coarse_f = restrictAxis(restrictAxis(residual_g, 1), 0) / 4.0;

    const auto & coarse = _levels[level + 1];
    auto coarse_u = torch::zeros({coarse.mx + 2, coarse.my + 2}, f.options());
    vcycle(level + 1, coarse_u, coarse_f);

    owned(u).add_(prolongAxis(prolongAxis(coarse_u, 1), 0));
    exchange(u, _L);

    smooth(u, f, level, 1);
    smooth(u, f, level, 0);
  }

  Layout _L;
  const DomainAction * _domain;
  std::vector<Level> _levels;
  torch::Tensor _coarse_inverse;
};

struct RunResult
{
  unsigned int its;
  double error; // global ||x - x_exact||_2
  double sum;   // global sum of the solution
  double sumsq; // global sum of the squared solution
};

/// Manufactured smooth solution evaluated at cell centers of the owned block.
torch::Tensor
exactSolution(const DistMultigrid & mg, int64_t n)
{
  const auto & lv = mg.fine();
  const auto xs = (torch::arange(lv.mx, torch::kDouble) + lv.begin0 + 0.5) / n;
  const auto ys = (torch::arange(lv.my, torch::kDouble) + lv.begin1 + 0.5) / n;
  const double two_pi = 2.0 * M_PI;
  return torch::sin(two_pi * xs).unsqueeze(1) * torch::cos(2.0 * two_pi * ys).unsqueeze(0) +
         0.5 * torch::cos(3.0 * two_pi * xs).unsqueeze(1) +
         0.25 * torch::sin(two_pi * ys).unsqueeze(0);
}

RunResult
solve(const DomainAction & domain, int64_t n)
{
  DistMultigrid mg(domain, n);
  const auto x_exact = exactSolution(mg, n);
  const auto b = mg.apply(x_exact);

  auto A = [&](const torch::Tensor & x) { return mg.apply(x); };
  auto M = [&](const torch::Tensor & r) { return mg.precondition(r); };
  auto reduce = [&](const torch::Tensor & t) { return domain.allreduceSum(t); };
  const auto [x, its, norm] =
      MooseTensor::conjugateGradientSolve(A, b, torch::Tensor(), rel_tol, 0, M, reduce);
  (void)norm;

  auto global = [&](const torch::Tensor & t)
  { return domain.allreduceSum(t.sum()).item<double>(); };
  return {its, std::sqrt(global((x - x_exact).pow(2))), global(x), global(x * x)};
}
} // namespace

TEST(TorchMultigridMPISpikeTest, SolvesAndIsRankCountIndependent)
{
  int rank, size;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  for (const int64_t n : {64, 128, 256})
  {
    auto md = buildDomain(n);
    const auto result = solve(*md.domain, n);
    if (rank == 0)
      printf("np=%d parts=%ux%u n=%ld MG-PCG: %u its, error %.3e, sum %.12e, sumsq %.12e\n",
             size,
             md.domain->getRealSpacePartitions()[0],
             md.domain->getRealSpacePartitions()[1],
             long(n),
             result.its,
             result.error,
             result.sum,
             result.sumsq);
    EXPECT_LE(result.its, 14u);
    EXPECT_LT(result.error, 1e-8);
  }
}
