/**********************************************************************/
/*                     DO NOT MODIFY THIS HEADER                      */
/*            Marlin, a Fourier spectral solver for MOOSE             */
/*                                                                    */
/*            Copyright 2024 Battelle Energy Alliance, LLC            */
/*                        ALL RIGHTS RESERVED                         */
/**********************************************************************/

// SPIKE (PR2b, not a maintained solver): PETSc KSPCG + PCMG on torch shell level
// operators, on real MPI ranks. The single-rank version is PetscTorchMultigridTest;
// the partition, rank index and communicator come from a REAL_SPACE DomainAction, as
// in TorchMultigridMPISpikeTest.
//
// Problem: periodic 2D  A u = -Laplacian(u) + sigma u, h = 1/n, 5-point stencil,
// cell-centered, CPU. Each rank's PETSc Vec holds its owned block [mx, my] row-major;
// PETSc's global numbering is rank-blocked (rank order), not the natural grid order.
//
// Choices made in this spike (PR2b in the roadmap):
//   halo     - every shell callback copies its owned block into a one-cell ghosted
//              tensor and fills the ghosts either with its own MPI exchange (the
//              TorchMultigridMPISpikeTest pattern; code duplicated, spike only) or with
//              DMGlobalToLocal on a per-level DMDA forced onto Marlin's partition
//              (explicit lx / ly, axes swapped so DMDA's x-fastest order is Marlin's
//              row-major [mx, my], ranks renumbered to match; Vecs placed on the
//              tensors' storage, no copies besides the ghosted tensor itself).
//   coarsest - an assembled MPIAIJ Mat in PETSc's rank-blocked numbering, solved by
//              PCREDUNDANT (gathered and LU-factored on every rank), or by PCTELESCOPE
//              (gathered to one rank; inner preonly + LU, or one GAMG V-cycle).
//              Telescope's default mode extracts rows with MatCreateSubMatrices, so it
//              needs an assembled Mat and cannot wrap the torch shell levels.
//   levels   - halve while n / 2 >= coarsest_n and every rank's owned extent is even;
//              otherwise that level becomes the (larger) redundant coarse level.
// Smoothers: KSPCHEBYSHEV (2 iterations) + PCJACOBI, levels marked MAT_SPD and
// MAT_SPD_ETERNAL, as in PetscTorchMultigridTest.
//
// Run:  mpiexec -n {1,2,4,8} ./marlin-unit-opt --gtest_filter='PetscTorchMultigridMPISpikeTest.*'

#include "ActionFactory.h"
#include "AppFactory.h"
#include "DomainAction.h"
#include "MarlinUtils.h"
#include "MooseApp.h"
#include "gtest/gtest.h"

#include <petscdmda.h>
#include <petscksp.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <map>
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

/// Rank layout needed by the halo exchange and the coarse-matrix numbering.
struct Layout
{
  std::array<unsigned int, 2> parts;
  std::array<unsigned int, 2> index;
  MPI_Comm comm;
  int size;
  // Fine-level owned counts per slab along each axis (DMDA's lx / ly).
  std::array<std::vector<int64_t>, 2> fine_counts;
  int toRank(unsigned int ix, unsigned int iy) const
  {
    return static_cast<int>(ix + parts[0] * iy);
  }
};

/// Fill the one-cell ghost frame of t = [mx + 2, my + 2] from the periodic
/// neighbours, dimension 0 then dimension 1 over the full extent (corners included).
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

torch::Tensor
owned(const torch::Tensor & g)
{
  return g.narrow(0, 1, g.size(0) - 2).narrow(1, 1, g.size(1) - 2);
}

/// Owned block -> ghosted copy; ghosts exchanged unless `skip_halo` (fault injection).
torch::Tensor
withGhosts(const torch::Tensor & o, const Layout & L, bool skip_halo = false)
{
  auto g = torch::zeros({o.size(0) + 2, o.size(1) + 2}, o.options());
  owned(g).copy_(o);
  if (!skip_halo)
    exchange(g, L);
  return g;
}

/// P^T along one axis for cell-centered linear interpolation (without the 1/2
/// per-axis scaling); input ghosted along `axis`, output m / 2 owned cells.
torch::Tensor
restrictAxis(const torch::Tensor & t, int64_t axis)
{
  const int64_t m = t.size(axis) - 2;
  return 0.75 * (t.slice(axis, 1, m + 1, 2) + t.slice(axis, 2, m + 2, 2)) +
         0.25 * (t.slice(axis, 0, m, 2) + t.slice(axis, 3, m + 2, 2));
}

/// Linear interpolation along one axis; input ghosted along `axis` with mc owned
/// cells, output 2 mc owned cells.
torch::Tensor
prolongAxis(const torch::Tensor & t, int64_t axis)
{
  const int64_t mc = t.size(axis) - 2;
  auto own = t.narrow(axis, 1, mc);
  auto even = 0.75 * own + 0.25 * t.narrow(axis, 0, mc);
  auto odd = 0.75 * own + 0.25 * t.narrow(axis, 2, mc);
  return torch::stack({even, odd}, axis + 1).flatten(axis, axis + 1);
}

/// 5-point stencil on the owned cells of a ghosted tensor.
torch::Tensor
applyGhosted(const torch::Tensor & g, double h2)
{
  const int64_t mx = g.size(0) - 2, my = g.size(1) - 2;
  auto c = owned(g);
  auto nb = g.narrow(0, 0, mx).narrow(1, 1, my) + g.narrow(0, 2, mx).narrow(1, 1, my) +
            g.narrow(0, 1, mx).narrow(1, 0, my) + g.narrow(0, 1, mx).narrow(1, 2, my);
  return (4.0 * c - nb) / h2 + sigma * c;
}

/// Per-level DMDA on Marlin's partition plus a global and a local Vec without
/// storage (placed onto tensors per call). DMDA axis 0 is Marlin axis 1 and vice versa,
/// so DMDA's x-fastest order is the row-major order of a torch [mx, my] block.
struct LevelDMDA
{
  DM da = nullptr;
  Vec global = nullptr, local = nullptr;
  ~LevelDMDA()
  {
    VecDestroy(&global);
    VecDestroy(&local);
    DMDestroy(&da);
  }
};

/// `comm` must number Marlin block (ix, iy) as iy + parts[1] * ix (DMDA's rank order
/// for the swapped axes); `shift` is the level number.
std::unique_ptr<LevelDMDA>
makeLevelDMDA(const Layout & L, MPI_Comm comm, int64_t n, int shift)
{
  std::array<std::vector<PetscInt>, 2> counts;
  for (unsigned int d = 0; d < 2; ++d)
    for (const auto f : L.fine_counts[d])
      counts[d].push_back(PetscInt(f >> shift));
  auto dm = std::make_unique<LevelDMDA>();
  PetscCallAbort(comm,
                 DMDACreate2d(comm,
                              DM_BOUNDARY_PERIODIC,
                              DM_BOUNDARY_PERIODIC,
                              DMDA_STENCIL_BOX, // corners: the transfers need them
                              n,
                              n,
                              L.parts[1],
                              L.parts[0],
                              1,
                              1,
                              counts[1].data(),
                              counts[0].data(),
                              &dm->da));
  PetscCallAbort(comm, DMSetUp(dm->da));
  // The DMDA must own exactly this rank's Marlin block (DMDA x = Marlin axis 1).
  PetscInt xs, ys, xm, ym, gxm, gym;
  PetscCallAbort(comm, DMDAGetCorners(dm->da, &xs, &ys, nullptr, &xm, &ym, nullptr));
  PetscCallAbort(comm, DMDAGetGhostCorners(dm->da, nullptr, nullptr, nullptr, &gxm, &gym, nullptr));
  int64_t bx = 0, by = 0;
  for (unsigned int i = 0; i < L.index[0]; ++i)
    bx += counts[0][i];
  for (unsigned int i = 0; i < L.index[1]; ++i)
    by += counts[1][i];
  PetscCheckAbort(xs == by && ys == bx && xm == counts[1][L.index[1]] &&
                      ym == counts[0][L.index[0]] && gxm == xm + 2 && gym == ym + 2,
                  comm,
                  PETSC_ERR_PLIB,
                  "DMDA ownership differs from Marlin's partition");
  // Vecs without storage: the callers place the tensors' arrays per call.
  PetscCallAbort(comm, VecCreateMPIWithArray(comm, 1, xm * ym, n * n, nullptr, &dm->global));
  PetscCallAbort(PETSC_COMM_SELF,
                 VecCreateSeqWithArray(PETSC_COMM_SELF, 1, gxm * gym, nullptr, &dm->local));
  return dm;
}

/// Owned block -> ghosted copy, ghosts filled by DMGlobalToLocal.
torch::Tensor
withGhostsDMDA(const torch::Tensor & o, const LevelDMDA & dm)
{
  const auto oc = o.contiguous();
  auto g = torch::zeros({o.size(0) + 2, o.size(1) + 2}, o.options());
  // DMGlobalToLocal only reads the global Vec.
  PetscCallAbort(PETSC_COMM_SELF, VecPlaceArray(dm.global, oc.data_ptr<double>()));
  PetscCallAbort(PETSC_COMM_SELF, VecPlaceArray(dm.local, g.data_ptr<double>()));
  PetscCallAbort(PETSC_COMM_SELF, DMGlobalToLocalBegin(dm.da, dm.global, INSERT_VALUES, dm.local));
  PetscCallAbort(PETSC_COMM_SELF, DMGlobalToLocalEnd(dm.da, dm.global, INSERT_VALUES, dm.local));
  PetscCallAbort(PETSC_COMM_SELF, VecResetArray(dm.local));
  PetscCallAbort(PETSC_COMM_SELF, VecResetArray(dm.global));
  return g;
}

/// One multigrid level as seen by its shell callbacks: global size n, owned extent
/// [mx, my], h^2, the rank layout, and (DMDA halo only) this level's and the next
/// coarser level's DMDA.
struct LevelContext
{
  int64_t n, mx, my;
  double h2;
  const Layout * L;
  bool skip_halo = false;
  const LevelDMDA * dmda = nullptr;
  const LevelDMDA * coarser_dmda = nullptr;
};

/// Ghosted copy of an owned block of this level (or of the next coarser level).
torch::Tensor
ghost(const torch::Tensor & o, const LevelContext & c, bool coarser = false)
{
  const LevelDMDA * dm = coarser ? c.coarser_dmda : c.dmda;
  if (c.skip_halo || !dm)
    return withGhosts(o, *c.L, c.skip_halo);
  return withGhostsDMDA(o, *dm);
}

/// Views x's local array as [in_x, in_y] and y's as [out_x, out_y], sets y = f(x).
template <typename F>
PetscErrorCode
applyOnBlocks(Vec x, Vec y, std::array<int64_t, 2> in, std::array<int64_t, 2> out, F f)
{
  const PetscScalar * x_array;
  PetscScalar * y_array;
  PetscCall(VecGetArrayRead(x, &x_array));
  PetscCall(VecGetArrayWrite(y, &y_array));
  const auto options = torch::TensorOptions().dtype(torch::kDouble);
  const auto u = torch::from_blob(const_cast<PetscScalar *>(x_array), {in[0], in[1]}, options);
  torch::from_blob(y_array, {out[0], out[1]}, options).copy_(f(u));
  PetscCall(VecRestoreArrayRead(x, &x_array));
  PetscCall(VecRestoreArrayWrite(y, &y_array));
  return PETSC_SUCCESS;
}

PetscErrorCode
levelMult(Mat A, Vec x, Vec y)
{
  LevelContext * c;
  PetscCall(MatShellGetContext(A, &c));
  return applyOnBlocks(x,
                       y,
                       {c->mx, c->my},
                       {c->mx, c->my},
                       [&](const torch::Tensor & u) { return applyGhosted(ghost(u, *c), c->h2); });
}

PetscErrorCode
levelDiagonal(Mat A, Vec diag)
{
  LevelContext * c;
  PetscCall(MatShellGetContext(A, &c));
  PetscCall(VecSet(diag, 4.0 / c->h2 + sigma));
  return PETSC_SUCCESS;
}

/// Transfer shells carry the FINE level's context.
PetscErrorCode
prolongMult(Mat P, Vec coarse, Vec fine)
{
  LevelContext * c;
  PetscCall(MatShellGetContext(P, &c));
  return applyOnBlocks(coarse,
                       fine,
                       {c->mx / 2, c->my / 2},
                       {c->mx, c->my},
                       [&](const torch::Tensor & uc)
                       { return prolongAxis(prolongAxis(ghost(uc, *c, /*coarser=*/true), 1), 0); });
}

PetscErrorCode
restrictMult(Mat R, Vec fine, Vec coarse)
{
  LevelContext * c;
  PetscCall(MatShellGetContext(R, &c));
  return applyOnBlocks(fine,
                       coarse,
                       {c->mx, c->my},
                       {c->mx / 2, c->my / 2},
                       [&](const torch::Tensor & f)
                       { return restrictAxis(restrictAxis(ghost(f, *c), 1), 0) / 4.0; });
}

Mat
makeShell(const Layout & L,
          int64_t local_rows,
          int64_t local_cols,
          int64_t rows,
          int64_t cols,
          void * ctx,
          PetscErrorCode (*mult)(Mat, Vec, Vec))
{
  Mat m;
  PetscCallAbort(L.comm, MatCreateShell(L.comm, local_rows, local_cols, rows, cols, ctx, &m));
  PetscCallAbort(L.comm, MatShellSetOperation(m, MATOP_MULT, (void (*)(void))mult));
  return m;
}

/// Coarsest operator as an assembled MPIAIJ Mat in PETSc's rank-blocked numbering:
/// global row of cell (gx, gy) = offset of its owning rank + local row-major index.
/// `shift` is the level number (owned counts are the fine counts / 2^shift).
Mat
makeCoarseAIJ(const LevelContext & c, int shift)
{
  const Layout & L = *c.L;
  std::array<std::vector<int64_t>, 2> counts, begins;
  for (unsigned int d = 0; d < 2; ++d)
  {
    int64_t b = 0;
    for (const auto f : L.fine_counts[d])
    {
      counts[d].push_back(f >> shift);
      begins[d].push_back(b);
      b += f >> shift;
    }
  }
  std::vector<int64_t> offset(L.size + 1, 0);
  for (int r = 0; r < L.size; ++r)
    offset[r + 1] = offset[r] + counts[0][r % L.parts[0]] * counts[1][r / L.parts[0]];

  auto slab = [&](unsigned int d, int64_t g)
  {
    unsigned int i = 0;
    while (g >= begins[d][i] + counts[d][i])
      ++i;
    return i;
  };
  auto global_row = [&](int64_t gx, int64_t gy)
  {
    gx = (gx + c.n) % c.n;
    gy = (gy + c.n) % c.n;
    const unsigned int ix = slab(0, gx), iy = slab(1, gy);
    return PetscInt(offset[L.toRank(ix, iy)] + (gx - begins[0][ix]) * counts[1][iy] +
                    (gy - begins[1][iy]));
  };

  const int64_t local = c.mx * c.my, N = c.n * c.n;
  Mat A;
  PetscCallAbort(L.comm, MatCreateAIJ(L.comm, local, local, N, N, 5, nullptr, 4, nullptr, &A));
  const int64_t bx = begins[0][L.index[0]], by = begins[1][L.index[1]];
  const PetscScalar off = -1.0 / c.h2, diag = 4.0 / c.h2 + sigma;
  for (int64_t i = 0; i < c.mx; ++i)
    for (int64_t j = 0; j < c.my; ++j)
    {
      const int64_t gx = bx + i, gy = by + j;
      const PetscInt row = global_row(gx, gy);
      const PetscInt cols[5] = {row,
                                global_row(gx - 1, gy),
                                global_row(gx + 1, gy),
                                global_row(gx, gy - 1),
                                global_row(gx, gy + 1)};
      const PetscScalar vals[5] = {diag, off, off, off, off};
      PetscCallAbort(L.comm, MatSetValues(A, 1, &row, 5, cols, vals, ADD_VALUES));
    }
  PetscCallAbort(L.comm, MatAssemblyBegin(A, MAT_FINAL_ASSEMBLY));
  PetscCallAbort(L.comm, MatAssemblyEnd(A, MAT_FINAL_ASSEMBLY));
  return A;
}

Layout
buildLayout(const DomainAction & domain)
{
  Layout L;
  L.parts = {domain.getRealSpacePartitions()[0], domain.getRealSpacePartitions()[1]};
  L.index = {domain.getRealSpaceIndex()[0], domain.getRealSpaceIndex()[1]};
  L.comm = domain.getMPIComm();
  MPI_Comm_size(L.comm, &L.size);

  // Per-slab owned counts, gathered from (index, local size) of every rank.
  const auto & local = domain.getLocalGridSize();
  std::array<int64_t, 4> mine{{int64_t(L.index[0]), int64_t(L.index[1]), local[0], local[1]}};
  std::vector<int64_t> all(4 * L.size);
  MPI_Allgather(mine.data(), 4, MPI_INT64_T, all.data(), 4, MPI_INT64_T, L.comm);
  for (unsigned int d = 0; d < 2; ++d)
  {
    L.fine_counts[d].assign(L.parts[d], 0);
    for (int r = 0; r < L.size; ++r)
      L.fine_counts[d][all[4 * r + d]] = all[4 * r + 2 + d];
  }
  return L;
}

enum class CoarseSolver
{
  Redundant,     // PCREDUNDANT: LU of the full coarse problem on every rank
  TelescopeLU,   // PCTELESCOPE to one rank, inner preonly + LU (exact, as Redundant)
  TelescopeGAMG, // PCTELESCOPE to one rank, inner preonly + PCGAMG (one AMG V-cycle)
};

enum class Halo
{
  Own,  // our MPI_Sendrecv exchange
  Dmda, // DMGlobalToLocal on a per-level DMDA
};

struct RunResult
{
  PetscInt its;
  KSPConvergedReason reason;
  double relative_error;
  int levels;
  int64_t coarse_n;
  double sum, sumsq; // global checksums of the solution
};

RunResult
solvePcmg(const DomainAction & domain,
          int64_t n,
          CoarseSolver coarse_solver = CoarseSolver::Redundant,
          Halo halo = Halo::Own,
          bool drop_fine_halo = false)
{
  const Layout L = buildLayout(domain);

  // Levels, fine first while building, then reversed to PCMG's coarse-to-fine order.
  std::vector<std::unique_ptr<LevelContext>> contexts;
  {
    int64_t level_n = n, mx = domain.getLocalGridSize()[0], my = domain.getLocalGridSize()[1];
    while (true)
    {
      contexts.push_back(std::make_unique<LevelContext>(
          LevelContext{level_n, mx, my, 1.0 / (double(level_n) * level_n), &L}));
      int odd = (mx % 2 || my % 2) ? 1 : 0;
      MPI_Allreduce(MPI_IN_PLACE, &odd, 1, MPI_INT, MPI_MAX, L.comm);
      if (level_n / 2 < coarsest_n || odd)
        break;
      level_n /= 2, mx /= 2, my /= 2;
    }
    std::reverse(contexts.begin(), contexts.end());
  }
  const PetscInt nlev = contexts.size();
  contexts.back()->skip_halo = drop_fine_halo;

  // DMDA halo: one DMDA per shell level (the coarse AIJ level only needs one as the
  // coarser side of the first transfer), on a communicator in DMDA's rank order.
  MPI_Comm dmda_comm = MPI_COMM_NULL;
  std::vector<std::unique_ptr<LevelDMDA>> dmdas;
  if (halo == Halo::Dmda)
  {
    MPI_Comm_split(L.comm, 0, int(L.index[1] + L.parts[1] * L.index[0]), &dmda_comm);
    for (PetscInt l = 0; l < nlev; ++l)
      dmdas.push_back(makeLevelDMDA(L, dmda_comm, contexts[l]->n, int(nlev - 1 - l)));
    for (PetscInt l = 1; l < nlev; ++l)
    {
      contexts[l]->dmda = dmdas[l].get();
      contexts[l]->coarser_dmda = dmdas[l - 1].get();
    }
  }

  std::vector<Mat> A(nlev);
  A[0] = makeCoarseAIJ(*contexts[0], int(nlev - 1));
  for (PetscInt l = 1; l < nlev; ++l)
  {
    const auto & c = *contexts[l];
    const int64_t local = c.mx * c.my, N = c.n * c.n;
    A[l] = makeShell(L, local, local, N, N, contexts[l].get(), levelMult);
    PetscCallAbort(L.comm,
                   MatShellSetOperation(A[l], MATOP_GET_DIAGONAL, (void (*)(void))levelDiagonal));
  }
  // -Laplacian + sigma is SPD on every level; ETERNAL so a re-assembly keeps the flag.
  for (auto & m : A)
  {
    PetscCallAbort(L.comm, MatSetOption(m, MAT_SPD, PETSC_TRUE));
    PetscCallAbort(L.comm, MatSetOption(m, MAT_SPD_ETERNAL, PETSC_TRUE));
  }

  KSP ksp;
  PetscCallAbort(L.comm, KSPCreate(L.comm, &ksp));
  PetscCallAbort(L.comm, KSPSetOperators(ksp, A[nlev - 1], A[nlev - 1]));
  PetscCallAbort(L.comm, KSPSetType(ksp, KSPCG));
  PetscCallAbort(L.comm, KSPSetNormType(ksp, KSP_NORM_UNPRECONDITIONED));
  PetscCallAbort(L.comm, KSPSetTolerances(ksp, rel_tol, 0.0, PETSC_DEFAULT, 1000));

  PC pc;
  PetscCallAbort(L.comm, KSPGetPC(ksp, &pc));
  PetscCallAbort(L.comm, PCSetType(pc, PCMG));
  PetscCallAbort(L.comm, PCMGSetLevels(pc, nlev, nullptr));
  PetscCallAbort(L.comm, PCMGSetType(pc, PC_MG_MULTIPLICATIVE));
  PetscCallAbort(L.comm, PCMGSetCycleType(pc, PC_MG_CYCLE_V));

  std::vector<Mat> transfers;
  for (PetscInt l = 1; l < nlev; ++l)
  {
    const auto & f = *contexts[l];
    const auto & c = *contexts[l - 1];
    const int64_t fl = f.mx * f.my, cl = c.mx * c.my, fN = f.n * f.n, cN = c.n * c.n;
    Mat P = makeShell(L, fl, cl, fN, cN, contexts[l].get(), prolongMult);
    Mat R = makeShell(L, cl, fl, cN, fN, contexts[l].get(), restrictMult);
    PetscCallAbort(L.comm, PCMGSetInterpolation(pc, l, P));
    PetscCallAbort(L.comm, PCMGSetRestriction(pc, l, R));
    transfers.push_back(P);
    transfers.push_back(R);
  }

  for (PetscInt l = 0; l < nlev; ++l)
  {
    KSP smoother;
    PC smoother_pc;
    PetscCallAbort(L.comm, PCMGGetSmoother(pc, l, &smoother));
    PetscCallAbort(L.comm, KSPSetOperators(smoother, A[l], A[l]));
    PetscCallAbort(L.comm, KSPGetPC(smoother, &smoother_pc));
    if (l == 0)
    {
      PetscCallAbort(L.comm, KSPSetType(smoother, KSPPREONLY));
      if (coarse_solver == CoarseSolver::Redundant)
        // Gather the coarse problem to every rank and LU-factor it there.
        PetscCallAbort(L.comm, PCSetType(smoother_pc, PCREDUNDANT));
      else
      {
        // Gather the coarse problem to one rank; the inner KSP is configured after
        // KSPSetUp, which is when telescope creates it (on the active rank only).
        PetscCallAbort(L.comm, PCSetType(smoother_pc, PCTELESCOPE));
        PetscCallAbort(L.comm, PCTelescopeSetReductionFactor(smoother_pc, L.size));
      }
    }
    else
    {
      PetscCallAbort(L.comm, KSPSetType(smoother, KSPCHEBYSHEV));
      PetscCallAbort(L.comm, PCSetType(smoother_pc, PCJACOBI));
      PetscCallAbort(L.comm, KSPSetTolerances(smoother, 0, 0, PETSC_DEFAULT, 2));
      PetscCallAbort(L.comm, KSPSetNormType(smoother, KSP_NORM_NONE));
    }
  }

  // Manufactured smooth solution on this rank's owned block (as the torch MPI spike).
  const auto & fine = *contexts.back();
  int64_t bx = 0, by = 0;
  for (unsigned int i = 0; i < L.index[0]; ++i)
    bx += L.fine_counts[0][i];
  for (unsigned int i = 0; i < L.index[1]; ++i)
    by += L.fine_counts[1][i];
  const auto xs = (torch::arange(fine.mx, torch::kDouble) + bx + 0.5) / n;
  const auto ys = (torch::arange(fine.my, torch::kDouble) + by + 0.5) / n;
  const double two_pi = 2.0 * M_PI;
  const auto x_exact =
      (torch::sin(two_pi * xs).unsqueeze(1) * torch::cos(2.0 * two_pi * ys).unsqueeze(0) +
       0.5 * torch::cos(3.0 * two_pi * xs).unsqueeze(1) +
       0.25 * torch::sin(two_pi * ys).unsqueeze(0))
          .contiguous();
  // b from the exact operator, independent of the (possibly faulted) shell.
  auto b = applyGhosted(withGhosts(x_exact, L), fine.h2).contiguous();
  auto x = torch::zeros_like(b);

  if (coarse_solver != CoarseSolver::Redundant)
  {
    PetscCallAbort(L.comm, KSPSetUp(ksp));
    KSP coarse, inner;
    PC coarse_pc, inner_pc;
    PetscCallAbort(L.comm, PCMGGetCoarseSolve(pc, &coarse));
    PetscCallAbort(L.comm, KSPGetPC(coarse, &coarse_pc));
    PetscCallAbort(L.comm, PCTelescopeGetKSP(coarse_pc, &inner));
    if (inner) // null on the inactive ranks
    {
      PetscCallAbort(PETSC_COMM_SELF, KSPGetPC(inner, &inner_pc));
      PetscCallAbort(
          PETSC_COMM_SELF,
          PCSetType(inner_pc, coarse_solver == CoarseSolver::TelescopeLU ? PCLU : PCGAMG));
    }
  }

  const int64_t local = fine.mx * fine.my, N = n * n;
  Vec b_vec, x_vec;
  PetscCallAbort(L.comm, VecCreateMPIWithArray(L.comm, 1, local, N, b.data_ptr<double>(), &b_vec));
  PetscCallAbort(L.comm, VecCreateMPIWithArray(L.comm, 1, local, N, x.data_ptr<double>(), &x_vec));
  PetscCallAbort(L.comm, KSPSolve(ksp, b_vec, x_vec));

  RunResult result{};
  PetscCallAbort(L.comm, KSPGetIterationNumber(ksp, &result.its));
  PetscCallAbort(L.comm, KSPGetConvergedReason(ksp, &result.reason));
  auto global = [&](const torch::Tensor & t)
  { return domain.allreduceSum(t.sum()).item<double>(); };
  result.relative_error =
      std::sqrt(global((x - x_exact).pow(2))) / std::sqrt(global(x_exact.pow(2)));
  result.levels = int(nlev);
  result.coarse_n = contexts[0]->n;
  result.sum = global(x);
  result.sumsq = global(x * x);

  PetscCallAbort(L.comm, VecDestroy(&b_vec));
  PetscCallAbort(L.comm, VecDestroy(&x_vec));
  PetscCallAbort(L.comm, KSPDestroy(&ksp));
  for (auto & m : A)
    PetscCallAbort(L.comm, MatDestroy(&m));
  for (auto & m : transfers)
    PetscCallAbort(L.comm, MatDestroy(&m));
  dmdas.clear();
  if (dmda_comm != MPI_COMM_NULL)
    MPI_Comm_free(&dmda_comm);
  return result;
}

int
worldRank()
{
  int r;
  MPI_Comm_rank(MPI_COMM_WORLD, &r);
  return r;
}
} // namespace

TEST(PetscTorchMultigridMPISpikeTest, PcmgOnShellsIsRankCountIndependent)
{
  int size;
  MPI_Comm_size(MPI_COMM_WORLD, &size);
  // Single-rank iteration counts (np = 1, 2, 4, 8 all gave these on CPU, 2026-10-18).
  // n = 96 does not halve down to coarsest_n = 8; the level rule stops at n = 12 on every
  // rank count. n = 112 is left out of the exact check: on 2x4 ranks a slab reaches an odd
  // extent (112 / 4 = 28 -> 14 -> 7), so the hierarchy is shallower there (3 levels,
  // coarse n = 28, 7 its) than on 1-4 ranks (4 levels, coarse n = 14, 9 its).
  const std::map<int64_t, PetscInt> single_rank_its = {{64, 9}, {128, 9}, {256, 9}, {96, 9}};
  for (const int64_t n : {64, 128, 256, 96, 112})
  {
    auto md = buildDomain(n);
    const auto r = solvePcmg(*md.domain, n);
    if (worldRank() == 0)
      printf("np=%d parts=%ux%u n=%ld levels=%d coarse n=%ld KSPCG+PCMG: %d its, reason %d, "
             "relative error %.3e, sum %.12e, sumsq %.12e\n",
             size,
             md.domain->getRealSpacePartitions()[0],
             md.domain->getRealSpacePartitions()[1],
             long(n),
             r.levels,
             long(r.coarse_n),
             int(r.its),
             int(r.reason),
             r.relative_error,
             r.sum,
             r.sumsq);
    EXPECT_GT(r.reason, 0);
    EXPECT_LT(r.relative_error, 1e-8);
    if (single_rank_its.count(n))
      EXPECT_EQ(r.its, single_rank_its.at(n));
  }
}

// Fault injection: the finest operator skips its halo exchange (ghosts left zero), so
// KSP solves a different system; the error against the periodic solution must be large.
TEST(PetscTorchMultigridMPISpikeTest, DroppedHaloBreaksTheSolve)
{
  const int64_t n = 64;
  auto md = buildDomain(n);
  const auto r =
      solvePcmg(*md.domain, n, CoarseSolver::Redundant, Halo::Own, /*drop_fine_halo=*/true);
  if (worldRank() == 0)
    printf("dropped halo, n=%ld: %d its, reason %d, relative error %.3e\n",
           long(n),
           int(r.its),
           int(r.reason),
           r.relative_error);
  EXPECT_GT(r.relative_error, 1e-3);
}

// PCTELESCOPE coarse solve. TelescopeLU solves the same coarse problem exactly, on one
// rank instead of all, so it must match PCREDUNDANT's iteration count. TelescopeGAMG
// replaces the exact coarse solve with one AMG V-cycle (a preconditioner choice); its
// count is printed, and only convergence and the error are checked.
TEST(PetscTorchMultigridMPISpikeTest, TelescopeCoarseSolve)
{
  int size;
  MPI_Comm_size(MPI_COMM_WORLD, &size);
  for (const int64_t n : {64, 256, 96, 112})
  {
    auto md = buildDomain(n);
    const auto redundant = solvePcmg(*md.domain, n, CoarseSolver::Redundant);
    const auto lu = solvePcmg(*md.domain, n, CoarseSolver::TelescopeLU);
    const auto gamg = solvePcmg(*md.domain, n, CoarseSolver::TelescopeGAMG);
    if (worldRank() == 0)
      printf("np=%d n=%ld levels=%d coarse n=%ld its: redundant %d, telescope+LU %d, "
             "telescope+GAMG %d (errors %.1e %.1e %.1e)\n",
             size,
             long(n),
             redundant.levels,
             long(redundant.coarse_n),
             int(redundant.its),
             int(lu.its),
             int(gamg.its),
             redundant.relative_error,
             lu.relative_error,
             gamg.relative_error);
    EXPECT_EQ(lu.its, redundant.its);
    for (const auto & r : {lu, gamg})
    {
      EXPECT_GT(r.reason, 0);
      EXPECT_LT(r.relative_error, 1e-8);
    }
  }
}

// DMDA halo: the same solve with every shell's ghosts filled by DMGlobalToLocal on a
// per-level DMDA instead of our exchange. Same operator and transfers, so the iteration
// count must match and the solutions must agree to round-off.
TEST(PetscTorchMultigridMPISpikeTest, DmdaHaloMatchesOwnExchange)
{
  int size;
  MPI_Comm_size(MPI_COMM_WORLD, &size);
  for (const int64_t n : {64, 256, 96, 112})
  {
    auto md = buildDomain(n);
    const auto own = solvePcmg(*md.domain, n, CoarseSolver::Redundant, Halo::Own);
    const auto dmda = solvePcmg(*md.domain, n, CoarseSolver::Redundant, Halo::Dmda);
    if (worldRank() == 0)
      printf("np=%d n=%ld its: own halo %d, DMDA halo %d; relative error %.1e %.1e; "
             "sumsq diff %.1e\n",
             size,
             long(n),
             int(own.its),
             int(dmda.its),
             own.relative_error,
             dmda.relative_error,
             std::abs(own.sumsq - dmda.sumsq));
    EXPECT_GT(dmda.reason, 0);
    EXPECT_EQ(dmda.its, own.its);
    EXPECT_LT(dmda.relative_error, 1e-8);
    EXPECT_NEAR(dmda.sumsq, own.sumsq, 1e-9 * own.sumsq);
  }
}
