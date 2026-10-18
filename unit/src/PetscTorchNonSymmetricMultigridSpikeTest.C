/**********************************************************************/
/*                     DO NOT MODIFY THIS HEADER                      */
/*            Marlin, a Fourier spectral solver for MOOSE             */
/*                                                                    */
/*            Copyright 2024 Battelle Energy Alliance, LLC            */
/*                        ALL RIGHTS RESERVED                         */
/**********************************************************************/

// SPIKE (PR2c, not a maintained solver): a non-symmetric operator under KSPGMRES + PCMG,
// with a torch red-black Gauss-Seidel smoother on the fine level through PCSHELL, and
// the fusion headroom of that smoother inside the PETSc callback.
//
// Problem: periodic 2D  A u = -D Laplacian(u) + v . grad(u) + sigma u, h = 1/n,
// cell-centred, 5-point diffusion, advection first-order upwind or second-order
// central, constant velocity at 30 degrees to the grid. The cell Peclet number
// Pe = |v| h / D is set on the fine grid; it doubles on every coarser level. Single
// rank (the multi-rank hierarchy is PR2b's).
//
// Hierarchy (the design adopted after PR2b): PETSc owns it. Fine DMDA, coarse DMDAs
// from DMCoarsenHierarchy, coarse levels rediscretised with the same scheme and
// assembled with MatSetValuesStencil (scaled by 4^k because PCMG restricts with P^T),
// cell-centred bilinear interpolation assembled as AIJ. The fine level is a torch
// MATSHELL. Coarsest level: LU.
//
// Smoothers compared (they change the preconditioner, not the system being solved):
//   ChebyJacobi - PETSc KSPCHEBYSHEV(2) + PCJACOBI on every level (Chebyshev bounds
//                 estimated by PETSc with GMRES, as it does for a non-SPD Mat).
//   TorchRBGS   - fine level: KSPRICHARDSON(1) + PCSHELL running torch symmetric
//                 red-black Gauss-Seidel (red, black, red half-sweeps); coarse levels:
//                 PETSc PCSOR (symmetric sweep) on the assembled Mats.
//   PetscSOR    - reference: the fine level also assembled as Pmat (Amat stays the
//                 shell), PETSc PCSOR (lexicographic symmetric sweep) on every level.
// Torch smoother modes: eager; traced with the passes TracedComputeSequence applies
// (EliminateDeadCode, ConstantPropagation, EliminateCommonSubexpression, FuseGraph) and
// run through a GraphExecutor; CUDA-graph captured (CUDA only).
//
// DISABLED_CudaSmootherFusionSweep times one PCApply of the fine-level PCSHELL with
// CUDA Vecs (callback on PETSc's stream) for the three modes. Run on a GPU machine with
//   --gtest_also_run_disabled_tests
//   --gtest_filter='*PetscTorchNonSymmetricMultigridSpikeTest.DISABLED_CudaSmootherFusionSweep'
// The CUDA path is compiled only when PETSc and libtorch both have CUDA; it has not
// been compiled or run where this was written.

#include "gtest/gtest.h"

#include <petscdmda.h>
#include <petscksp.h>

#include <torch/cuda.h>
#include <torch/torch.h>
#include <torch/csrc/jit/frontend/tracer.h>
#include <torch/csrc/jit/passes/common_subexpression_elimination.h>
#include <torch/csrc/jit/passes/constant_propagation.h>
#include <torch/csrc/jit/passes/dead_code_elimination.h>
#include <torch/csrc/jit/passes/graph_fuser.h>
#include <torch/csrc/jit/runtime/graph_executor.h>

#include <chrono>
#include <cmath>
#include <memory>
#include <string>
#include <vector>

#if defined(__has_include)
#if __has_include(<ATen/cuda/CUDAConfig.h>)
#define MARLIN_PCMG_TORCH_CUDA 1
#endif
#endif
#if PetscDefined(HAVE_CUDA) && defined(MARLIN_PCMG_TORCH_CUDA)
#define MARLIN_PCMG_CUDA_LEG 1
#include <ATen/cuda/CUDAGraph.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>
#include <petscdevice_cuda.h>
#else
#define MARLIN_PCMG_CUDA_LEG 0
#endif

namespace
{
constexpr double diffusivity = 1.0;
constexpr double sigma = 1.0;
constexpr double velocity_angle = M_PI / 6.0; // not grid-aligned
constexpr int64_t coarsest_n = 8;
constexpr double rel_tol = 1e-10;

enum class Scheme
{
  Upwind,
  Central,
};

enum class SmootherKind
{
  ChebyJacobi,
  TorchRBGS,
  PetscSOR,
};

enum class Mode
{
  Eager,
  Traced,
  CudaGraph,
};

const char *
name(Scheme s)
{
  return s == Scheme::Upwind ? "upwind" : "central";
}

const char *
name(SmootherKind k)
{
  return k == SmootherKind::ChebyJacobi ? "cheby/jacobi"
         : k == SmootherKind::TorchRBGS ? "torch RBGS"
                                        : "petsc SOR";
}

/// 5-point coefficients of  -D Laplacian + v . grad + sigma  at spacing h:
/// (A u)_ij = c u_ij + w u_(i-1)j + e u_(i+1)j + s u_i(j-1) + n u_i(j+1), i along x.
struct Stencil
{
  double c, w, e, s, n;
};

Stencil
makeStencil(Scheme scheme, double vx, double vy, double h)
{
  const double d = diffusivity / (h * h);
  Stencil st{4.0 * d + sigma, -d, -d, -d, -d};
  auto advect = [&](double v, double & lower, double & upper)
  {
    if (scheme == Scheme::Central)
    {
      lower -= v / (2.0 * h);
      upper += v / (2.0 * h);
    }
    else if (v > 0) // upwind: backward difference
    {
      st.c += v / h;
      lower -= v / h;
    }
    else // forward difference
    {
      st.c -= v / h;
      upper += v / h;
    }
  };
  advect(vx, st.w, st.e);
  advect(vy, st.s, st.n);
  return st;
}

/// A u on a periodic [ny, nx] tensor (axis 1 = x = DMDA's fastest index).
torch::Tensor
applyStencil(const torch::Tensor & u, const Stencil & st)
{
  return st.c * u + st.w * torch::roll(u, 1, 1) + st.e * torch::roll(u, -1, 1) +
         st.s * torch::roll(u, 1, 0) + st.n * torch::roll(u, -1, 0);
}

/// Symmetric red-black Gauss-Seidel for A z = r from z = 0: red, black, red half-sweeps
/// (a second black half-sweep would repeat the first). Valid for a 5-point stencil,
/// where every neighbour of a point has the other colour.
torch::Tensor
redBlackGaussSeidel(const torch::Tensor & r, const Stencil & st, const torch::Tensor & red)
{
  auto z = torch::zeros_like(r);
  const auto black = red.logical_not();
  for (const auto * mask : {&red, &black, &red})
    z = torch::where(*mask, z + (r - applyStencil(z, st)) / st.c, z);
  return z;
}

#if MARLIN_PCMG_CUDA_LEG
/// Makes torch enqueue on PETSc's current CUDA stream while a callback runs.
class PetscStreamGuard
{
public:
  PetscStreamGuard()
  {
    cudaStream_t petsc_stream;
    PetscCallAbort(PETSC_COMM_SELF, PetscGetCurrentCUDAStream(&petsc_stream));
    _guard = std::make_unique<c10::cuda::CUDAStreamGuard>(
        c10::cuda::getStreamFromExternal(petsc_stream, c10::cuda::current_device()));
  }

private:
  std::unique_ptr<c10::cuda::CUDAStreamGuard> _guard;
};
#endif

/// The torch smoother in one of the three execution modes.
class TorchSmoother
{
public:
  TorchSmoother(const Stencil & st, int64_t n, const torch::TensorOptions & options, Mode mode)
    : _st(st), _mode(mode)
  {
    const auto i = torch::arange(n, options.dtype(torch::kLong));
    _red = ((i.unsqueeze(0) + i.unsqueeze(1)) % 2) == 0;

    if (mode == Mode::Traced)
    {
      const auto example = torch::randn({n, n}, options);
      auto forward = [this](torch::jit::Stack inputs) -> torch::jit::Stack
      { return {redBlackGaussSeidel(inputs[0].toTensor(), _st, _red)}; };
      auto traced = torch::jit::tracer::trace(
          {example},
          forward,
          [](const torch::Tensor &) { return std::string(); },
          /*strict=*/false,
          /*force_outplace=*/false);
      auto graph = std::get<0>(traced)->graph;
      torch::jit::EliminateDeadCode(graph);
      torch::jit::ConstantPropagation(graph);
      torch::jit::EliminateCommonSubexpression(graph);
      torch::jit::FuseGraph(graph, /*strict=*/true);
      _traced = std::make_unique<torch::jit::GraphExecutor>(graph, "rbgs");
    }
    else if (mode == Mode::CudaGraph)
    {
#if MARLIN_PCMG_CUDA_LEG
      // Capture needs a non-default stream; replay runs on whatever stream is current.
      _in = torch::zeros({n, n}, options);
      auto side = c10::cuda::getStreamFromPool();
      {
        c10::cuda::CUDAStreamGuard guard(side);
        for (int k = 0; k < 3; ++k)
          redBlackGaussSeidel(_in, _st, _red);
        side.synchronize();
        _graph = std::make_unique<at::cuda::CUDAGraph>();
        _graph->capture_begin();
        _out = redBlackGaussSeidel(_in, _st, _red);
        _graph->capture_end();
      }
      torch::cuda::synchronize();
#else
      throw std::runtime_error("CUDA-graph mode needs the CUDA build");
#endif
    }
  }

  torch::Tensor apply(const torch::Tensor & r)
  {
    switch (_mode)
    {
      case Mode::Eager:
        return redBlackGaussSeidel(r, _st, _red);
      case Mode::Traced:
      {
        torch::jit::Stack stack{r};
        _traced->run(stack);
        return stack.back().toTensor();
      }
      case Mode::CudaGraph:
#if MARLIN_PCMG_CUDA_LEG
        _in.copy_(r);
        _graph->replay();
        return _out;
#else
        break;
#endif
    }
    throw std::runtime_error("unreachable smoother mode");
  }

private:
  Stencil _st;
  Mode _mode;
  torch::Tensor _red;
  std::unique_ptr<torch::jit::GraphExecutor> _traced;
#if MARLIN_PCMG_CUDA_LEG
  std::unique_ptr<at::cuda::CUDAGraph> _graph;
  torch::Tensor _in, _out;
#endif
};

/// Views x and y as [n, n] grids on the device their memory lives on and sets y = f(x).
template <typename F>
PetscErrorCode
applyOnGrid(Vec x, Vec y, int64_t n, F f)
{
  const PetscScalar * x_array;
  PetscScalar * y_array;
  PetscMemType memtype, y_memtype;
  PetscCall(VecGetArrayReadAndMemType(x, &x_array, &memtype));
  PetscCall(VecGetArrayWriteAndMemType(y, &y_array, &y_memtype));
  PetscCheck(PetscMemTypeDevice(memtype) == PetscMemTypeDevice(y_memtype),
             PETSC_COMM_SELF,
             PETSC_ERR_ARG_INCOMP,
             "x and y must live on the same device");
  auto options = torch::TensorOptions().dtype(torch::kDouble);
#if MARLIN_PCMG_CUDA_LEG
  std::unique_ptr<PetscStreamGuard> guard;
  if (PetscMemTypeDevice(memtype))
  {
    options = options.device(torch::kCUDA, c10::cuda::current_device());
    guard = std::make_unique<PetscStreamGuard>();
  }
#endif
  const auto in = torch::from_blob(const_cast<PetscScalar *>(x_array), {n, n}, options);
  torch::from_blob(y_array, {n, n}, options).copy_(f(in));
  PetscCall(VecRestoreArrayReadAndMemType(x, &x_array));
  PetscCall(VecRestoreArrayWriteAndMemType(y, &y_array));
  return PETSC_SUCCESS;
}

struct FineContext
{
  int64_t n;
  Stencil st;
  TorchSmoother * smoother = nullptr;
};

PetscErrorCode
fineMult(Mat A, Vec x, Vec y)
{
  FineContext * c;
  PetscCall(MatShellGetContext(A, &c));
  return applyOnGrid(x, y, c->n, [&](const torch::Tensor & u) { return applyStencil(u, c->st); });
}

PetscErrorCode
fineDiagonal(Mat A, Vec diag)
{
  FineContext * c;
  PetscCall(MatShellGetContext(A, &c));
  PetscCall(VecSet(diag, c->st.c));
  return PETSC_SUCCESS;
}

PetscErrorCode
smootherApply(PC pc, Vec r, Vec z)
{
  FineContext * c;
  PetscCall(PCShellGetContext(pc, &c));
  return applyOnGrid(
      r, z, c->n, [&](const torch::Tensor & residual) { return c->smoother->apply(residual); });
}

/// Rediscretised operator on a DMDA, scaled by `scale`, assembled by grid index.
void
assembleLevel(DM da, Mat B, const Stencil & st, double scale)
{
  PetscInt xs, ys, xm, ym;
  PetscCallAbort(PETSC_COMM_SELF, DMDAGetCorners(da, &xs, &ys, nullptr, &xm, &ym, nullptr));
  const PetscScalar vals[5] = {
      scale * st.c, scale * st.w, scale * st.e, scale * st.s, scale * st.n};
  const PetscInt di[5] = {0, -1, 1, 0, 0}, dj[5] = {0, 0, 0, -1, 1};
  for (PetscInt j = ys; j < ys + ym; ++j)
    for (PetscInt i = xs; i < xs + xm; ++i)
    {
      MatStencil row{}, cols[5]{};
      row.i = i, row.j = j;
      for (int q = 0; q < 5; ++q)
        cols[q].i = i + di[q], cols[q].j = j + dj[q];
      PetscCallAbort(PETSC_COMM_SELF,
                     MatSetValuesStencil(B, 1, &row, 5, cols, vals, INSERT_VALUES));
    }
  PetscCallAbort(PETSC_COMM_SELF, MatAssemblyBegin(B, MAT_FINAL_ASSEMBLY));
  PetscCallAbort(PETSC_COMM_SELF, MatAssemblyEnd(B, MAT_FINAL_ASSEMBLY));
}

/// Cell-centred bilinear interpolation [nc, nc] -> [2 nc, 2 nc], periodic, single rank
/// (DMDA global numbering = natural, x fastest): 9/16 own coarse cell, 3/16 each
/// neighbour on the fine cell's side, 1/16 the diagonal one.
Mat
cellCentredInterpolation(int64_t nc)
{
  const int64_t nf = 2 * nc;
  Mat P;
  PetscCallAbort(PETSC_COMM_SELF,
                 MatCreateSeqAIJ(PETSC_COMM_SELF, nf * nf, nc * nc, 4, nullptr, &P));
  auto col = [&](int64_t ic, int64_t jc)
  { return PetscInt(((jc + nc) % nc) * nc + (ic + nc) % nc); };
  const PetscScalar vals[4] = {0.5625, 0.1875, 0.1875, 0.0625};
  for (int64_t j = 0; j < nf; ++j)
    for (int64_t i = 0; i < nf; ++i)
    {
      const PetscInt row = j * nf + i;
      const int64_t ic = i / 2, jc = j / 2;
      const int64_t in = i % 2 ? ic + 1 : ic - 1, jn = j % 2 ? jc + 1 : jc - 1;
      const PetscInt cols[4] = {col(ic, jc), col(in, jc), col(ic, jn), col(in, jn)};
      PetscCallAbort(PETSC_COMM_SELF, MatSetValues(P, 1, &row, 4, cols, vals, INSERT_VALUES));
    }
  PetscCallAbort(PETSC_COMM_SELF, MatAssemblyBegin(P, MAT_FINAL_ASSEMBLY));
  PetscCallAbort(PETSC_COMM_SELF, MatAssemblyEnd(P, MAT_FINAL_ASSEMBLY));
  return P;
}

struct Case
{
  int64_t n;
  Scheme scheme;
  double pe; // cell Peclet number |v| h / D on the fine grid
  SmootherKind kind;
  Mode mode = Mode::Eager;
};

struct Result
{
  PetscInt its;
  KSPConvergedReason reason;
  double relative_error;
  double solve_ms;
  int levels;
  torch::Tensor x;
};

void
configureSmoother(KSP smoother, bool fine, SmootherKind kind, FineContext * fine_ctx)
{
  PC pc;
  PetscCallAbort(PETSC_COMM_SELF, KSPGetPC(smoother, &pc));
  PetscCallAbort(PETSC_COMM_SELF, KSPSetNormType(smoother, KSP_NORM_NONE));
  if (kind == SmootherKind::ChebyJacobi)
  {
    PetscCallAbort(PETSC_COMM_SELF, KSPSetType(smoother, KSPCHEBYSHEV));
    PetscCallAbort(PETSC_COMM_SELF, PCSetType(pc, PCJACOBI));
    PetscCallAbort(PETSC_COMM_SELF, KSPSetTolerances(smoother, 0, 0, PETSC_DEFAULT, 2));
    return;
  }
  PetscCallAbort(PETSC_COMM_SELF, KSPSetType(smoother, KSPRICHARDSON));
  PetscCallAbort(PETSC_COMM_SELF, KSPSetTolerances(smoother, 0, 0, PETSC_DEFAULT, 1));
  if (fine && kind == SmootherKind::TorchRBGS)
  {
    PetscCallAbort(PETSC_COMM_SELF, PCSetType(pc, PCSHELL));
    PetscCallAbort(PETSC_COMM_SELF, PCShellSetContext(pc, fine_ctx));
    PetscCallAbort(PETSC_COMM_SELF, PCShellSetApply(pc, smootherApply));
    PetscCallAbort(PETSC_COMM_SELF, PCShellSetName(pc, "torch symmetric red-black GS"));
  }
  else
  {
    PetscCallAbort(PETSC_COMM_SELF, PCSetType(pc, PCSOR));
    PetscCallAbort(PETSC_COMM_SELF, PCSORSetSymmetric(pc, SOR_SYMMETRIC_SWEEP));
  }
}

Result
solve(const Case & cs)
{
  const int64_t n = cs.n;
  const double h = 1.0 / double(n);
  const double speed = cs.pe * diffusivity / h;
  const double vx = speed * std::cos(velocity_angle), vy = speed * std::sin(velocity_angle);
  FineContext fine{n, makeStencil(cs.scheme, vx, vy, h)};
  const auto options = torch::TensorOptions().dtype(torch::kDouble);
  std::unique_ptr<TorchSmoother> smoother;
  if (cs.kind == SmootherKind::TorchRBGS)
  {
    smoother = std::make_unique<TorchSmoother>(fine.st, n, options, cs.mode);
    fine.smoother = smoother.get();
  }

  DM fine_da;
  PetscCallAbort(PETSC_COMM_SELF,
                 DMDACreate2d(PETSC_COMM_SELF,
                              DM_BOUNDARY_PERIODIC,
                              DM_BOUNDARY_PERIODIC,
                              DMDA_STENCIL_STAR,
                              n,
                              n,
                              1,
                              1,
                              1,
                              1,
                              nullptr,
                              nullptr,
                              &fine_da));
  PetscCallAbort(PETSC_COMM_SELF, DMSetUp(fine_da));
  PetscInt nlev = 1;
  for (int64_t m = n; m % 2 == 0 && m / 2 >= coarsest_n; m /= 2)
    ++nlev;
  std::vector<DM> coarse_dms(nlev - 1); // [k] is k + 1 levels below the fine one
  PetscCallAbort(PETSC_COMM_SELF, DMCoarsenHierarchy(fine_da, nlev - 1, coarse_dms.data()));

  Mat A;
  PetscCallAbort(PETSC_COMM_SELF,
                 MatCreateShell(PETSC_COMM_SELF, n * n, n * n, n * n, n * n, &fine, &A));
  PetscCallAbort(PETSC_COMM_SELF, MatShellSetOperation(A, MATOP_MULT, (void (*)(void))fineMult));
  PetscCallAbort(PETSC_COMM_SELF,
                 MatShellSetOperation(A, MATOP_GET_DIAGONAL, (void (*)(void))fineDiagonal));
  Mat fine_pmat = A;
  if (cs.kind == SmootherKind::PetscSOR)
  {
    PetscCallAbort(PETSC_COMM_SELF, DMCreateMatrix(fine_da, &fine_pmat));
    assembleLevel(fine_da, fine_pmat, fine.st, 1.0);
  }

  KSP ksp;
  PC pc;
  PetscCallAbort(PETSC_COMM_SELF, KSPCreate(PETSC_COMM_SELF, &ksp));
  PetscCallAbort(PETSC_COMM_SELF, KSPSetOperators(ksp, A, fine_pmat));
  PetscCallAbort(PETSC_COMM_SELF, KSPSetType(ksp, KSPGMRES));
  PetscCallAbort(PETSC_COMM_SELF, KSPSetPCSide(ksp, PC_RIGHT));
  PetscCallAbort(PETSC_COMM_SELF, KSPSetNormType(ksp, KSP_NORM_UNPRECONDITIONED));
  PetscCallAbort(PETSC_COMM_SELF, KSPSetTolerances(ksp, rel_tol, 0.0, PETSC_DEFAULT, 500));
  PetscCallAbort(PETSC_COMM_SELF, KSPGetPC(ksp, &pc));
  PetscCallAbort(PETSC_COMM_SELF, PCSetType(pc, PCMG));
  PetscCallAbort(PETSC_COMM_SELF, PCMGSetLevels(pc, nlev, nullptr));
  PetscCallAbort(PETSC_COMM_SELF, PCMGSetType(pc, PC_MG_MULTIPLICATIVE));
  PetscCallAbort(PETSC_COMM_SELF, PCMGSetCycleType(pc, PC_MG_CYCLE_V));

  std::vector<Mat> owned;
  for (PetscInt k = 0; k < nlev - 1; ++k)
  {
    const int64_t m = n >> (k + 1);
    const double scale = double(n / m) * double(n / m);
    Mat B;
    PetscCallAbort(PETSC_COMM_SELF, DMCreateMatrix(coarse_dms[k], &B));
    assembleLevel(coarse_dms[k], B, makeStencil(cs.scheme, vx, vy, 1.0 / double(m)), scale);
    KSP level;
    PetscCallAbort(PETSC_COMM_SELF, PCMGGetSmoother(pc, nlev - 2 - k, &level));
    PetscCallAbort(PETSC_COMM_SELF, KSPSetOperators(level, B, B));
    Mat P = cellCentredInterpolation(m);
    PetscCallAbort(PETSC_COMM_SELF, PCMGSetInterpolation(pc, nlev - 1 - k, P));
    owned.push_back(B);
    owned.push_back(P);
  }
  for (PetscInt l = 0; l < nlev; ++l)
  {
    KSP level;
    PetscCallAbort(PETSC_COMM_SELF, PCMGGetSmoother(pc, l, &level));
    if (l == nlev - 1)
      PetscCallAbort(PETSC_COMM_SELF, KSPSetOperators(level, A, fine_pmat));
    if (l == 0)
    {
      PC coarse_pc;
      PetscCallAbort(PETSC_COMM_SELF, KSPSetType(level, KSPPREONLY));
      PetscCallAbort(PETSC_COMM_SELF, KSPGetPC(level, &coarse_pc));
      PetscCallAbort(PETSC_COMM_SELF, PCSetType(coarse_pc, PCLU));
    }
    else
      configureSmoother(level, l == nlev - 1, cs.kind, &fine);
  }

  // Manufactured smooth periodic solution, b = A x_exact for the discrete operator.
  const auto xs = (torch::arange(n, options) + 0.5) / double(n);
  const double two_pi = 2.0 * M_PI;
  const auto x_exact =
      (torch::sin(two_pi * xs).unsqueeze(0) * torch::cos(2.0 * two_pi * xs).unsqueeze(1) +
       0.5 * torch::cos(3.0 * two_pi * xs).unsqueeze(0) +
       0.25 * torch::sin(two_pi * xs).unsqueeze(1))
          .contiguous();
  auto b = applyStencil(x_exact, fine.st).contiguous();
  auto x = torch::zeros_like(b);
  Vec b_vec, x_vec;
  PetscCallAbort(PETSC_COMM_SELF,
                 VecCreateSeqWithArray(PETSC_COMM_SELF, 1, n * n, b.data_ptr<double>(), &b_vec));
  PetscCallAbort(PETSC_COMM_SELF,
                 VecCreateSeqWithArray(PETSC_COMM_SELF, 1, n * n, x.data_ptr<double>(), &x_vec));
  PetscCallAbort(PETSC_COMM_SELF, KSPSetUp(ksp));
  const auto t0 = std::chrono::steady_clock::now();
  PetscCallAbort(PETSC_COMM_SELF, KSPSolve(ksp, b_vec, x_vec));
  const double solve_ms =
      std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();

  Result result{};
  PetscCallAbort(PETSC_COMM_SELF, KSPGetIterationNumber(ksp, &result.its));
  PetscCallAbort(PETSC_COMM_SELF, KSPGetConvergedReason(ksp, &result.reason));
  result.relative_error = ((x - x_exact).norm() / x_exact.norm()).item<double>();
  result.solve_ms = solve_ms;
  result.levels = int(nlev);
  result.x = x.clone();

  PetscCallAbort(PETSC_COMM_SELF, VecDestroy(&b_vec));
  PetscCallAbort(PETSC_COMM_SELF, VecDestroy(&x_vec));
  PetscCallAbort(PETSC_COMM_SELF, KSPDestroy(&ksp));
  if (fine_pmat != A)
    PetscCallAbort(PETSC_COMM_SELF, MatDestroy(&fine_pmat));
  PetscCallAbort(PETSC_COMM_SELF, MatDestroy(&A));
  for (auto & m : owned)
    PetscCallAbort(PETSC_COMM_SELF, MatDestroy(&m));
  for (auto & d : coarse_dms)
    PetscCallAbort(PETSC_COMM_SELF, DMDestroy(&d));
  PetscCallAbort(PETSC_COMM_SELF, DMDestroy(&fine_da));
  return result;
}

/// Wall time of fn() in ms, device-synchronized on both sides when on CUDA.
template <typename F>
double
timedMs(bool cuda, F fn)
{
  if (cuda)
    torch::cuda::synchronize();
  const auto t0 = std::chrono::steady_clock::now();
  fn();
  if (cuda)
    torch::cuda::synchronize();
  return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
}

struct SmootherTiming
{
  double ms;           // mean time of one PCApply
  double max_rel_diff; // against the eager result
};

/// One PCApply of the fine-level PCSHELL smoother (upwind, Pe = 1) with Vecs on the
/// CPU or on CUDA, `reps` times after a warm-up; compares the result with eager.
SmootherTiming
timeSmootherInCallback(int64_t n, bool cuda, Mode mode, int reps, const torch::Tensor & eager_z)
{
  const double h = 1.0 / double(n), speed = diffusivity / h;
  FineContext ctx{
      n,
      makeStencil(
          Scheme::Upwind, speed * std::cos(velocity_angle), speed * std::sin(velocity_angle), h)};
  auto options = torch::TensorOptions().dtype(torch::kDouble);
  if (cuda)
    options = options.device(torch::kCUDA, 0);
  TorchSmoother smoother(ctx.st, n, options, mode);
  ctx.smoother = &smoother;

  // Draw the input on the CPU generator (the CUDA one gives a different sequence), as the
  // reference does, then copy it to the device.
  torch::manual_seed(0);
  auto r = torch::randn({n, n}, torch::kDouble).to(options.device());
  auto z = torch::zeros_like(r);
  Vec r_vec, z_vec;
  if (cuda)
  {
#if MARLIN_PCMG_CUDA_LEG
    PetscCallAbort(
        PETSC_COMM_SELF,
        VecCreateSeqCUDAWithArray(PETSC_COMM_SELF, 1, n * n, r.data_ptr<double>(), &r_vec));
    PetscCallAbort(
        PETSC_COMM_SELF,
        VecCreateSeqCUDAWithArray(PETSC_COMM_SELF, 1, n * n, z.data_ptr<double>(), &z_vec));
#else
    throw std::runtime_error("CUDA Vecs need the CUDA build");
#endif
  }
  else
  {
    PetscCallAbort(PETSC_COMM_SELF,
                   VecCreateSeqWithArray(PETSC_COMM_SELF, 1, n * n, r.data_ptr<double>(), &r_vec));
    PetscCallAbort(PETSC_COMM_SELF,
                   VecCreateSeqWithArray(PETSC_COMM_SELF, 1, n * n, z.data_ptr<double>(), &z_vec));
  }
  // PCSetUp needs an operator; the fine shell, as in the solve.
  Mat A;
  PetscCallAbort(PETSC_COMM_SELF,
                 MatCreateShell(PETSC_COMM_SELF, n * n, n * n, n * n, n * n, &ctx, &A));
  PetscCallAbort(PETSC_COMM_SELF, MatShellSetOperation(A, MATOP_MULT, (void (*)(void))fineMult));
#if MARLIN_PCMG_CUDA_LEG
  if (cuda)
    PetscCallAbort(PETSC_COMM_SELF, MatShellSetVecType(A, VECSEQCUDA));
#endif
  PC pc;
  PetscCallAbort(PETSC_COMM_SELF, PCCreate(PETSC_COMM_SELF, &pc));
  PetscCallAbort(PETSC_COMM_SELF, PCSetOperators(pc, A, A));
  PetscCallAbort(PETSC_COMM_SELF, PCSetType(pc, PCSHELL));
  PetscCallAbort(PETSC_COMM_SELF, PCShellSetContext(pc, &ctx));
  PetscCallAbort(PETSC_COMM_SELF, PCShellSetApply(pc, smootherApply));

  for (int k = 0; k < 3; ++k) // warm-up (the profiling executor specialises here)
    PetscCallAbort(PETSC_COMM_SELF, PCApply(pc, r_vec, z_vec));
  const double total = timedMs(cuda,
                               [&]
                               {
                                 for (int k = 0; k < reps; ++k)
                                   PetscCallAbort(PETSC_COMM_SELF, PCApply(pc, r_vec, z_vec));
                               });
  // Read z through PETSc so a device result is synchronised back into the tensor.
  const PetscScalar * z_array;
  PetscMemType memtype;
  PetscCallAbort(PETSC_COMM_SELF, VecGetArrayReadAndMemType(z_vec, &z_array, &memtype));
  PetscCallAbort(PETSC_COMM_SELF, VecRestoreArrayReadAndMemType(z_vec, &z_array));
  const auto z_cpu = z.cpu();
  const double diff = eager_z.defined()
                          ? ((z_cpu - eager_z).abs().max() / eager_z.abs().max()).item<double>()
                          : 0.0;

  PetscCallAbort(PETSC_COMM_SELF, PCDestroy(&pc));
  PetscCallAbort(PETSC_COMM_SELF, MatDestroy(&A));
  PetscCallAbort(PETSC_COMM_SELF, VecDestroy(&r_vec));
  PetscCallAbort(PETSC_COMM_SELF, VecDestroy(&z_vec));
  return {total / reps, diff};
}

/// Eager smoother result for the timing input (CPU), the reference for the other modes.
torch::Tensor
eagerSmootherResult(int64_t n)
{
  const double h = 1.0 / double(n), speed = diffusivity / h;
  const auto st = makeStencil(
      Scheme::Upwind, speed * std::cos(velocity_angle), speed * std::sin(velocity_angle), h);
  const auto options = torch::TensorOptions().dtype(torch::kDouble);
  TorchSmoother smoother(st, n, options, Mode::Eager);
  torch::manual_seed(0);
  return smoother.apply(torch::randn({n, n}, torch::kDouble));
}

int
worldRank()
{
  int r;
  MPI_Comm_rank(MPI_COMM_WORLD, &r);
  return r;
}
} // namespace

// Pe sweep. Reported, not assumed: only convergence where the theory gives it is
// asserted (upwind gives an M-matrix on every level, for which Gauss-Seidel smoothing
// converges), plus the error of every converged solve against the manufactured solution.
TEST(PetscTorchNonSymmetricMultigridSpikeTest, PecletSweep)
{
  if (worldRank() != 0)
    GTEST_SKIP() << "single-rank spike";
  for (const int64_t n : {64, 128, 256})
    for (const auto scheme : {Scheme::Upwind, Scheme::Central})
      for (const double pe : {0.1, 1.0, 10.0})
        for (const auto kind :
             {SmootherKind::ChebyJacobi, SmootherKind::TorchRBGS, SmootherKind::PetscSOR})
        {
          const auto r = solve({n, scheme, pe, kind});
          printf("n=%4ld levels=%d %-7s Pe=%5.1f %-12s: %3d its, reason %3d, "
                 "relative error %.2e, solve %8.2f ms\n",
                 long(n),
                 r.levels,
                 name(scheme),
                 pe,
                 name(kind),
                 int(r.its),
                 int(r.reason),
                 r.relative_error,
                 r.solve_ms);
          if (r.reason > 0)
            EXPECT_LT(r.relative_error, 1e-8);
          if (scheme == Scheme::Upwind && kind != SmootherKind::ChebyJacobi)
            EXPECT_GT(r.reason, 0) << name(kind) << " upwind Pe=" << pe << " n=" << n;
        }
}

// The traced smoother is the same linear operator as the eager one, so the solve must
// take the same iterations and reach the same solution; the PCApply times on the CPU are
// printed (FuseGraph's legacy fuser does not fuse on the CPU, so no speed-up is expected
// here; the measurement that matters is the CUDA sweep below).
TEST(PetscTorchNonSymmetricMultigridSpikeTest, TracedSmootherMatchesEager)
{
  if (worldRank() != 0)
    GTEST_SKIP() << "single-rank spike";
  for (const double pe : {0.1, 1.0, 10.0})
  {
    const auto eager = solve({128, Scheme::Upwind, pe, SmootherKind::TorchRBGS, Mode::Eager});
    const auto traced = solve({128, Scheme::Upwind, pe, SmootherKind::TorchRBGS, Mode::Traced});
    const double diff = ((eager.x - traced.x).norm() / eager.x.norm()).item<double>();
    printf("n=128 upwind Pe=%4.1f torch RBGS: eager %d its, traced %d its, solution diff %.1e\n",
           pe,
           int(eager.its),
           int(traced.its),
           diff);
    EXPECT_EQ(traced.its, eager.its);
    EXPECT_LT(diff, 1e-12);
  }
  for (const int64_t n : {256, 1024})
  {
    const auto reference = eagerSmootherResult(n);
    const auto eager = timeSmootherInCallback(n, false, Mode::Eager, 20, reference);
    const auto traced = timeSmootherInCallback(n, false, Mode::Traced, 20, reference);
    printf("CPU PCApply n=%ld: eager %.3f ms, traced %.3f ms (max rel diff %.1e)\n",
           long(n),
           eager.ms,
           traced.ms,
           traced.max_rel_diff);
    EXPECT_LT(eager.max_rel_diff, 1e-14);
    EXPECT_LT(traced.max_rel_diff, 1e-12);
  }
}

TEST(PetscTorchNonSymmetricMultigridSpikeTest, DISABLED_CudaSmootherFusionSweep)
{
#if MARLIN_PCMG_CUDA_LEG
  ASSERT_TRUE(torch::cuda::is_available()) << "needs a CUDA-enabled libtorch and a GPU";
  for (const int64_t n : {256, 512, 1024, 2048, 4096, 8192})
  {
    const auto reference = eagerSmootherResult(n);
    const auto eager = timeSmootherInCallback(n, true, Mode::Eager, 50, reference);
    const auto traced = timeSmootherInCallback(n, true, Mode::Traced, 50, reference);
    const auto graph = timeSmootherInCallback(n, true, Mode::CudaGraph, 50, reference);
    printf("CUDA PCApply n=%ld: eager %.3f ms, traced %.3f ms, CUDA graph %.3f ms "
           "(max rel diff vs CPU eager: %.1e %.1e %.1e)\n",
           long(n),
           eager.ms,
           traced.ms,
           graph.ms,
           eager.max_rel_diff,
           traced.max_rel_diff,
           graph.max_rel_diff);
    EXPECT_LT(eager.max_rel_diff, 1e-12);
    EXPECT_LT(traced.max_rel_diff, 1e-12);
    EXPECT_LT(graph.max_rel_diff, 1e-12);
  }
#else
  GTEST_SKIP() << "built without CUDA in PETSc and libtorch";
#endif
}
