/**********************************************************************/
/*                     DO NOT MODIFY THIS HEADER                      */
/*            Marlin, a Fourier spectral solver for MOOSE             */
/*                                                                    */
/*            Copyright 2024 Battelle Energy Alliance, LLC            */
/*                        ALL RIGHTS RESERVED                         */
/**********************************************************************/

// SPIKE (not a maintained solver): PETSc KSPCG + PCMG on the same problem as
// TorchMultigridSpikeTest, to compare PETSc's own multigrid machinery with the
// hand-written torch V-cycle.
//
// Problem: periodic 2D  A u = -Laplacian(u) + sigma u, h = 1/n, 5-point stencil,
// cell-centered, single rank. Every level operator is a MATSHELL whose MatMult is
// the torch stencil (rediscretized, h doubles per level); interpolation is the
// cell-centered bilinear P and restriction R = P^T / 4, both MATSHELLs, exactly
// as in the torch spike. The coarsest level (n = 8) is a dense Mat solved by LU.
//
// Smoother: KSPCHEBYSHEV (2 iterations) preconditioned by PCJACOBI, set
// explicitly (PCMG's default PC is SOR, which a shell Mat cannot apply). The torch spike uses
// red-black Gauss-Seidel, which PETSc cannot apply to a shell Mat, so iteration counts are not
// expected to match. The smoother changes the preconditioner only, not the system being solved.
//
// DISABLED_CudaTimingSweep runs the same solve, with and without MAT_SPD set on the
// level Mats, with CUDA Vecs/tensors (shell
// callbacks reach the device pointer through VecGet*ArrayAndMemType and run on
// PETSc's stream) and prints KSPSetUp, first and repeat solve times and the cost
// of one PCApply (= one V-cycle, comparable to TorchMultigridSpikeTest's
// DISABLED_CudaVcycleTimingSweep). Run on a GPU machine with
//   --gtest_also_run_disabled_tests
//   --gtest_filter='*PetscTorchMultigridTest.DISABLED_CudaTimingSweep'
// The CUDA path is compiled only when PETSc and libtorch both have CUDA; it has
// not been compiled or run where this was written.

#include "MarlinUtils.h"
#include "gtest/gtest.h"

#include <petscksp.h>

#include <torch/cuda.h>

#include <chrono>
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
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>
#include <petscdevice_cuda.h>
#if __has_include(<nvtx3/nvToolsExt.h>)
#include <nvtx3/nvToolsExt.h>
#define MARLIN_PCMG_NVTX 1
#endif
#else
#define MARLIN_PCMG_CUDA_LEG 0
#endif

namespace
{
constexpr double sigma = 1.0;
constexpr int64_t coarsest_n = 8;
constexpr double rel_tol = 1e-10;

/// Per-level data behind a shell Mat: grid size n (the fine size for transfers)
/// and h^2 of the level.
struct LevelContext
{
  int64_t n;
  double h2;
};

#if MARLIN_PCMG_CUDA_LEG
/// Makes torch enqueue on PETSc's current CUDA stream while a shell callback runs,
/// so the stencil kernels are ordered with PETSc's own vector kernels.
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

/// Views x as an [nx, nx] grid, y as an [ny, ny] grid, and sets y = f(x), on the
/// device the Vec's memory lives on (host Vec -> CPU tensors, CUDA Vec -> the
/// device pointer, never a host round trip).
template <typename F>
PetscErrorCode
applyOnGrids(Vec x, Vec y, int64_t nx, int64_t ny, F f)
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
  (void)memtype;
  const auto in = torch::from_blob(const_cast<PetscScalar *>(x_array), {nx, nx}, options);
  torch::from_blob(y_array, {ny, ny}, options).copy_(f(in));
  PetscCall(VecRestoreArrayReadAndMemType(x, &x_array));
  PetscCall(VecRestoreArrayWriteAndMemType(y, &y_array));
  return PETSC_SUCCESS;
}

torch::Tensor
stencil(const torch::Tensor & u, double h2)
{
  return (4.0 * u - torch::roll(u, 1, 0) - torch::roll(u, -1, 0) - torch::roll(u, 1, 1) -
          torch::roll(u, -1, 1)) /
             h2 +
         sigma * u;
}

PetscErrorCode
levelMult(Mat A, Vec x, Vec y)
{
  LevelContext * ctx;
  PetscCall(MatShellGetContext(A, &ctx));
  return applyOnGrids(
      x, y, ctx->n, ctx->n, [&](const torch::Tensor & u) { return stencil(u, ctx->h2); });
}

/// The stencil diagonal is the same at every point; PCJACOBI asks for it.
PetscErrorCode
levelDiagonal(Mat A, Vec diag)
{
  LevelContext * ctx;
  PetscCall(MatShellGetContext(A, &ctx));
  PetscCall(VecSet(diag, 4.0 / ctx->h2 + sigma));
  return PETSC_SUCCESS;
}

/// Linear interpolation along one axis, periodic, cell-centered (as in the spike).
torch::Tensor
prolongAxis(const torch::Tensor & c, int64_t axis)
{
  const auto even = 0.75 * c + 0.25 * torch::roll(c, 1, axis);
  const auto odd = 0.75 * c + 0.25 * torch::roll(c, -1, axis);
  return torch::stack({even, odd}, axis + 1).flatten(axis, axis + 1);
}

/// P^T along one axis (without the 1/2-per-axis scaling), periodic.
torch::Tensor
restrictAxis(const torch::Tensor & f, int64_t axis)
{
  const int64_t nc = f.size(axis) / 2;
  auto pairs = f.unflatten(axis, {nc, 2});
  const auto even = pairs.select(axis + 1, 0);
  const auto odd = pairs.select(axis + 1, 1);
  return 0.75 * (even + odd) + 0.25 * (torch::roll(odd, 1, axis) + torch::roll(even, -1, axis));
}

/// ctx->n is the FINE size for both transfer shells.
PetscErrorCode
prolongMult(Mat P, Vec coarse, Vec fine)
{
  LevelContext * ctx;
  PetscCall(MatShellGetContext(P, &ctx));
  return applyOnGrids(coarse,
                      fine,
                      ctx->n / 2,
                      ctx->n,
                      [](const torch::Tensor & c) { return prolongAxis(prolongAxis(c, 1), 0); });
}

PetscErrorCode
restrictMult(Mat R, Vec fine, Vec coarse)
{
  LevelContext * ctx;
  PetscCall(MatShellGetContext(R, &ctx));
  return applyOnGrids(fine,
                      coarse,
                      ctx->n,
                      ctx->n / 2,
                      [](const torch::Tensor & f)
                      { return restrictAxis(restrictAxis(f, 1), 0) / 4.0; });
}

Mat
makeShell(int64_t rows, int64_t cols, void * ctx, PetscErrorCode (*mult)(Mat, Vec, Vec), bool cuda)
{
  Mat m;
  PetscCallAbort(PETSC_COMM_SELF, MatCreateShell(PETSC_COMM_SELF, rows, cols, rows, cols, ctx, &m));
  PetscCallAbort(PETSC_COMM_SELF, MatShellSetOperation(m, MATOP_MULT, (void (*)(void))mult));
#if MARLIN_PCMG_CUDA_LEG
  // Vecs that PCMG and the level KSPs create from this Mat live on the device.
  if (cuda)
    PetscCallAbort(PETSC_COMM_SELF, MatShellSetVecType(m, VECSEQCUDA));
#else
  (void)cuda;
#endif
  return m;
}

/// Dense coarsest operator, columns = stencil applied to unit vectors.
Mat
makeCoarseDense(int64_t n, double h2, bool cuda)
{
  const int64_t m = n * n;
  auto columns = torch::zeros({m, m}, torch::kDouble);
  for (int64_t k = 0; k < m; ++k)
  {
    auto e = torch::zeros({n, n}, torch::kDouble);
    e[k / n][k % n] = 1.0;
    columns.select(1, k).copy_(stencil(e, h2).reshape({-1}));
  }
  // PETSc dense storage is column-major: a row-major copy of the transpose.
  const auto storage = columns.t().contiguous();
  Mat A;
  PetscCallAbort(PETSC_COMM_SELF, MatCreateSeqDense(PETSC_COMM_SELF, m, m, nullptr, &A));
  PetscScalar * array;
  PetscCallAbort(PETSC_COMM_SELF, MatDenseGetArray(A, &array));
  std::copy_n(storage.data_ptr<double>(), m * m, array);
  PetscCallAbort(PETSC_COMM_SELF, MatDenseRestoreArray(A, &array));
#if MARLIN_PCMG_CUDA_LEG
  if (cuda)
    PetscCallAbort(PETSC_COMM_SELF, MatConvert(A, MATSEQDENSECUDA, MAT_INPLACE_MATRIX, &A));
#else
  (void)cuda;
#endif
  return A;
}

struct PetscMGResult
{
  PetscInt its;
  double relative_error;
  // Filled only when timing was requested (ms).
  double setup_ms = 0, first_ms = 0, repeat_ms = 0, vcycle_ms = 0;
};

/// Scoped NVTX range (no-op without the NVTX headers or without the CUDA leg).
class Range
{
public:
  explicit Range(const std::string & name)
  {
#ifdef MARLIN_PCMG_NVTX
    nvtxRangePushA(name.c_str());
#else
    (void)name;
#endif
  }
  ~Range()
  {
#ifdef MARLIN_PCMG_NVTX
    nvtxRangePop();
#endif
  }
};

/// Wall time of fn() in ms, device-synchronized on both sides when on CUDA.
template <typename F>
double
timedMs(bool cuda, const std::string & label, F fn)
{
  if (cuda)
    torch::cuda::synchronize();
  Range range(label);
  const auto t0 = std::chrono::steady_clock::now();
  fn();
  if (cuda)
    torch::cuda::synchronize();
  return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
}

PetscMGResult
solvePetscMG(int64_t n, bool cuda = false, bool time_it = false, bool mark_spd = false)
{
  const auto device = cuda ? torch::Device(torch::kCUDA, 0) : torch::Device(torch::kCPU);
  const auto dopts = torch::TensorOptions().dtype(torch::kDouble).device(device);
  const std::string tag = " n=" + std::to_string(n);

  // Levels ordered coarse (0) to fine (L-1), PCMG's convention.
  std::vector<int64_t> sizes;
  for (int64_t s = n; s >= coarsest_n; s /= 2)
    sizes.insert(sizes.begin(), s);
  const PetscInt L = sizes.size();

  std::vector<std::unique_ptr<LevelContext>> contexts;
  for (const auto s : sizes)
    contexts.push_back(std::make_unique<LevelContext>(LevelContext{s, 1.0 / (double(s) * s)}));

  std::vector<Mat> A(L);
  A[0] = makeCoarseDense(sizes[0], contexts[0]->h2, cuda);
  for (PetscInt l = 1; l < L; ++l)
  {
    A[l] = makeShell(sizes[l] * sizes[l], sizes[l] * sizes[l], contexts[l].get(), levelMult, cuda);
    PetscCallAbort(PETSC_COMM_SELF,
                   MatShellSetOperation(A[l], MATOP_GET_DIAGONAL, (void (*)(void))levelDiagonal));
  }
  // Optionally assert what is true of this operator (-Laplacian + sigma, sigma > 0):
  // every level is SPD. PETSc then skips PCJACOBI's host zero-diagonal check
  // (jacobi.c) and estimates Chebyshev bounds with CG instead of GMRES (cheby.c).
  // It does not change the system or its solution.
  if (mark_spd)
    for (auto & m : A)
      PetscCallAbort(PETSC_COMM_SELF, MatSetOption(m, MAT_SPD, PETSC_TRUE));

  KSP ksp;
  PetscCallAbort(PETSC_COMM_SELF, KSPCreate(PETSC_COMM_SELF, &ksp));
  PetscCallAbort(PETSC_COMM_SELF, KSPSetOperators(ksp, A[L - 1], A[L - 1]));
  PetscCallAbort(PETSC_COMM_SELF, KSPSetType(ksp, KSPCG));
  PetscCallAbort(PETSC_COMM_SELF, KSPSetNormType(ksp, KSP_NORM_UNPRECONDITIONED));
  PetscCallAbort(PETSC_COMM_SELF, KSPSetTolerances(ksp, rel_tol, 0.0, PETSC_DEFAULT, n * n));

  PC pc;
  PetscCallAbort(PETSC_COMM_SELF, KSPGetPC(ksp, &pc));
  PetscCallAbort(PETSC_COMM_SELF, PCSetType(pc, PCMG));
  PetscCallAbort(PETSC_COMM_SELF, PCMGSetLevels(pc, L, nullptr));
  PetscCallAbort(PETSC_COMM_SELF, PCMGSetType(pc, PC_MG_MULTIPLICATIVE));
  PetscCallAbort(PETSC_COMM_SELF, PCMGSetCycleType(pc, PC_MG_CYCLE_V));

  std::vector<Mat> transfers;
  for (PetscInt l = 1; l < L; ++l)
  {
    const int64_t nf = sizes[l], nc = sizes[l - 1];
    Mat P = makeShell(nf * nf, nc * nc, contexts[l].get(), prolongMult, cuda);
    Mat R = makeShell(nc * nc, nf * nf, contexts[l].get(), restrictMult, cuda);
    PetscCallAbort(PETSC_COMM_SELF, PCMGSetInterpolation(pc, l, P));
    PetscCallAbort(PETSC_COMM_SELF, PCMGSetRestriction(pc, l, R));
    transfers.push_back(P);
    transfers.push_back(R);
  }

  for (PetscInt l = 0; l < L; ++l)
  {
    KSP smoother;
    PetscCallAbort(PETSC_COMM_SELF, PCMGGetSmoother(pc, l, &smoother));
    PetscCallAbort(PETSC_COMM_SELF, KSPSetOperators(smoother, A[l], A[l]));
    if (l == 0)
    {
      PC coarse_pc;
      PetscCallAbort(PETSC_COMM_SELF, KSPSetType(smoother, KSPPREONLY));
      PetscCallAbort(PETSC_COMM_SELF, KSPGetPC(smoother, &coarse_pc));
      PetscCallAbort(PETSC_COMM_SELF, PCSetType(coarse_pc, PCLU));
    }
    else
    {
      PC smoother_pc;
      PetscCallAbort(PETSC_COMM_SELF, KSPSetType(smoother, KSPCHEBYSHEV));
      PetscCallAbort(PETSC_COMM_SELF, KSPGetPC(smoother, &smoother_pc));
      PetscCallAbort(PETSC_COMM_SELF, PCSetType(smoother_pc, PCJACOBI));
      PetscCallAbort(PETSC_COMM_SELF, KSPSetTolerances(smoother, 0, 0, PETSC_DEFAULT, 2));
      PetscCallAbort(PETSC_COMM_SELF, KSPSetNormType(smoother, KSP_NORM_NONE));
    }
  }

  // Manufactured smooth solution, same as the torch MPI spike (single rank).
  const auto xs = (torch::arange(n, dopts) + 0.5) / n;
  const double two_pi = 2.0 * M_PI;
  const auto x_exact =
      (torch::sin(two_pi * xs).unsqueeze(1) * torch::cos(2.0 * two_pi * xs).unsqueeze(0) +
       0.5 * torch::cos(3.0 * two_pi * xs).unsqueeze(1) +
       0.25 * torch::sin(two_pi * xs).unsqueeze(0))
          .contiguous();
  auto b = stencil(x_exact, contexts[L - 1]->h2).contiguous();
  auto x = torch::zeros({n, n}, dopts);
  auto y = torch::zeros({n, n}, dopts); // PCApply output for the V-cycle timing

  // Zero-copy Vecs over the tensors (host or CUDA memory).
  auto wrap = [&](torch::Tensor & t)
  {
    Vec v;
#if MARLIN_PCMG_CUDA_LEG
    if (cuda)
    {
      PetscCallAbort(
          PETSC_COMM_SELF,
          VecCreateSeqCUDAWithArray(PETSC_COMM_SELF, 1, n * n, t.data_ptr<double>(), &v));
      return v;
    }
#endif
    PetscCallAbort(PETSC_COMM_SELF,
                   VecCreateSeqWithArray(PETSC_COMM_SELF, 1, n * n, t.data_ptr<double>(), &v));
    return v;
  };
  Vec b_vec = wrap(b), x_vec = wrap(x), y_vec = wrap(y);

  PetscMGResult timings{};
  if (time_it)
  {
    timings.setup_ms = timedMs(
        cuda, "PCMG KSPSetUp" + tag, [&] { PetscCallAbort(PETSC_COMM_SELF, KSPSetUp(ksp)); });
    timings.first_ms =
        timedMs(cuda,
                "KSPCG+PCMG first" + tag,
                [&] { PetscCallAbort(PETSC_COMM_SELF, KSPSolve(ksp, b_vec, x_vec)); });
    PetscInt first_its;
    PetscCallAbort(PETSC_COMM_SELF, KSPGetIterationNumber(ksp, &first_its));
    x.zero_();
    timings.repeat_ms =
        timedMs(cuda,
                "KSPCG+PCMG repeat" + tag,
                [&] { PetscCallAbort(PETSC_COMM_SELF, KSPSolve(ksp, b_vec, x_vec)); });
    PetscInt repeat_its;
    PetscCallAbort(PETSC_COMM_SELF, KSPGetIterationNumber(ksp, &repeat_its));
    EXPECT_EQ(first_its, repeat_its);

    // One PCApply = one V-cycle from a zero guess.
    constexpr int reps = 20;
    PetscCallAbort(PETSC_COMM_SELF, PCApply(pc, b_vec, y_vec)); // warm-up
    timings.vcycle_ms = timedMs(cuda,
                                "PCMG PCApply x20" + tag,
                                [&]
                                {
                                  for (int i = 0; i < reps; ++i)
                                    PetscCallAbort(PETSC_COMM_SELF, PCApply(pc, b_vec, y_vec));
                                }) /
                        reps;
    // Leave x holding the repeat solution for the error check below.
  }
  else
    PetscCallAbort(PETSC_COMM_SELF, KSPSolve(ksp, b_vec, x_vec));

  KSPConvergedReason reason;
  PetscInt its;
  PetscCallAbort(PETSC_COMM_SELF, KSPGetConvergedReason(ksp, &reason));
  PetscCallAbort(PETSC_COMM_SELF, KSPGetIterationNumber(ksp, &its));
  EXPECT_GT(reason, 0) << "KSP+PCMG failed to converge, reason " << reason;

  const double relative_error = ((x - x_exact).norm() / x_exact.norm()).item<double>();

  PetscCallAbort(PETSC_COMM_SELF, VecDestroy(&b_vec));
  PetscCallAbort(PETSC_COMM_SELF, VecDestroy(&x_vec));
  PetscCallAbort(PETSC_COMM_SELF, VecDestroy(&y_vec));
  PetscCallAbort(PETSC_COMM_SELF, KSPDestroy(&ksp));
  for (auto & m : A)
    PetscCallAbort(PETSC_COMM_SELF, MatDestroy(&m));
  for (auto & m : transfers)
    PetscCallAbort(PETSC_COMM_SELF, MatDestroy(&m));
  timings.its = its;
  timings.relative_error = relative_error;
  return timings;
}
} // namespace

TEST(PetscTorchMultigridTest, KspCgWithPcmgSolvesAndIsGridIndependent)
{
  std::vector<PetscInt> its;
  for (const int64_t n : {64, 128, 256, 512})
  {
    const auto result = solvePetscMG(n);
    printf("n=%ld PETSc KSPCG+PCMG (Chebyshev/Jacobi smoother): %d its, relative error %.3e\n",
           long(n),
           int(result.its),
           result.relative_error);
    EXPECT_LT(result.relative_error, 1e-8);
    its.push_back(result.its);
  }
  EXPECT_LE(its.back(), its.front() + 2);
}

// The timing path (explicit KSPSetUp, first and repeat solve, PCApply) must give
// the same solution quality as the plain path; this is what the CUDA sweep runs.
TEST(PetscTorchMultigridTest, TimedPathMatchesPlainPath)
{
  const auto plain = solvePetscMG(128);
  const auto timed = solvePetscMG(128, /*cuda=*/false, /*time_it=*/true);
  EXPECT_EQ(plain.its, timed.its);
  EXPECT_LT(timed.relative_error, 1e-8);
  EXPECT_GT(timed.vcycle_ms, 0.0);
}

// Marking the levels SPD changes only how PETSc sets up (Chebyshev eigenvalue
// estimator, Jacobi diagonal check), so the solve must still converge accurately in
// a similar number of iterations (the CG and GMRES eigenvalue estimates differ, so
// the counts need not be identical).
TEST(PetscTorchMultigridTest, MarkingLevelsSpdKeepsTheSolution)
{
  for (const int64_t n : {64, 256})
  {
    const auto plain = solvePetscMG(n);
    const auto spd = solvePetscMG(n, /*cuda=*/false, /*time_it=*/false, /*mark_spd=*/true);
    printf("n=%ld PCMG its: unmarked %d, MAT_SPD %d | relative error %.2e, %.2e\n",
           long(n),
           int(plain.its),
           int(spd.its),
           plain.relative_error,
           spd.relative_error);
    EXPECT_LT(spd.relative_error, 1e-8);
    EXPECT_LE(spd.its, plain.its + 2);
  }
}

// PICARD SWEEP: variable-coefficient problem and preconditioner-update policies.
//
// Test problem (defined here for the benchmark, not a Marlin operator):
//   -div(D(u) grad u) + sigma u = f,  D(u) = exp(beta u),  periodic, cell-centered.
// Discretization: 5-point finite volume, face coefficient = harmonic mean of the two
// adjacent cell values; coarse levels use the 2x2 arithmetic mean of D and the same
// face rule (rediscretization). f is manufactured from a smooth u_exact so the
// under-relaxed Picard iteration
//   u_{k+1} = (1 - omega) u_k + omega solve(A(D(u_k))),  omega = picard_omega,
// has the known fixed point u_exact (relaxation does not move the fixed point).
// Unrelaxed Picard diverged at beta = 9 (step-0 iterate reached D ~ 1e13).
// Each linear solve starts from the previous iterate and uses the same
// unpreconditioned relative tolerance as the rest of this file.
//
// The shell MatMult always reads the CURRENT coefficients, so the operator is exact
// in every policy; only the PCMG setup (Chebyshev bounds, Jacobi diagonals) can be
// stale. The coarse dense LU is refactored every step under every policy, because
// setCoefficients() refills it and MatDenseRestoreArray bumps its state. Policies:
//   Always    - full rebuild before every Picard step (KSPSetOperators on every level)
//   BumpOnly  - only bump the level Mats' state. This is a PARTIAL rebuild, inside
//               the next KSPSolve: PCSetUp_MG re-sets-up the finest smoother only, and
//               the other levels' Chebyshev keeps its old eigenvalue bounds; the Jacobi
//               diagonals are rebuilt lazily on every level (PETSc 3.25.4, mg.c/
//               cheby.c). Bumping also clears MAT_SPD unless MAT_SPD_ETERNAL is set.
//   Never     - set up once, before step 0
//   Threshold - update when the last solve took more than baseline + extra_its
//               iterations (baseline = iterations right after the last update)
// A solve that ends with a negative converged reason is redone once after a forced
// update and counted.
namespace
{
/// Picard under-relaxation factor (agreed with the user for this benchmark).
constexpr double picard_omega = 0.5;

struct VarLevelContext
{
  int64_t n;
  double h2;
  torch::Tensor dxp, dyp; // face coefficients between cell i and i+1 along dims 0, 1
};

torch::Tensor
varStencil(const torch::Tensor & u, const VarLevelContext & c)
{
  // Works on [n, n] or batched [m, n, n] input (grids on the last two dims).
  const auto fx = c.dxp * (u - torch::roll(u, -1, -2));
  const auto fy = c.dyp * (u - torch::roll(u, -1, -1));
  return (fx - torch::roll(fx, 1, -2) + fy - torch::roll(fy, 1, -1)) / c.h2 + sigma * u;
}

void
setFaceCoefficients(VarLevelContext & c, const torch::Tensor & D)
{
  auto harmonic = [&](int64_t dim)
  {
    const auto shifted = torch::roll(D, -1, dim);
    return 2.0 * D * shifted / (D + shifted);
  };
  c.dxp = harmonic(0);
  c.dyp = harmonic(1);
}

PetscErrorCode
varLevelMult(Mat A, Vec x, Vec y)
{
  VarLevelContext * ctx;
  PetscCall(MatShellGetContext(A, &ctx));
  return applyOnGrids(
      x, y, ctx->n, ctx->n, [&](const torch::Tensor & u) { return varStencil(u, *ctx); });
}

template <typename F>
PetscErrorCode
writeOnGrid(Vec y, int64_t ny, F f)
{
  PetscScalar * y_array;
  PetscMemType memtype;
  PetscCall(VecGetArrayWriteAndMemType(y, &y_array, &memtype));
  auto options = torch::TensorOptions().dtype(torch::kDouble);
#if MARLIN_PCMG_CUDA_LEG
  std::unique_ptr<PetscStreamGuard> guard;
  if (PetscMemTypeDevice(memtype))
  {
    options = options.device(torch::kCUDA, c10::cuda::current_device());
    guard = std::make_unique<PetscStreamGuard>();
  }
#endif
  (void)memtype;
  torch::from_blob(y_array, {ny, ny}, options).copy_(f());
  PetscCall(VecRestoreArrayWriteAndMemType(y, &y_array));
  return PETSC_SUCCESS;
}

/// Pointwise diagonal of the variable-coefficient operator (what PCJACOBI asks for).
/// Calls are counted: PCJACOBI asks for the diagonal exactly when PCMG rebuilds a
/// smoother, so the count shows whether an update really reached the levels.
int diagonal_calls = 0;

PetscErrorCode
varLevelDiagonal(Mat A, Vec diag)
{
  ++diagonal_calls;
  VarLevelContext * ctx;
  PetscCall(MatShellGetContext(A, &ctx));
  return writeOnGrid(diag,
                     ctx->n,
                     [&]
                     {
                       return (ctx->dxp + torch::roll(ctx->dxp, 1, 0) + ctx->dyp +
                               torch::roll(ctx->dyp, 1, 1)) /
                                  ctx->h2 +
                              sigma;
                     });
}

/// Dense coarsest operator from the current coefficients: row k of the batched
/// stencil applied to unit vectors is column k of A, and A is symmetric.
void
fillCoarseDense(Mat A, const VarLevelContext & c)
{
  const int64_t m = c.n * c.n;
  const auto E = torch::eye(m, c.dxp.options()).view({m, c.n, c.n});
  const auto dense = varStencil(E, c).reshape({m, m}).cpu().contiguous();
  PetscScalar * array;
  PetscCallAbort(PETSC_COMM_SELF, MatDenseGetArray(A, &array));
  std::copy_n(dense.data_ptr<double>(), m * m, array);
  PetscCallAbort(PETSC_COMM_SELF, MatDenseRestoreArray(A, &array));
}

Vec
wrapTensor(torch::Tensor & t, int64_t n, bool cuda)
{
  Vec v;
#if MARLIN_PCMG_CUDA_LEG
  if (cuda)
  {
    PetscCallAbort(PETSC_COMM_SELF,
                   VecCreateSeqCUDAWithArray(PETSC_COMM_SELF, 1, n * n, t.data_ptr<double>(), &v));
    return v;
  }
#endif
  (void)cuda;
  PetscCallAbort(PETSC_COMM_SELF,
                 VecCreateSeqWithArray(PETSC_COMM_SELF, 1, n * n, t.data_ptr<double>(), &v));
  return v;
}

/// KSPCG + PCMG hierarchy for the variable-coefficient operator; same structure as
/// solvePetscMG (shell level Mats, bilinear P / R = P^T/4, Chebyshev+Jacobi
/// smoothers, dense LU at n = 8) with updatable coefficients.
class PicardMG
{
public:
  PicardMG(int64_t n, bool cuda, torch::Device device)
  {
    for (int64_t s = n; s >= coarsest_n; s /= 2)
      _sizes.insert(_sizes.begin(), s);
    const PetscInt L = _sizes.size();
    const auto dopts = torch::TensorOptions().dtype(torch::kDouble).device(device);

    for (const auto s : _sizes)
    {
      auto ctx = std::make_unique<VarLevelContext>();
      ctx->n = s;
      ctx->h2 = 1.0 / (double(s) * s);
      ctx->dxp = torch::ones({s, s}, dopts);
      ctx->dyp = torch::ones({s, s}, dopts);
      _contexts.push_back(std::move(ctx));
      _transfer_contexts.push_back(
          std::make_unique<LevelContext>(LevelContext{s, 1.0 / (double(s) * s)}));
    }

    _A.resize(L);
    const int64_t mc = _sizes[0] * _sizes[0];
    PetscCallAbort(PETSC_COMM_SELF, MatCreateSeqDense(PETSC_COMM_SELF, mc, mc, nullptr, &_A[0]));
#if MARLIN_PCMG_CUDA_LEG
    if (cuda)
      PetscCallAbort(PETSC_COMM_SELF,
                     MatConvert(_A[0], MATSEQDENSECUDA, MAT_INPLACE_MATRIX, &_A[0]));
#endif
    fillCoarseDense(_A[0], *_contexts[0]);
    for (PetscInt l = 1; l < L; ++l)
    {
      _A[l] = makeShell(
          _sizes[l] * _sizes[l], _sizes[l] * _sizes[l], _contexts[l].get(), varLevelMult, cuda);
      PetscCallAbort(
          PETSC_COMM_SELF,
          MatShellSetOperation(_A[l], MATOP_GET_DIAGONAL, (void (*)(void))varLevelDiagonal));
    }

    PetscCallAbort(PETSC_COMM_SELF, KSPCreate(PETSC_COMM_SELF, &_ksp));
    PetscCallAbort(PETSC_COMM_SELF, KSPSetOperators(_ksp, _A[L - 1], _A[L - 1]));
    PetscCallAbort(PETSC_COMM_SELF, KSPSetType(_ksp, KSPCG));
    PetscCallAbort(PETSC_COMM_SELF, KSPSetNormType(_ksp, KSP_NORM_UNPRECONDITIONED));
    PetscCallAbort(PETSC_COMM_SELF, KSPSetInitialGuessNonzero(_ksp, PETSC_TRUE));
    PetscCallAbort(PETSC_COMM_SELF, KSPSetTolerances(_ksp, rel_tol, 0.0, PETSC_DEFAULT, 500));

    PC pc;
    PetscCallAbort(PETSC_COMM_SELF, KSPGetPC(_ksp, &pc));
    PetscCallAbort(PETSC_COMM_SELF, PCSetType(pc, PCMG));
    PetscCallAbort(PETSC_COMM_SELF, PCMGSetLevels(pc, L, nullptr));
    PetscCallAbort(PETSC_COMM_SELF, PCMGSetType(pc, PC_MG_MULTIPLICATIVE));
    PetscCallAbort(PETSC_COMM_SELF, PCMGSetCycleType(pc, PC_MG_CYCLE_V));
    for (PetscInt l = 1; l < L; ++l)
    {
      const int64_t nf = _sizes[l], nc = _sizes[l - 1];
      Mat P = makeShell(nf * nf, nc * nc, _transfer_contexts[l].get(), prolongMult, cuda);
      Mat R = makeShell(nc * nc, nf * nf, _transfer_contexts[l].get(), restrictMult, cuda);
      PetscCallAbort(PETSC_COMM_SELF, PCMGSetInterpolation(pc, l, P));
      PetscCallAbort(PETSC_COMM_SELF, PCMGSetRestriction(pc, l, R));
      _transfers.push_back(P);
      _transfers.push_back(R);
    }
    _smoothers.resize(L);
    for (PetscInt l = 0; l < L; ++l)
    {
      KSP smoother;
      PetscCallAbort(PETSC_COMM_SELF, PCMGGetSmoother(pc, l, &smoother));
      PetscCallAbort(PETSC_COMM_SELF, KSPSetOperators(smoother, _A[l], _A[l]));
      _smoothers[l] = smoother;
      if (l == 0)
      {
        PC coarse_pc;
        PetscCallAbort(PETSC_COMM_SELF, KSPSetType(smoother, KSPPREONLY));
        PetscCallAbort(PETSC_COMM_SELF, KSPGetPC(smoother, &coarse_pc));
        PetscCallAbort(PETSC_COMM_SELF, PCSetType(coarse_pc, PCLU));
      }
      else
      {
        PC smoother_pc;
        PetscCallAbort(PETSC_COMM_SELF, KSPSetType(smoother, KSPCHEBYSHEV));
        PetscCallAbort(PETSC_COMM_SELF, KSPGetPC(smoother, &smoother_pc));
        PetscCallAbort(PETSC_COMM_SELF, PCSetType(smoother_pc, PCJACOBI));
        PetscCallAbort(PETSC_COMM_SELF, KSPSetTolerances(smoother, 0, 0, PETSC_DEFAULT, 2));
        PetscCallAbort(PETSC_COMM_SELF, KSPSetNormType(smoother, KSP_NORM_NONE));
      }
    }
  }

  PicardMG(const PicardMG &) = delete;
  PicardMG & operator=(const PicardMG &) = delete;

  ~PicardMG()
  {
    PetscCallAbort(PETSC_COMM_SELF, KSPDestroy(&_ksp));
    for (auto & m : _A)
      PetscCallAbort(PETSC_COMM_SELF, MatDestroy(&m));
    for (auto & m : _transfers)
      PetscCallAbort(PETSC_COMM_SELF, MatDestroy(&m));
  }

  KSP ksp() const { return _ksp; }
  const VarLevelContext & fine() const { return *_contexts.back(); }

  /// Install D (fine-level cell values) on every level and refresh the dense coarse
  /// matrix. The shell operators see the new values immediately; the PCMG setup
  /// does not until update() says so.
  void setCoefficients(const torch::Tensor & D)
  {
    auto level_D = D;
    for (int64_t l = _sizes.size() - 1; l >= 0; --l)
    {
      setFaceCoefficients(*_contexts[l], level_D);
      if (l > 0)
      {
        const int64_t nc = _sizes[l - 1];
        level_D = level_D.view({nc, 2, nc, 2}).mean({1, 3});
      }
    }
    fillCoarseDense(_A[0], *_contexts[0]);
  }

  /// Bump the shell Mats' state (MatAssemblyBegin/End); if `reattach`, also call
  /// KSPSetOperators on every level so the next KSPSetUp rebuilds every level's
  /// smoother here. Without `reattach` the KSPSetUp below is a no-op and the bump only
  /// causes the partial lazy rebuild described at the policy list.
  void update(bool reattach)
  {
    for (size_t l = 1; l < _A.size(); ++l)
    {
      // The documented public way to bump a MATSHELL's state (PetscObjectStateIncrease
      // is a private-header call).
      PetscCallAbort(PETSC_COMM_SELF, MatAssemblyBegin(_A[l], MAT_FINAL_ASSEMBLY));
      PetscCallAbort(PETSC_COMM_SELF, MatAssemblyEnd(_A[l], MAT_FINAL_ASSEMBLY));
    }
    if (reattach)
    {
      for (size_t l = 0; l < _A.size(); ++l)
        PetscCallAbort(PETSC_COMM_SELF, KSPSetOperators(_smoothers[l], _A[l], _A[l]));
      PetscCallAbort(PETSC_COMM_SELF, KSPSetOperators(_ksp, _A.back(), _A.back()));
    }
    PetscCallAbort(PETSC_COMM_SELF, KSPSetUp(_ksp));
  }

private:
  std::vector<int64_t> _sizes;
  std::vector<std::unique_ptr<VarLevelContext>> _contexts;
  std::vector<std::unique_ptr<LevelContext>> _transfer_contexts;
  std::vector<Mat> _A, _transfers;
  std::vector<KSP> _smoothers;
  KSP _ksp;
};

enum class UpdatePolicy
{
  Always,
  BumpOnly,
  Never,
  Threshold
};

std::string
policyName(UpdatePolicy p, int extra_its)
{
  switch (p)
  {
    case UpdatePolicy::Always:
      return "always";
    case UpdatePolicy::BumpOnly:
      return "bump-only";
    case UpdatePolicy::Never:
      return "never";
    case UpdatePolicy::Threshold:
      return "threshold+" + std::to_string(extra_its);
  }
  return "";
}

struct PicardStep
{
  PetscInt its;
  int reason;
  int failed_reason;  // converged reason of the attempt that was retried (0 if none)
  int diagonal_calls; // MatGetDiagonal callbacks during this step's update and solve
  double update_ms, solve_ms;
  bool updated, retried;
  double change; // ||u_{k+1} - u_k|| / ||u_{k+1}||
};

struct PicardRun
{
  std::vector<PicardStep> steps;
  double coeff_ms = 0, error = 0;
  torch::Tensor u;
};

PicardRun
runPicard(int64_t n, bool cuda, double beta, int num_steps, UpdatePolicy policy, int extra_its)
{
  const auto device = cuda ? torch::Device(torch::kCUDA, 0) : torch::Device(torch::kCPU);
  const auto dopts = torch::TensorOptions().dtype(torch::kDouble).device(device);
  const std::string tag = " n=" + std::to_string(n);
  PicardMG mg(n, cuda, device);

  // u_exact in [0.1, 0.9]; f from the same discretization at D(u_exact).
  const auto xs = ((torch::arange(n, dopts) + 0.5) / n);
  const double two_pi = 2.0 * M_PI;
  const auto u_exact =
      (0.5 +
       0.25 * torch::sin(two_pi * xs).unsqueeze(1) * torch::cos(2.0 * two_pi * xs).unsqueeze(0) +
       0.1 * torch::cos(3.0 * two_pi * xs).unsqueeze(1) +
       0.05 * torch::sin(two_pi * xs).unsqueeze(0))
          .contiguous();
  mg.setCoefficients(torch::exp(beta * u_exact));
  auto b = varStencil(u_exact, mg.fine()).contiguous();

  auto x = torch::full({n, n}, 0.5, dopts); // Picard start u_0 = 0.5
  Vec b_vec = wrapTensor(b, n, cuda), x_vec = wrapTensor(x, n, cuda);

  PicardRun run;
  PetscInt baseline = -1, last_its = 0;
  for (int step = 0; step < num_steps; ++step)
  {
    const auto u_prev = x.clone();
    run.coeff_ms +=
        timedMs(cuda, "coefficients" + tag, [&] { mg.setCoefficients(torch::exp(beta * u_prev)); });

    PicardStep rec{};
    const int diagonal_calls_before = diagonal_calls;
    const bool want_update = step == 0 || policy == UpdatePolicy::Always ||
                             (policy == UpdatePolicy::Threshold && last_its > baseline + extra_its);
    if (want_update)
    {
      rec.updated = true;
      rec.update_ms = timedMs(cuda, "PC update" + tag, [&] { mg.update(true); });
    }
    else if (policy == UpdatePolicy::BumpOnly)
      rec.update_ms = timedMs(cuda, "PC bump" + tag, [&] { mg.update(false); });

    auto solve_once = [&]
    {
      rec.solve_ms +=
          timedMs(cuda,
                  "KSPSolve" + tag,
                  [&] { PetscCallAbort(PETSC_COMM_SELF, KSPSolve(mg.ksp(), b_vec, x_vec)); });
      KSPConvergedReason reason;
      PetscCallAbort(PETSC_COMM_SELF, KSPGetConvergedReason(mg.ksp(), &reason));
      PetscCallAbort(PETSC_COMM_SELF, KSPGetIterationNumber(mg.ksp(), &rec.its));
      rec.reason = reason;
    };
    solve_once();
    if (rec.reason < 0 && !rec.updated)
    {
      // Stale or indefinite preconditioner (or no convergence): force an update, redo.
      rec.retried = true;
      rec.failed_reason = rec.reason;
      rec.update_ms += timedMs(cuda, "PC update (retry)" + tag, [&] { mg.update(true); });
      rec.updated = true;
      x.copy_(u_prev);
      // solve_ms keeps the failed attempt: its cost is part of the policy.
      solve_once();
    }
    // Under-relaxation: x currently holds the linear-solve result.
    x.mul_(picard_omega).add_(u_prev, 1.0 - picard_omega);
    if (rec.updated)
      baseline = rec.its;
    last_its = rec.its;
    rec.diagonal_calls = diagonal_calls - diagonal_calls_before;
    rec.change = ((x - u_prev).norm() / x.norm()).item<double>();
    run.steps.push_back(rec);
  }
  run.error = ((x - u_exact).norm() / u_exact.norm()).item<double>();
  run.u = x.clone();

  PetscCallAbort(PETSC_COMM_SELF, VecDestroy(&b_vec));
  PetscCallAbort(PETSC_COMM_SELF, VecDestroy(&x_vec));
  return run;
}

void
printPicardRun(int64_t n, double beta, UpdatePolicy policy, int extra_its, const PicardRun & r)
{
  double update = 0, solve = 0;
  int updates = 0, retries = 0, negatives = 0;
  std::string its, upd, diag, failed;
  PetscInt max_its = 0;
  for (const auto & s : r.steps)
  {
    update += s.update_ms;
    solve += s.solve_ms;
    updates += s.updated;
    retries += s.retried;
    negatives += s.reason < 0;
    max_its = std::max(max_its, s.its);
    its += std::to_string(s.its) + " ";
    diag += std::to_string(s.diagonal_calls) + " ";
    if (s.retried)
      failed += "step" + std::to_string(&s - r.steps.data()) + ":reason" +
                std::to_string(s.failed_reason) + " ";
    upd += s.retried ? "R" : (s.updated ? "U" : ".");
  }
  printf("n=%ld beta=%.1f policy=%-12s total %.1f ms (update %.1f + solve %.1f), %d updates, %d "
         "retries, %d failed-final | its: %s| upd: %s | diag calls: %s| retried: %s| max its %d | "
         "last Picard change %.2e, "
         "error vs u_exact %.2e\n",
         long(n),
         beta,
         policyName(policy, extra_its).c_str(),
         update + solve,
         update,
         solve,
         updates,
         retries,
         negatives,
         its.c_str(),
         upd.c_str(),
         diag.c_str(),
         failed.empty() ? "none " : failed.c_str(),
         int(max_its),
         r.steps.back().change,
         r.error);
}
} // namespace

// CPU check of the Picard machinery: every policy must reach the same iterate (the
// policy changes only the preconditioner), and the Picard iteration must converge
// to the manufactured solution.
TEST(PetscTorchMultigridTest, PicardPoliciesReachTheSameIterate)
{
  const int64_t n = 64;
  const double beta = 3.0;
  const int steps = 30;
  const auto reference = runPicard(n, false, beta, steps, UpdatePolicy::Always, 0);
  printPicardRun(n, beta, UpdatePolicy::Always, 0, reference);
  EXPECT_LT(reference.steps.back().change, 1e-6);
  for (const auto [policy, extra] : {std::pair{UpdatePolicy::BumpOnly, 0},
                                     std::pair{UpdatePolicy::Never, 0},
                                     std::pair{UpdatePolicy::Threshold, 2}})
  {
    const auto run = runPicard(n, false, beta, steps, policy, extra);
    printPicardRun(n, beta, policy, extra, run);
    EXPECT_LT((run.u - reference.u).norm().item<double>() / reference.u.norm().item<double>(),
              1e-8);
  }
}

#if MARLIN_PCMG_CUDA_LEG
// Timing only; correctness is asserted through the relative error. n = 256..8192
// matches the torch V-cycle sweep. Per size: KSPSetUp, first and repeat solve,
// iteration count, and the cost of one V-cycle (PCApply).
TEST(PetscTorchMultigridTest, DISABLED_CudaTimingSweep)
{
  ASSERT_TRUE(torch::cuda::is_available()) << "needs a CUDA-enabled libtorch and a GPU";
  for (const int64_t n : {256, 512, 1024, 2048, 4096, 8192})
  {
    for (const bool spd : {false, true})
    {
      const auto r = solvePetscMG(n, /*cuda=*/true, /*time_it=*/true, spd);
      EXPECT_LT(r.relative_error, 1e-8);
      printf("n=%ld %s PETSc KSPCG+PCMG CUDA: setup %.3f ms, %d its, solve %.1f ms, repeat %.1f ms "
             "(%.3f ms/it) | PCApply (V-cycle) %.3f ms | relative error %.2e\n",
             long(n),
             spd ? "MAT_SPD " : "unmarked",
             r.setup_ms,
             int(r.its),
             r.first_ms,
             r.repeat_ms,
             r.repeat_ms / r.its,
             r.vcycle_ms,
             r.relative_error);
    }
  }
}
#endif

#if MARLIN_PCMG_CUDA_LEG
// Timing only. beta = 3 and 9 give a D contrast of about 11x and 1300x over u_exact.
// 30 steps: with omega = 0.5 the Picard contraction is at best 0.5 per step.
TEST(PetscTorchMultigridTest, DISABLED_CudaPicardSweep)
{
  ASSERT_TRUE(torch::cuda::is_available()) << "needs a CUDA-enabled libtorch and a GPU";
  constexpr int steps = 30;
  for (const int64_t n : {1024, 4096})
    for (const double beta : {3.0, 9.0})
      for (const auto [policy, extra] : {std::pair{UpdatePolicy::Always, 0},
                                         std::pair{UpdatePolicy::BumpOnly, 0},
                                         std::pair{UpdatePolicy::Never, 0},
                                         std::pair{UpdatePolicy::Threshold, 2}})
        printPicardRun(n, beta, policy, extra, runPicard(n, true, beta, steps, policy, extra));
}
#endif
