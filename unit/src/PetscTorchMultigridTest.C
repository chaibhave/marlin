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
// Smoother: PCMG's default for matrix-free levels, KSPCHEBYSHEV (2 iterations)
// preconditioned by PCJACOBI. The torch spike uses red-black Gauss-Seidel, which
// PETSc cannot apply to a shell Mat, so iteration counts are not expected to match.
// The smoother changes the preconditioner only, not the system being solved.
//
// DISABLED_CudaTimingSweep runs the same solve with CUDA Vecs/tensors (shell
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
  PetscMemType memtype;
  PetscCall(VecGetArrayReadAndMemType(x, &x_array, &memtype));
  PetscCall(VecGetArrayWriteAndMemType(y, &y_array, nullptr));
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
makeShell(int64_t rows,
          int64_t cols,
          LevelContext * ctx,
          PetscErrorCode (*mult)(Mat, Vec, Vec),
          bool cuda)
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
solvePetscMG(int64_t n, bool cuda = false, bool time_it = false)
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

#if MARLIN_PCMG_CUDA_LEG
// Timing only; correctness is asserted through the relative error. n = 256..8192
// matches the torch V-cycle sweep. Per size: KSPSetUp, first and repeat solve,
// iteration count, and the cost of one V-cycle (PCApply).
TEST(PetscTorchMultigridTest, DISABLED_CudaTimingSweep)
{
  ASSERT_TRUE(torch::cuda::is_available()) << "needs a CUDA-enabled libtorch and a GPU";
  for (const int64_t n : {256, 512, 1024, 2048, 4096, 8192})
  {
    const auto r = solvePetscMG(n, /*cuda=*/true, /*time_it=*/true);
    EXPECT_LT(r.relative_error, 1e-8);
    printf("n=%ld PETSc KSPCG+PCMG CUDA: setup %.3f ms, %d its, solve %.1f ms, repeat %.1f ms "
           "(%.3f ms/it) | PCApply (V-cycle) %.3f ms | relative error %.2e\n",
           long(n),
           r.setup_ms,
           int(r.its),
           r.first_ms,
           r.repeat_ms,
           r.repeat_ms / r.its,
           r.vcycle_ms,
           r.relative_error);
  }
}
#endif
