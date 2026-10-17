/**********************************************************************/
/*                     DO NOT MODIFY THIS HEADER                      */
/*            Marlin, a Fourier spectral solver for MOOSE             */
/*                                                                    */
/*            Copyright 2024 Battelle Energy Alliance, LLC            */
/*                        ALL RIGHTS RESERVED                         */
/**********************************************************************/

// PR1a: torch-native (MooseTensor::conjugateGradientSolve) vs. PETSc KSPSolve,
// both matrix-free against the identical 2D 5-point Laplacian stencil action
// FiniteDifferenceLaplacian uses. See the roadmap plan's PR1a section for
// context (zero-copy requirement, R6, decision 5 in Sec. 3).

#include "MarlinUtils.h"
#include "MooseError.h"
#include "gtest/gtest.h"

#include <petscksp.h>

#include <chrono>
#include <sys/resource.h>

// Section 7 (CUDA leg) needs both PETSc's CUDA Vec API and libtorch's CUDA
// stream-interop headers. Guard on both independently: PetscDefined(HAVE_CUDA)
// is PETSc's own compile-time flag (false here: this machine's conda PETSc is
// CPU-only), and the __has_include check detects whether libtorch was built
// with CUDA (ATen/cuda/CUDAConfig.h only exists in a CUDA-enabled build; this
// machine's libtorch is CPU-only, so neither branch compiles here). Combining
// both avoids compiling PETSc-CUDA calls against a CPU-only libtorch or vice
// versa, either of which would fail to link.
#if defined(__has_include)
#if __has_include(<ATen/cuda/CUDAConfig.h>)
#define MARLIN_TORCH_CUDA_AVAILABLE 1
#endif
#endif
#if !defined(MARLIN_TORCH_CUDA_AVAILABLE)
#define MARLIN_TORCH_CUDA_AVAILABLE 0
#endif
#if PetscDefined(HAVE_CUDA) && MARLIN_TORCH_CUDA_AVAILABLE
#define MARLIN_PETSC_TORCH_CUDA_LEG 1
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>
#include <petscdevice_cuda.h>
#else
#define MARLIN_PETSC_TORCH_CUDA_LEG 0
#endif

namespace
{
/// Process peak resident set size, in KB, for the memory-scaling benchmarks
/// below (Sections 4/6). ru_maxrss units differ by platform: bytes on
/// macOS/BSD, KB on Linux. This file is developed/run on macOS (plan's
/// PR1a status); the __APPLE__ branch converts to KB for a consistent unit
/// across platforms.
long
peakRssKB()
{
  struct rusage usage;
  getrusage(RUSAGE_SELF, &usage);
#if defined(__APPLE__)
  return usage.ru_maxrss / 1024;
#else
  return usage.ru_maxrss;
#endif
}
} // namespace

// SECTION 1: problem setup - periodic 2D Poisson-like SPD system and the
// shared matrix-free stencil action, applied identically by both legs.
namespace
{

// 3-point 1D Laplacian coefficients, matching FiniteDifferenceLaplacian's
// stencil_width=3 case (src/tensor_computes/FiniteDifferenceLaplacian.C).
constexpr double c0_3 = -2.0;
constexpr double c1_3 = 1.0;

/// Applies -Laplacian(u) + shift*u on a periodic nxn grid (unit grid spacing),
/// via the identical 5-point stencil coefficients FiniteDifferenceLaplacian
/// uses for its 3-point case, with circular padding for periodicity. The
/// shift makes the operator SPD (the bare periodic Laplacian is singular,
/// with the constant vector as its null space), as CG requires.
class PeriodicStencilOperator
{
public:
  PeriodicStencilOperator(int64_t n, double shift, torch::Device device = torch::kCPU)
    : _n(n), _shift(shift)
  {
    _kernel =
        torch::zeros({1, 1, 3, 3}, torch::TensorOptions().dtype(torch::kDouble).device(device));
    _kernel[0][0][1][1] = -2.0 * c0_3;
    _kernel[0][0][0][1] = -c1_3;
    _kernel[0][0][2][1] = -c1_3;
    _kernel[0][0][1][0] = -c1_3;
    _kernel[0][0][1][2] = -c1_3;
  }

  /// Operator action A(x) = -Laplacian(x) + shift*x, for a flat length-n^2
  /// vector representing a row-major nxn periodic field.
  torch::Tensor operator()(const torch::Tensor & x) const
  {
    auto grid = x.view({1, 1, _n, _n});
    auto padded = torch::nn::functional::pad(
        grid, torch::nn::functional::PadFuncOptions({1, 1, 1, 1}).mode(torch::kCircular));
    auto lap_term = torch::nn::functional::conv2d(padded, _kernel);
    return (lap_term.view({_n * _n}) + _shift * x);
  }

  int64_t size() const { return _n * _n; }

private:
  int64_t _n;
  double _shift;
  torch::Tensor _kernel;
};

/// Jacobi (diagonal-scaling) preconditioner for any of this file's stencil
/// operators, whose diagonal entry is identical at every grid point (so a
/// single scalar characterizes it): 4 + shift for PeriodicStencilOperator,
/// 2*Dx + 2*Dy + shift for AnisotropicStencilOperator (Sec. 5).
class JacobiPreconditioner
{
public:
  explicit JacobiPreconditioner(double diag_value) : _inv_diag(1.0 / diag_value) {}
  torch::Tensor operator()(const torch::Tensor & r) const { return r * _inv_diag; }

private:
  double _inv_diag;
};

} // namespace

// SECTION 2: torch-native leg - MooseTensor::conjugateGradientSolve driven by
// the stencil functor directly (no PETSc involved at all).
namespace
{

constexpr double shift = 1.0;
constexpr double rel_tol = 1e-10;

/// Solves A*x = b for a manufactured, exactly-known solution x_exact via the
/// torch-native leg, and returns (x, iterations, final residual norm).
std::tuple<torch::Tensor, unsigned int, double>
solveTorchNative(int64_t n, const torch::Tensor & b)
{
  // The stencil's conv2d kernel must live on the same device as b, or torch
  // raises a device-mismatch error (CPU weight vs. CUDA input) rather than
  // silently copying - this matters once this leg is also used as the
  // comparison baseline for a CUDA b (Sec. 7).
  PeriodicStencilOperator A(n, shift, b.device());
  JacobiPreconditioner M(4.0 + shift);
  return MooseTensor::conjugateGradientSolve(A, b, torch::Tensor(), rel_tol, 0, M);
}

} // namespace

TEST(PetscTorchMatrixFreeBenchmarkTest, TorchNativeLegSolvesStencilSystem)
{
  const int64_t n = 16;
  PeriodicStencilOperator A(n, shift);

  // Manufacture b from a known x_exact so correctness is checked against a
  // ground truth, not just "CG terminated".
  const auto x_exact = torch::randn({n * n}, torch::kDouble);
  const auto b = A(x_exact);

  const auto [x, it, norm] = solveTorchNative(n, b);

  EXPECT_GT(it, 0u);
  EXPECT_NEAR((x - x_exact).norm().item<double>(), 0.0, 1e-6);
  EXPECT_NEAR((A(x) - b).norm().item<double>(), 0.0, 1e-6);
}

// SECTION 3: PETSc leg - MatCreateShell wrapping the identical stencil functor,
// KSPSolve with matching (unpreconditioned) stopping test, Vec built both
// zero-copy and via naive copy; correctness + zero-copy pointer-equality test.
namespace
{

/// Context handed to the MatShell's MatMult/GetDiagonal callbacks: the same
/// stencil operator, a scratch tensor to view PETSc's raw Vec arrays through
/// without copying, and the stencil's (constant, grid-point-independent)
/// diagonal entry (4 + shift) that PCJACOBI needs via MatGetDiagonal.
struct ShellContext
{
  const PeriodicStencilOperator * op;
  int64_t n;
  double diag_value;
};

PetscErrorCode
shellMatMult(Mat mat, Vec x, Vec y)
{
  ShellContext * ctx;
  PetscCallAbort(PETSC_COMM_SELF, MatShellGetContext(mat, &ctx));

  const PetscScalar * x_arr;
  PetscScalar * y_arr;
  PetscCallAbort(PETSC_COMM_SELF, VecGetArrayRead(x, &x_arr));
  PetscCallAbort(PETSC_COMM_SELF, VecGetArray(y, &y_arr));

  // View PETSc's raw arrays as torch tensors with no copy: the MatMult
  // callback is the one point execution crosses into torch's kernel per
  // iteration (plan Sec. 2.0a).
  auto x_tensor =
      torch::from_blob(const_cast<PetscScalar *>(x_arr), {ctx->n * ctx->n}, torch::kDouble);
  auto result = (*ctx->op)(x_tensor);
  std::memcpy(y_arr, result.data_ptr<double>(), ctx->n * ctx->n * sizeof(double));

  PetscCallAbort(PETSC_COMM_SELF, VecRestoreArrayRead(x, &x_arr));
  PetscCallAbort(PETSC_COMM_SELF, VecRestoreArray(y, &y_arr));
  return PETSC_SUCCESS;
}

/// PCJACOBI calls MatGetDiagonal to build its diagonal scaling; a MatShell
/// has no matrix entries to read it from, so this supplies the stencil's
/// known-constant diagonal (4 + shift, same value JacobiPreconditioner uses)
/// at every grid point.
PetscErrorCode
shellGetDiagonal(Mat mat, Vec diag)
{
  ShellContext * ctx;
  PetscCallAbort(PETSC_COMM_SELF, MatShellGetContext(mat, &ctx));
  PetscCallAbort(PETSC_COMM_SELF, VecSet(diag, ctx->diag_value));
  return PETSC_SUCCESS;
}

/// Builds a KSP (CG+Jacobi) against a MatCreateShell wrapping `op`, with the
/// same unpreconditioned stopping test as conjugateGradientSolve so iteration
/// counts are comparable like-for-like.
KSP
buildShellKSP(const PeriodicStencilOperator & op, ShellContext & ctx, int64_t n)
{
  Mat mat;
  PetscCallAbort(PETSC_COMM_SELF,
                 MatCreateShell(PETSC_COMM_SELF, n * n, n * n, n * n, n * n, &ctx, &mat));
  PetscCallAbort(PETSC_COMM_SELF,
                 MatShellSetOperation(mat, MATOP_MULT, (void (*)(void))shellMatMult));
  PetscCallAbort(PETSC_COMM_SELF,
                 MatShellSetOperation(mat, MATOP_GET_DIAGONAL, (void (*)(void))shellGetDiagonal));

  KSP ksp;
  PetscCallAbort(PETSC_COMM_SELF, KSPCreate(PETSC_COMM_SELF, &ksp));
  PetscCallAbort(PETSC_COMM_SELF, KSPSetOperators(ksp, mat, mat));
  PetscCallAbort(PETSC_COMM_SELF, KSPSetType(ksp, KSPCG));
  PetscCallAbort(PETSC_COMM_SELF, KSPSetNormType(ksp, KSP_NORM_UNPRECONDITIONED));
  PetscCallAbort(PETSC_COMM_SELF, KSPSetTolerances(ksp, rel_tol, 0.0, PETSC_DEFAULT, n * n));

  PC pc;
  PetscCallAbort(PETSC_COMM_SELF, KSPGetPC(ksp, &pc));
  PetscCallAbort(PETSC_COMM_SELF, PCSetType(pc, PCJACOBI));

  return ksp;
}

} // namespace

TEST(PetscTorchMatrixFreeBenchmarkTest, PetscLegZeroCopyVecMatchesTorchNativeLeg)
{
  const int64_t n = 16;
  PeriodicStencilOperator A(n, shift);
  ShellContext ctx{&A, n, 4.0 + shift};

  const auto x_exact = torch::randn({n * n}, torch::kDouble);
  auto b_tensor = A(x_exact);
  // b_tensor must stay alive and contiguous for the lifetime of the wrapped
  // Vec: VecCreateSeqWithArray does not take ownership or copy the buffer.
  b_tensor = b_tensor.contiguous();

  Vec b_vec, x_vec;
  PetscCallAbort(
      PETSC_COMM_SELF,
      VecCreateSeqWithArray(PETSC_COMM_SELF, 1, n * n, b_tensor.data_ptr<double>(), &b_vec));

  auto x_tensor = torch::zeros({n * n}, torch::kDouble);
  PetscCallAbort(
      PETSC_COMM_SELF,
      VecCreateSeqWithArray(PETSC_COMM_SELF, 1, n * n, x_tensor.data_ptr<double>(), &x_vec));

  // Zero-copy check: PETSc's Vec must resolve to the same memory as the
  // tensor's own buffer, not a copy (plan Sec. 2.0a) - checked directly, not
  // inferred from timing, per PR3's existing verification requirement.
  PetscScalar * vec_array;
  PetscCallAbort(PETSC_COMM_SELF, VecGetArray(x_vec, &vec_array));
  EXPECT_EQ(static_cast<void *>(vec_array), static_cast<void *>(x_tensor.data_ptr<double>()));
  PetscCallAbort(PETSC_COMM_SELF, VecRestoreArray(x_vec, &vec_array));

  KSP ksp = buildShellKSP(A, ctx, n);
  PetscCallAbort(PETSC_COMM_SELF, KSPSolve(ksp, b_vec, x_vec));

  PetscInt petsc_iterations;
  PetscCallAbort(PETSC_COMM_SELF, KSPGetIterationNumber(ksp, &petsc_iterations));

  // x_vec aliases x_tensor's buffer, so the solution is already in x_tensor.
  EXPECT_NEAR((x_tensor - x_exact).norm().item<double>(), 0.0, 1e-6);

  const auto [x_torch, torch_iterations, torch_norm] = solveTorchNative(n, b_tensor);
  EXPECT_NEAR((x_tensor - x_torch).norm().item<double>(), 0.0, 1e-6);
  // Both legs use the identical unpreconditioned stopping test and the same
  // Jacobi-equivalent preconditioning, so their iteration counts should
  // match closely (not necessarily bit-identically, since PETSc's internal
  // CG recurrence and MooseTensor's differ in implementation detail).
  EXPECT_NEAR(static_cast<double>(petsc_iterations), static_cast<double>(torch_iterations), 2.0);
  (void)torch_norm;

  PetscCallAbort(PETSC_COMM_SELF, KSPDestroy(&ksp));
  PetscCallAbort(PETSC_COMM_SELF, VecDestroy(&b_vec));
  PetscCallAbort(PETSC_COMM_SELF, VecDestroy(&x_vec));
}

TEST(PetscTorchMatrixFreeBenchmarkTest, PetscLegNaiveCopyVecMatchesZeroCopyLeg)
{
  const int64_t n = 16;
  PeriodicStencilOperator A(n, shift);
  ShellContext ctx{&A, n, 4.0 + shift};

  const auto x_exact = torch::randn({n * n}, torch::kDouble);
  const auto b_tensor = A(x_exact).contiguous();

  // Naive copy: allocate a separate PETSc-owned Vec and copy values in,
  // rather than wrapping the tensor's buffer. Comparing this leg's result
  // against the zero-copy leg isolates whether the hand-off itself (not
  // just its cost) changes anything.
  Vec b_vec, x_vec;
  PetscCallAbort(PETSC_COMM_SELF, VecCreate(PETSC_COMM_SELF, &b_vec));
  PetscCallAbort(PETSC_COMM_SELF, VecSetSizes(b_vec, PETSC_DECIDE, n * n));
  PetscCallAbort(PETSC_COMM_SELF, VecSetFromOptions(b_vec));
  PetscCallAbort(PETSC_COMM_SELF, VecDuplicate(b_vec, &x_vec));

  PetscScalar * b_arr;
  PetscCallAbort(PETSC_COMM_SELF, VecGetArray(b_vec, &b_arr));
  std::memcpy(b_arr, b_tensor.data_ptr<double>(), n * n * sizeof(double));
  PetscCallAbort(PETSC_COMM_SELF, VecRestoreArray(b_vec, &b_arr));
  PetscCallAbort(PETSC_COMM_SELF, VecZeroEntries(x_vec));

  KSP ksp = buildShellKSP(A, ctx, n);
  PetscCallAbort(PETSC_COMM_SELF, KSPSolve(ksp, b_vec, x_vec));

  const PetscScalar * x_arr;
  PetscCallAbort(PETSC_COMM_SELF, VecGetArrayRead(x_vec, &x_arr));
  auto x_tensor = torch::from_blob(const_cast<PetscScalar *>(x_arr), {n * n}, torch::kDouble);
  EXPECT_NEAR((x_tensor - x_exact).norm().item<double>(), 0.0, 1e-6);
  PetscCallAbort(PETSC_COMM_SELF, VecRestoreArrayRead(x_vec, &x_arr));

  PetscCallAbort(PETSC_COMM_SELF, KSPDestroy(&ksp));
  PetscCallAbort(PETSC_COMM_SELF, VecDestroy(&b_vec));
  PetscCallAbort(PETSC_COMM_SELF, VecDestroy(&x_vec));
}

// SECTION 4: DISABLED_ timing sweep - wall-clock/iteration-count comparison,
// run only on request (--gtest_also_run_disabled_tests), per the plan's
// "too slow/noisy for every CI run" decision. DISABLED_ (rather than #if 0,
// as in the existing unit/src/BenchmarkParsedTensors.C) keeps this code
// compiling and type-checked on every build.
TEST(PetscTorchMatrixFreeBenchmarkTest, DISABLED_TimingSweep)
{
  using std::chrono::duration;
  using std::chrono::high_resolution_clock;

  for (const int64_t n : {32, 64, 128, 256, 512, 1024, 2048})
  {
    PeriodicStencilOperator A(n, shift);
    ShellContext ctx{&A, n, 4.0 + shift};
    const auto x_exact = torch::randn({n * n}, torch::kDouble);
    const auto b_tensor = A(x_exact).contiguous();

    // torch-native leg timing
    auto t0 = high_resolution_clock::now();
    const auto [x_torch, torch_it, torch_norm] = solveTorchNative(n, b_tensor);
    auto t1 = high_resolution_clock::now();
    (void)x_torch;
    (void)torch_norm;
    const duration<double, std::milli> torch_ms = t1 - t0;

    // PETSc leg timing, zero-copy Vec hand-off included in the timed region
    // (the hand-off cost is part of what PR1a measures, per the plan).
    auto t2 = high_resolution_clock::now();
    auto x_tensor = torch::zeros({n * n}, torch::kDouble);
    Vec b_vec, x_vec;
    PetscCallAbort(
        PETSC_COMM_SELF,
        VecCreateSeqWithArray(
            PETSC_COMM_SELF, 1, n * n, const_cast<double *>(b_tensor.data_ptr<double>()), &b_vec));
    PetscCallAbort(
        PETSC_COMM_SELF,
        VecCreateSeqWithArray(PETSC_COMM_SELF, 1, n * n, x_tensor.data_ptr<double>(), &x_vec));
    KSP ksp = buildShellKSP(A, ctx, n);
    PetscCallAbort(PETSC_COMM_SELF, KSPSolve(ksp, b_vec, x_vec));
    auto t3 = high_resolution_clock::now();
    const duration<double, std::milli> petsc_zero_copy_ms = t3 - t2;

    PetscInt petsc_it;
    PetscCallAbort(PETSC_COMM_SELF, KSPGetIterationNumber(ksp, &petsc_it));

    Moose::out << "n=" << n << " torch-native: " << torch_ms.count() << "ms, " << torch_it
               << " iters | PETSc (zero-copy Vec): " << petsc_zero_copy_ms.count() << "ms, "
               << petsc_it << " iters | peak RSS so far: " << peakRssKB() << " KB\n";

    PetscCallAbort(PETSC_COMM_SELF, KSPDestroy(&ksp));
    PetscCallAbort(PETSC_COMM_SELF, VecDestroy(&b_vec));
    PetscCallAbort(PETSC_COMM_SELF, VecDestroy(&x_vec));
  }
}

// SECTION 5: ill-conditioned (anisotropic) comparison - demonstrates R7: on a
// stencil severe enough that a direction-blind preconditioner (Jacobi, used
// by both legs elsewhere in this file) converges slowly, PETSc's access to
// algebraic multigrid (AMG) built from a separately assembled Pmat (PR4a's
// mixed mode, plan Sec. 2.0a: matrix-free Amat + assembled Pmat) converges in
// far fewer iterations. Pmat's assembly here is a one-off direct
// MatSetValue loop, not a production stencilAt()-based path - that is PR4a's
// job once PetscFDSolverBase (PR3) exists.
namespace
{

// Anisotropic diffusivity: Dx >> Dy models the strongly direction-dependent
// mobility tensors R7 is concerned about (e.g. KKS/grand-potential). The
// resulting matrix has a condition number that grows with Dx/Dy, which is
// what defeats a direction-blind (scalar) Jacobi preconditioner.
constexpr double aniso_Dx = 1.0;
constexpr double aniso_Dy = 1.0e-4;
constexpr double aniso_shift = 1.0e-3;

/// Applies -div(D grad(u)) + shift*u on a periodic nxn grid (unit spacing)
/// with anisotropic, direction-dependent diffusivity (Dx along axis 1, Dy
/// along axis 0 of the row-major grid). Same circular-padding/conv2d
/// construction as PeriodicStencilOperator, just a non-uniform kernel.
class AnisotropicStencilOperator
{
public:
  AnisotropicStencilOperator(
      int64_t n, double Dx, double Dy, double shift, torch::Device device = torch::kCPU)
    : _n(n), _Dx(Dx), _Dy(Dy), _shift(shift)
  {
    _kernel =
        torch::zeros({1, 1, 3, 3}, torch::TensorOptions().dtype(torch::kDouble).device(device));
    _kernel[0][0][1][1] = 2.0 * (_Dx + _Dy);
    _kernel[0][0][0][1] = -_Dy;
    _kernel[0][0][2][1] = -_Dy;
    _kernel[0][0][1][0] = -_Dx;
    _kernel[0][0][1][2] = -_Dx;
  }

  torch::Tensor operator()(const torch::Tensor & x) const
  {
    auto grid = x.view({1, 1, _n, _n});
    auto padded = torch::nn::functional::pad(
        grid, torch::nn::functional::PadFuncOptions({1, 1, 1, 1}).mode(torch::kCircular));
    auto lap_term = torch::nn::functional::conv2d(padded, _kernel);
    return (lap_term.view({_n * _n}) + _shift * x);
  }

  int64_t n() const { return _n; }
  double diagValue() const { return 2.0 * (_Dx + _Dy) + _shift; }

private:
  int64_t _n;
  double _Dx, _Dy, _shift;
  torch::Tensor _kernel;
};

struct AnisoShellContext
{
  const AnisotropicStencilOperator * op;
  int64_t n;
  double diag_value;
};

PetscErrorCode
anisoShellMatMult(Mat mat, Vec x, Vec y)
{
  AnisoShellContext * ctx;
  PetscCallAbort(PETSC_COMM_SELF, MatShellGetContext(mat, &ctx));

  const PetscScalar * x_arr;
  PetscScalar * y_arr;
  PetscCallAbort(PETSC_COMM_SELF, VecGetArrayRead(x, &x_arr));
  PetscCallAbort(PETSC_COMM_SELF, VecGetArray(y, &y_arr));

  auto x_tensor =
      torch::from_blob(const_cast<PetscScalar *>(x_arr), {ctx->n * ctx->n}, torch::kDouble);
  auto result = (*ctx->op)(x_tensor);
  std::memcpy(y_arr, result.data_ptr<double>(), ctx->n * ctx->n * sizeof(double));

  PetscCallAbort(PETSC_COMM_SELF, VecRestoreArrayRead(x, &x_arr));
  PetscCallAbort(PETSC_COMM_SELF, VecRestoreArray(y, &y_arr));
  return PETSC_SUCCESS;
}

PetscErrorCode
anisoShellGetDiagonal(Mat mat, Vec diag)
{
  AnisoShellContext * ctx;
  PetscCallAbort(PETSC_COMM_SELF, MatShellGetContext(mat, &ctx));
  PetscCallAbort(PETSC_COMM_SELF, VecSet(diag, ctx->diag_value));
  return PETSC_SUCCESS;
}

/// Assembles the real sparse 5-point matrix for AnisotropicStencilOperator
/// on an nxn periodic grid, row-major flat indexing (row = i*n+j). This is
/// PR4a's Pmat role (feeds AMG) when used as the preconditioning matrix
/// alongside a matrix-free Amat, and also doubles as this test's fully
/// assembled Amat for the "regular PETSc" comparison leg (Section 6): the
/// matrix is identical either way, only how it's used (Pmat-only vs.
/// Amat-and-Pmat) differs.
Mat
assembleAnisotropicMat(int64_t n, double Dx, double Dy, double shift)
{
  Mat mat;
  PetscCallAbort(PETSC_COMM_SELF, MatCreate(PETSC_COMM_SELF, &mat));
  PetscCallAbort(PETSC_COMM_SELF, MatSetSizes(mat, n * n, n * n, n * n, n * n));
  PetscCallAbort(PETSC_COMM_SELF, MatSetFromOptions(mat));
  // Preallocate the known 5-nonzeros-per-row sparsity pattern. Without this,
  // MatSetValue's unpreallocated insertion path is far slower (effectively
  // O(nnz^2)) and dominates assembly time at the larger grid sizes Section 6
  // sweeps over.
  PetscCallAbort(PETSC_COMM_SELF, MatSeqAIJSetPreallocation(mat, 5, nullptr));
  PetscCallAbort(PETSC_COMM_SELF, MatSetUp(mat));

  const double diag = 2.0 * (Dx + Dy) + shift;
  for (const auto i : libMesh::make_range(n))
    for (const auto j : libMesh::make_range(n))
    {
      const PetscInt row = static_cast<PetscInt>(i * n + j);
      const PetscInt col_up = static_cast<PetscInt>(((i + n - 1) % n) * n + j);
      const PetscInt col_down = static_cast<PetscInt>(((i + 1) % n) * n + j);
      const PetscInt col_left = static_cast<PetscInt>(i * n + (j + n - 1) % n);
      const PetscInt col_right = static_cast<PetscInt>(i * n + (j + 1) % n);

      PetscCallAbort(PETSC_COMM_SELF, MatSetValue(mat, row, row, diag, INSERT_VALUES));
      PetscCallAbort(PETSC_COMM_SELF, MatSetValue(mat, row, col_up, -Dy, INSERT_VALUES));
      PetscCallAbort(PETSC_COMM_SELF, MatSetValue(mat, row, col_down, -Dy, INSERT_VALUES));
      PetscCallAbort(PETSC_COMM_SELF, MatSetValue(mat, row, col_left, -Dx, INSERT_VALUES));
      PetscCallAbort(PETSC_COMM_SELF, MatSetValue(mat, row, col_right, -Dx, INSERT_VALUES));
    }
  PetscCallAbort(PETSC_COMM_SELF, MatAssemblyBegin(mat, MAT_FINAL_ASSEMBLY));
  PetscCallAbort(PETSC_COMM_SELF, MatAssemblyEnd(mat, MAT_FINAL_ASSEMBLY));
  return mat;
}

} // namespace

TEST(PetscTorchMatrixFreeBenchmarkTest, AnisotropicAMGConvergesFasterThanJacobi)
{
  using std::chrono::duration;
  using std::chrono::high_resolution_clock;

  const int64_t n = 64;
  AnisotropicStencilOperator A(n, aniso_Dx, aniso_Dy, aniso_shift);
  AnisoShellContext ctx{&A, n, A.diagValue()};

  const auto x_exact = torch::randn({n * n}, torch::kDouble);
  auto b_tensor = A(x_exact).contiguous();

  Mat amat;
  PetscCallAbort(PETSC_COMM_SELF,
                 MatCreateShell(PETSC_COMM_SELF, n * n, n * n, n * n, n * n, &ctx, &amat));
  PetscCallAbort(PETSC_COMM_SELF,
                 MatShellSetOperation(amat, MATOP_MULT, (void (*)(void))anisoShellMatMult));
  PetscCallAbort(
      PETSC_COMM_SELF,
      MatShellSetOperation(amat, MATOP_GET_DIAGONAL, (void (*)(void))anisoShellGetDiagonal));

  // Returns (iterations, solve wall-clock ms). Fewer AMG iterations is only
  // a genuine win if the wall-clock time is also lower - GAMG's per-iteration
  // setup/apply cost is higher than Jacobi's, so the iteration-count
  // comparison alone (as this test originally only reported) doesn't say
  // which leg is actually faster.
  auto run = [&](PCType pc_type, Mat pmat) -> std::pair<PetscInt, double>
  {
    auto x_tensor = torch::zeros({n * n}, torch::kDouble);
    Vec b_vec, x_vec;
    PetscCallAbort(
        PETSC_COMM_SELF,
        VecCreateSeqWithArray(PETSC_COMM_SELF, 1, n * n, b_tensor.data_ptr<double>(), &b_vec));
    PetscCallAbort(
        PETSC_COMM_SELF,
        VecCreateSeqWithArray(PETSC_COMM_SELF, 1, n * n, x_tensor.data_ptr<double>(), &x_vec));

    KSP ksp;
    PetscCallAbort(PETSC_COMM_SELF, KSPCreate(PETSC_COMM_SELF, &ksp));
    PetscCallAbort(PETSC_COMM_SELF, KSPSetOperators(ksp, amat, pmat));
    PetscCallAbort(PETSC_COMM_SELF, KSPSetType(ksp, KSPCG));
    PetscCallAbort(PETSC_COMM_SELF, KSPSetNormType(ksp, KSP_NORM_UNPRECONDITIONED));
    PetscCallAbort(PETSC_COMM_SELF, KSPSetTolerances(ksp, rel_tol, 0.0, PETSC_DEFAULT, 10 * n * n));

    PC pc;
    PetscCallAbort(PETSC_COMM_SELF, KSPGetPC(ksp, &pc));
    PetscCallAbort(PETSC_COMM_SELF, PCSetType(pc, pc_type));

    // Timed region covers KSPSolve only, matching this file's other timing
    // sweeps (Sections 4/6/7) - PC/KSP setup above is configuration, not
    // part of the per-solve cost being compared.
    const auto t0 = high_resolution_clock::now();
    PetscCallAbort(PETSC_COMM_SELF, KSPSolve(ksp, b_vec, x_vec));
    const auto t1 = high_resolution_clock::now();
    const duration<double, std::milli> solve_ms = t1 - t0;

    PetscInt its;
    PetscCallAbort(PETSC_COMM_SELF, KSPGetIterationNumber(ksp, &its));

    KSPConvergedReason reason;
    PetscCallAbort(PETSC_COMM_SELF, KSPGetConvergedReason(ksp, &reason));
    EXPECT_GT(reason, 0) << "KSP (" << pc_type << ") failed to converge";
    EXPECT_NEAR((x_tensor - x_exact).norm().item<double>(), 0.0, 1e-5);

    PetscCallAbort(PETSC_COMM_SELF, KSPDestroy(&ksp));
    PetscCallAbort(PETSC_COMM_SELF, VecDestroy(&b_vec));
    PetscCallAbort(PETSC_COMM_SELF, VecDestroy(&x_vec));
    return {its, solve_ms.count()};
  };

  // Jacobi-preconditioned leg: pmat == amat (PCJACOBI only needs the
  // diagonal, which the shell's MatGetDiagonal callback already supplies).
  const auto [jacobi_its, jacobi_ms] = run(PCJACOBI, amat);

  // AMG-preconditioned leg: a real assembled Pmat built from the same
  // coefficients, matrix-free Amat unchanged (PR4a's mixed mode). Pmat
  // assembly is timed separately from the solve, since it is a one-time
  // setup cost (Section~sec:decisions item 3's reassembly cadence), not
  // part of each KSPSolve.
  const auto t_asm0 = high_resolution_clock::now();
  Mat pmat = assembleAnisotropicMat(n, aniso_Dx, aniso_Dy, aniso_shift);
  const auto t_asm1 = high_resolution_clock::now();
  const duration<double, std::milli> pmat_assembly_ms = t_asm1 - t_asm0;
  const auto [amg_its, amg_ms] = run(PCGAMG, pmat);

  // The whole point of R7/PR4a: AMG built from the real coefficients should
  // need substantially fewer iterations than direction-blind Jacobi once the
  // anisotropy ratio (Dx/Dy here: 1e4) is severe enough to defeat it.
  EXPECT_LT(amg_its, jacobi_its);
  Moose::out << "anisotropic (Dx/Dy=" << (aniso_Dx / aniso_Dy) << "): Jacobi " << jacobi_its
             << " its, " << jacobi_ms << "ms | AMG (PCGAMG) " << amg_its << " its, " << amg_ms
             << "ms solve + " << pmat_assembly_ms.count() << "ms Pmat assembly\n";

  PetscCallAbort(PETSC_COMM_SELF, MatDestroy(&pmat));
  PetscCallAbort(PETSC_COMM_SELF, MatDestroy(&amat));
}

// SECTION 6: DISABLED_ scaling/memory sweep for the fully assembled Mat
// fallback (plan Sec. 2.0a/decision 5: "a fully assembled Amat is kept only
// as a documented last-resort fallback"). This leg uses assembleAnisotropicMat
// with Dx=Dy=1 (reproducing PeriodicStencilOperator's own matrix exactly) as
// a real, non-shell Amat, and times/measures peak RSS against the same
// problem's matrix-free torch-native and PETSc-shell legs (Section 4) at
// matching grid sizes. The assembled matrix's own storage (5 nonzeros/row,
// 8 bytes + 4 bytes index per nonzero in PETSc's default AIJ format) grows
// linearly in n^2, unlike the matrix-free legs' O(1) extra memory per field
// - this test demonstrates that difference directly rather than inferring it.
TEST(PetscTorchMatrixFreeBenchmarkTest, DISABLED_AssembledMatScalingSweep)
{
  using std::chrono::duration;
  using std::chrono::high_resolution_clock;

  for (const int64_t n : {32, 64, 128, 256, 512, 1024, 2048})
  {
    const auto x_exact = torch::randn({n * n}, torch::kDouble);
    PeriodicStencilOperator A(n, shift);
    const auto b_tensor = A(x_exact).contiguous();

    auto t0 = high_resolution_clock::now();
    Mat amat = assembleAnisotropicMat(n, 1.0, 1.0, shift);
    auto t1 = high_resolution_clock::now();
    const duration<double, std::milli> assembly_ms = t1 - t0;

    auto x_tensor = torch::zeros({n * n}, torch::kDouble);
    Vec b_vec, x_vec;
    PetscCallAbort(
        PETSC_COMM_SELF,
        VecCreateSeqWithArray(PETSC_COMM_SELF, 1, n * n, b_tensor.data_ptr<double>(), &b_vec));
    PetscCallAbort(
        PETSC_COMM_SELF,
        VecCreateSeqWithArray(PETSC_COMM_SELF, 1, n * n, x_tensor.data_ptr<double>(), &x_vec));

    KSP ksp;
    PetscCallAbort(PETSC_COMM_SELF, KSPCreate(PETSC_COMM_SELF, &ksp));
    PetscCallAbort(PETSC_COMM_SELF, KSPSetOperators(ksp, amat, amat));
    PetscCallAbort(PETSC_COMM_SELF, KSPSetType(ksp, KSPCG));
    PetscCallAbort(PETSC_COMM_SELF, KSPSetNormType(ksp, KSP_NORM_UNPRECONDITIONED));
    PetscCallAbort(PETSC_COMM_SELF, KSPSetTolerances(ksp, rel_tol, 0.0, PETSC_DEFAULT, n * n));
    PC pc;
    PetscCallAbort(PETSC_COMM_SELF, KSPGetPC(ksp, &pc));
    PetscCallAbort(PETSC_COMM_SELF, PCSetType(pc, PCJACOBI));

    auto t2 = high_resolution_clock::now();
    PetscCallAbort(PETSC_COMM_SELF, KSPSolve(ksp, b_vec, x_vec));
    auto t3 = high_resolution_clock::now();
    const duration<double, std::milli> solve_ms = t3 - t2;

    PetscInt its;
    PetscCallAbort(PETSC_COMM_SELF, KSPGetIterationNumber(ksp, &its));

    MatInfo info;
    PetscCallAbort(PETSC_COMM_SELF, MatGetInfo(amat, MAT_LOCAL, &info));

    Moose::out << "n=" << n << " assembled-Mat: assembly " << assembly_ms.count() << "ms, solve "
               << solve_ms.count() << "ms, " << its
               << " iters, Mat nz_used=" << static_cast<long>(info.nz_used) << " (~"
               << (info.nz_used * (sizeof(PetscScalar) + sizeof(PetscInt)) / 1024.0)
               << " KB in Mat storage alone) | peak RSS so far: " << peakRssKB() << " KB\n";

    PetscCallAbort(PETSC_COMM_SELF, KSPDestroy(&ksp));
    PetscCallAbort(PETSC_COMM_SELF, VecDestroy(&b_vec));
    PetscCallAbort(PETSC_COMM_SELF, VecDestroy(&x_vec));
    PetscCallAbort(PETSC_COMM_SELF, MatDestroy(&amat));
  }
}

// SECTION 7: CUDA leg - zero-copy PETSc Vec over a CUDA tensor's device
// pointer, exercised only when both PETSc (PetscDefined(HAVE_CUDA)) and
// libtorch (MARLIN_TORCH_CUDA_AVAILABLE, Sec. 0 above) were built with CUDA
// support, which is NOT the case for either dependency on the development
// machine this file was authored and tested on (confirmed: no
// ATen/cuda/CUDAConfig.h in this machine's libtorch, no CUDA petscvariables
// in this machine's conda PETSc). This section has therefore NOT been
// compiled or run anywhere; it is written from PETSc's documented API
// contracts (manual pages for VecCreateSeqCUDAWithArray, VecCUDAGetArray/
// VecCUDARestoreArray, MatShellSetVecType, PetscGetCurrentCUDAStream) and
// needs verification on an actual CUDA-enabled build before it can be
// trusted. See the plan's "Running on CUDA" instructions for how to get
// that build and report back a reviewable log.
//
// Design, to avoid wasting memory or adding synchronization overhead beyond
// what PETSc's own Vec/KSP machinery already does:
//  - b/x Vecs are built via VecCreateSeqCUDAWithArray wrapping the CUDA
//    tensor's own data_ptr() directly (zero-copy: no host round-trip, no
//    separate PETSc-owned device allocation), mirroring the CPU zero-copy
//    leg's VecCreateSeqWithArray in Section 3.
//  - The shell's MatMult callback uses VecCUDAGetArray/VecCUDARestoreArray
//    (not VecGetArray/VecRestoreArray): these hand back the device pointer
//    directly with no implicit device<->host copy, whereas VecGetArray on a
//    CUDA Vec would force a device-to-host sync every iteration - exactly
//    the per-iteration overhead this design avoids.
//  - MatShellSetVecType(mat, VECSEQCUDA) tells KSP to allocate its own
//    internal CG work vectors (search direction, residual, etc., which the
//    user-level code never sees) as CUDA Vecs too. Without this, KSP
//    defaults those to the Vec type of the Amat/Pmat arguments' *container*
//    type, and a MatCreateShell's default is VECSTANDARD (host), which would
//    silently force a device<->host copy on every KSP-internal vector
//    operation even though the user-supplied b/x Vecs are on-device.
//  - torch's own CUDA kernels, invoked from inside the MatMult callback via
//    the identical stencil operator functor used by the CPU legs, run on
//    PETSc's current CUDA stream (via PetscGetCurrentCUDAStream +
//    c10::cuda::CUDAStreamGuard) rather than torch's own default stream.
//    Without this, torch's kernel launches and PETSc's own CUDA kernels
//    (dot products, AXPY, norms inside KSPSolve) would run on two unrelated
//    streams with no implicit ordering between them, forcing either a race
//    or a cudaDeviceSynchronize() call every iteration to serialize them -
//    exactly the overhead this design is written to avoid. Running torch on
//    PETSc's own stream keeps everything in one stream's natural ordering
//    with no explicit sync needed.
#if MARLIN_PETSC_TORCH_CUDA_LEG

namespace
{

/// RAII guard making torch's CUDA kernels (for the duration of the guard's
/// lifetime) run on PETSc's current CUDA stream, so a MatMult callback's
/// torch:: calls interleave correctly with PETSc's own CUDA work (dot
/// products, AXPY, norms in KSPSolve) without an explicit
/// cudaDeviceSynchronize() every iteration.
class PetscStreamGuard
{
public:
  PetscStreamGuard()
  {
    cudaStream_t petsc_stream;
    PetscCallAbort(PETSC_COMM_SELF, PetscGetCurrentCUDAStream(&petsc_stream));
    const auto device_index = c10::cuda::current_device();
    _guard = std::make_unique<c10::cuda::CUDAStreamGuard>(
        c10::cuda::getStreamFromExternal(petsc_stream, device_index));
  }

private:
  std::unique_ptr<c10::cuda::CUDAStreamGuard> _guard;
};

PetscErrorCode
cudaShellMatMult(Mat mat, Vec x, Vec y)
{
  ShellContext * ctx;
  PetscCallAbort(PETSC_COMM_SELF, MatShellGetContext(mat, &ctx));

  PetscStreamGuard stream_guard;

  // VecCUDAGetArray/VecCUDARestoreArray hand back the device pointer with no
  // implicit device<->host copy - unlike VecGetArray on a CUDA Vec, which
  // would force one. x's device data is read, not modified, by the stencil
  // action, but there is no VecCUDAGetArrayRead counterpart used here
  // because PETSc's const-correctness for the read-only overload still
  // requires the same device-resident buffer; VecCUDAGetArray is used
  // symmetrically for both x and y for clarity, matching the CPU leg's
  // VecGetArrayRead/VecGetArray split in spirit, not in the exact overload.
  const PetscScalar * x_arr;
  PetscScalar * y_arr;
  PetscCallAbort(PETSC_COMM_SELF, VecCUDAGetArrayRead(x, &x_arr));
  PetscCallAbort(PETSC_COMM_SELF, VecCUDAGetArrayWrite(y, &y_arr));

  auto x_tensor = torch::from_blob(const_cast<PetscScalar *>(x_arr),
                                   {ctx->n * ctx->n},
                                   torch::TensorOptions()
                                       .dtype(torch::kDouble)
                                       .device(torch::kCUDA, c10::cuda::current_device()));
  auto result = (*ctx->op)(x_tensor);
  // result is a torch-allocated CUDA tensor (the stencil op's conv2d output,
  // not an alias of x_arr/y_arr), so this device-to-device copy is required
  // regardless of the Vec hand-off being zero-copy; it is a cudaMemcpyAsync
  // on PETSc's stream (via the guard above), not a host round-trip.
  cudaMemcpyAsync(y_arr,
                  result.data_ptr<double>(),
                  ctx->n * ctx->n * sizeof(double),
                  cudaMemcpyDeviceToDevice,
                  c10::cuda::getCurrentCUDAStream().stream());

  PetscCallAbort(PETSC_COMM_SELF, VecCUDARestoreArrayRead(x, &x_arr));
  PetscCallAbort(PETSC_COMM_SELF, VecCUDARestoreArrayWrite(y, &y_arr));
  return PETSC_SUCCESS;
}

PetscErrorCode
cudaShellGetDiagonal(Mat mat, Vec diag)
{
  ShellContext * ctx;
  PetscCallAbort(PETSC_COMM_SELF, MatShellGetContext(mat, &ctx));
  PetscCallAbort(PETSC_COMM_SELF, VecSet(diag, ctx->diag_value));
  return PETSC_SUCCESS;
}

/// CUDA counterparts of Section 5's anisoShellMatMult/anisoShellGetDiagonal,
/// identical in structure to cudaShellMatMult/cudaShellGetDiagonal above but
/// operating on AnisoShellContext (Section 5) instead of ShellContext.
PetscErrorCode
cudaAnisoShellMatMult(Mat mat, Vec x, Vec y)
{
  AnisoShellContext * ctx;
  PetscCallAbort(PETSC_COMM_SELF, MatShellGetContext(mat, &ctx));

  PetscStreamGuard stream_guard;

  const PetscScalar * x_arr;
  PetscScalar * y_arr;
  PetscCallAbort(PETSC_COMM_SELF, VecCUDAGetArrayRead(x, &x_arr));
  PetscCallAbort(PETSC_COMM_SELF, VecCUDAGetArrayWrite(y, &y_arr));

  auto x_tensor = torch::from_blob(const_cast<PetscScalar *>(x_arr),
                                   {ctx->n * ctx->n},
                                   torch::TensorOptions()
                                       .dtype(torch::kDouble)
                                       .device(torch::kCUDA, c10::cuda::current_device()));
  auto result = (*ctx->op)(x_tensor);
  cudaMemcpyAsync(y_arr,
                  result.data_ptr<double>(),
                  ctx->n * ctx->n * sizeof(double),
                  cudaMemcpyDeviceToDevice,
                  c10::cuda::getCurrentCUDAStream().stream());

  PetscCallAbort(PETSC_COMM_SELF, VecCUDARestoreArrayRead(x, &x_arr));
  PetscCallAbort(PETSC_COMM_SELF, VecCUDARestoreArrayWrite(y, &y_arr));
  return PETSC_SUCCESS;
}

PetscErrorCode
cudaAnisoShellGetDiagonal(Mat mat, Vec diag)
{
  AnisoShellContext * ctx;
  PetscCallAbort(PETSC_COMM_SELF, MatShellGetContext(mat, &ctx));
  PetscCallAbort(PETSC_COMM_SELF, VecSet(diag, ctx->diag_value));
  return PETSC_SUCCESS;
}

} // namespace

TEST(PetscTorchMatrixFreeBenchmarkTest, CudaLegZeroCopyVecMatchesTorchNativeLeg)
{
  ASSERT_TRUE(torch::cuda::is_available())
      << "PETSC_HAVE_CUDA and libtorch CUDA headers are both present at compile time, but no "
         "CUDA device is available at runtime on this machine.";

  const int64_t n = 16;
  const auto cuda_opts = torch::TensorOptions().dtype(torch::kDouble).device(torch::kCUDA);

  PeriodicStencilOperator A(n, shift, torch::kCUDA);
  ShellContext ctx{&A, n, 4.0 + shift};

  // x_exact is manufactured on CPU for a reproducible, device-independent
  // random draw, then moved to CUDA; A (whose kernel now lives on CUDA, for
  // the shell's MatMult callback below) is applied to it there to get b.
  const auto x_exact_cpu = torch::randn({n * n}, torch::kDouble);
  const auto x_exact = x_exact_cpu.to(cuda_opts);
  auto b_tensor = A(x_exact).contiguous();

  Mat mat;
  PetscCallAbort(PETSC_COMM_SELF,
                 MatCreateShell(PETSC_COMM_SELF, n * n, n * n, n * n, n * n, &ctx, &mat));
  PetscCallAbort(PETSC_COMM_SELF,
                 MatShellSetOperation(mat, MATOP_MULT, (void (*)(void))cudaShellMatMult));
  PetscCallAbort(
      PETSC_COMM_SELF,
      MatShellSetOperation(mat, MATOP_GET_DIAGONAL, (void (*)(void))cudaShellGetDiagonal));
  // Without this, KSP's own internally allocated work vectors (search
  // direction, residual, etc.) default to a host Vec type, forcing a
  // device<->host copy on every KSP-internal vector op even though the
  // user-supplied b/x Vecs below are on-device.
  PetscCallAbort(PETSC_COMM_SELF, MatShellSetVecType(mat, VECSEQCUDA));

  Vec b_vec, x_vec;
  PetscCallAbort(
      PETSC_COMM_SELF,
      VecCreateSeqCUDAWithArray(PETSC_COMM_SELF, 1, n * n, b_tensor.data_ptr<double>(), &b_vec));

  auto x_tensor = torch::zeros({n * n}, cuda_opts);
  PetscCallAbort(
      PETSC_COMM_SELF,
      VecCreateSeqCUDAWithArray(PETSC_COMM_SELF, 1, n * n, x_tensor.data_ptr<double>(), &x_vec));

  // Zero-copy check: PETSc's Vec must resolve to the same device memory as
  // the tensor's own buffer, not a copy - the CUDA analog of the CPU leg's
  // VecGetArray pointer-equality check in Section 3.
  PetscScalar * vec_array;
  PetscCallAbort(PETSC_COMM_SELF, VecCUDAGetArray(x_vec, &vec_array));
  EXPECT_EQ(static_cast<void *>(vec_array), static_cast<void *>(x_tensor.data_ptr<double>()));
  PetscCallAbort(PETSC_COMM_SELF, VecCUDARestoreArray(x_vec, &vec_array));

  KSP ksp;
  PetscCallAbort(PETSC_COMM_SELF, KSPCreate(PETSC_COMM_SELF, &ksp));
  PetscCallAbort(PETSC_COMM_SELF, KSPSetOperators(ksp, mat, mat));
  PetscCallAbort(PETSC_COMM_SELF, KSPSetType(ksp, KSPCG));
  PetscCallAbort(PETSC_COMM_SELF, KSPSetNormType(ksp, KSP_NORM_UNPRECONDITIONED));
  PetscCallAbort(PETSC_COMM_SELF, KSPSetTolerances(ksp, rel_tol, 0.0, PETSC_DEFAULT, n * n));
  PC pc;
  PetscCallAbort(PETSC_COMM_SELF, KSPGetPC(ksp, &pc));
  PetscCallAbort(PETSC_COMM_SELF, PCSetType(pc, PCJACOBI));

  PetscCallAbort(PETSC_COMM_SELF, KSPSolve(ksp, b_vec, x_vec));

  PetscInt petsc_iterations;
  PetscCallAbort(PETSC_COMM_SELF, KSPGetIterationNumber(ksp, &petsc_iterations));

  // x_vec aliases x_tensor's device buffer, so the solution is already in
  // x_tensor. .item<double>() below forces one host sync for the test
  // assertion itself (not part of the solve being measured).
  EXPECT_NEAR((x_tensor - x_exact).norm().item<double>(), 0.0, 1e-6);

  const auto [x_torch, torch_iterations, torch_norm] = solveTorchNative(n, b_tensor);
  EXPECT_NEAR((x_tensor - x_torch).norm().item<double>(), 0.0, 1e-6);
  EXPECT_NEAR(static_cast<double>(petsc_iterations), static_cast<double>(torch_iterations), 2.0);
  (void)torch_norm;

  PetscCallAbort(PETSC_COMM_SELF, KSPDestroy(&ksp));
  PetscCallAbort(PETSC_COMM_SELF, VecDestroy(&b_vec));
  PetscCallAbort(PETSC_COMM_SELF, VecDestroy(&x_vec));
  PetscCallAbort(PETSC_COMM_SELF, MatDestroy(&mat));
}

// DISABLED_: same device<->device timing/memory sweep as Section 4's CPU
// DISABLED_TimingSweep, run only on request. peakRssKB() (host RSS) is
// supplemented here with cudaMemGetInfo so device memory, not just host
// memory, is reported - the quantity that actually matters for a GPU
// comparison.
TEST(PetscTorchMatrixFreeBenchmarkTest, DISABLED_CudaTimingSweep)
{
  using std::chrono::duration;
  using std::chrono::high_resolution_clock;

  const auto cuda_opts = torch::TensorOptions().dtype(torch::kDouble).device(torch::kCUDA);

  for (const int64_t n : {32, 64, 128, 256, 512, 1024, 2048})
  {
    PeriodicStencilOperator A(n, shift, torch::kCUDA);
    ShellContext ctx{&A, n, 4.0 + shift};
    const auto x_exact = torch::randn({n * n}, cuda_opts);
    const auto b_tensor = A(x_exact).contiguous();

    cudaDeviceSynchronize();
    auto t0 = high_resolution_clock::now();
    const auto [x_torch, torch_it, torch_norm] = solveTorchNative(n, b_tensor);
    cudaDeviceSynchronize();
    auto t1 = high_resolution_clock::now();
    (void)x_torch;
    (void)torch_norm;
    const duration<double, std::milli> torch_ms = t1 - t0;

    auto x_tensor = torch::zeros({n * n}, cuda_opts);
    Mat mat;
    PetscCallAbort(PETSC_COMM_SELF,
                   MatCreateShell(PETSC_COMM_SELF, n * n, n * n, n * n, n * n, &ctx, &mat));
    PetscCallAbort(PETSC_COMM_SELF,
                   MatShellSetOperation(mat, MATOP_MULT, (void (*)(void))cudaShellMatMult));
    PetscCallAbort(
        PETSC_COMM_SELF,
        MatShellSetOperation(mat, MATOP_GET_DIAGONAL, (void (*)(void))cudaShellGetDiagonal));
    PetscCallAbort(PETSC_COMM_SELF, MatShellSetVecType(mat, VECSEQCUDA));

    Vec b_vec, x_vec;
    PetscCallAbort(
        PETSC_COMM_SELF,
        VecCreateSeqCUDAWithArray(PETSC_COMM_SELF, 1, n * n, b_tensor.data_ptr<double>(), &b_vec));
    PetscCallAbort(
        PETSC_COMM_SELF,
        VecCreateSeqCUDAWithArray(PETSC_COMM_SELF, 1, n * n, x_tensor.data_ptr<double>(), &x_vec));

    KSP ksp;
    PetscCallAbort(PETSC_COMM_SELF, KSPCreate(PETSC_COMM_SELF, &ksp));
    PetscCallAbort(PETSC_COMM_SELF, KSPSetOperators(ksp, mat, mat));
    PetscCallAbort(PETSC_COMM_SELF, KSPSetType(ksp, KSPCG));
    PetscCallAbort(PETSC_COMM_SELF, KSPSetNormType(ksp, KSP_NORM_UNPRECONDITIONED));
    PetscCallAbort(PETSC_COMM_SELF, KSPSetTolerances(ksp, rel_tol, 0.0, PETSC_DEFAULT, n * n));
    PC pc;
    PetscCallAbort(PETSC_COMM_SELF, KSPGetPC(ksp, &pc));
    PetscCallAbort(PETSC_COMM_SELF, PCSetType(pc, PCJACOBI));

    cudaDeviceSynchronize();
    auto t2 = high_resolution_clock::now();
    PetscCallAbort(PETSC_COMM_SELF, KSPSolve(ksp, b_vec, x_vec));
    cudaDeviceSynchronize();
    auto t3 = high_resolution_clock::now();
    const duration<double, std::milli> petsc_zero_copy_ms = t3 - t2;

    PetscInt petsc_it;
    PetscCallAbort(PETSC_COMM_SELF, KSPGetIterationNumber(ksp, &petsc_it));

    size_t free_bytes, total_bytes;
    cudaMemGetInfo(&free_bytes, &total_bytes);
    const double used_mb = (total_bytes - free_bytes) / (1024.0 * 1024.0);

    Moose::out << "n=" << n << " torch-native: " << torch_ms.count() << "ms, " << torch_it
               << " iters | PETSc CUDA (zero-copy Vec): " << petsc_zero_copy_ms.count() << "ms, "
               << petsc_it << " iters | device mem used so far: " << used_mb << " MB"
               << " | host peak RSS so far: " << peakRssKB() << " KB\n";

    PetscCallAbort(PETSC_COMM_SELF, KSPDestroy(&ksp));
    PetscCallAbort(PETSC_COMM_SELF, VecDestroy(&b_vec));
    PetscCallAbort(PETSC_COMM_SELF, VecDestroy(&x_vec));
    PetscCallAbort(PETSC_COMM_SELF, MatDestroy(&mat));
  }
}

// CUDA counterpart of Section 5's AnisotropicAMGConvergesFasterThanJacobi:
// the same ill-conditioned (anisotropic) comparison, with the matrix-free
// Amat's shell callbacks running on CUDA (cudaAnisoShellMatMult/
// cudaAnisoShellGetDiagonal above) and b/x as zero-copy CUDA Vecs, so R7's
// AMG-vs-Jacobi claim is checked on-device rather than only on CPU. Pmat
// (feeding PCGAMG) stays a host-assembled Mat, matching Section 5 and PR4a's
// mixed-mode design (only Amat and the solve Vecs are zero-copy/on-device;
// Pmat's one-off assembly cost is paid on the host either way).
TEST(PetscTorchMatrixFreeBenchmarkTest, AnisotropicCudaAMGConvergesFasterThanJacobi)
{
  ASSERT_TRUE(torch::cuda::is_available())
      << "PETSC_HAVE_CUDA and libtorch CUDA headers are both present at compile time, but no "
         "CUDA device is available at runtime on this machine.";

  const int64_t n = 64;
  const auto cuda_opts = torch::TensorOptions().dtype(torch::kDouble).device(torch::kCUDA);

  AnisotropicStencilOperator A(n, aniso_Dx, aniso_Dy, aniso_shift, torch::kCUDA);
  AnisoShellContext ctx{&A, n, A.diagValue()};

  const auto x_exact = torch::randn({n * n}, cuda_opts);
  auto b_tensor = A(x_exact).contiguous();

  Mat amat;
  PetscCallAbort(PETSC_COMM_SELF,
                 MatCreateShell(PETSC_COMM_SELF, n * n, n * n, n * n, n * n, &ctx, &amat));
  PetscCallAbort(PETSC_COMM_SELF,
                 MatShellSetOperation(amat, MATOP_MULT, (void (*)(void))cudaAnisoShellMatMult));
  PetscCallAbort(
      PETSC_COMM_SELF,
      MatShellSetOperation(amat, MATOP_GET_DIAGONAL, (void (*)(void))cudaAnisoShellGetDiagonal));
  PetscCallAbort(PETSC_COMM_SELF, MatShellSetVecType(amat, VECSEQCUDA));

  using std::chrono::duration;
  using std::chrono::high_resolution_clock;

  // Returns (iterations, solve wall-clock ms). cudaDeviceSynchronize()
  // brackets the timed region (matching DISABLED_CudaTimingSweep's
  // pattern), since CUDA kernel launches are asynchronous and the host
  // clock would otherwise measure launch latency, not completion.
  auto run = [&](PCType pc_type, Mat pmat) -> std::pair<PetscInt, double>
  {
    auto x_tensor = torch::zeros({n * n}, cuda_opts);
    Vec b_vec, x_vec;
    PetscCallAbort(
        PETSC_COMM_SELF,
        VecCreateSeqCUDAWithArray(PETSC_COMM_SELF, 1, n * n, b_tensor.data_ptr<double>(), &b_vec));
    PetscCallAbort(
        PETSC_COMM_SELF,
        VecCreateSeqCUDAWithArray(PETSC_COMM_SELF, 1, n * n, x_tensor.data_ptr<double>(), &x_vec));

    KSP ksp;
    PetscCallAbort(PETSC_COMM_SELF, KSPCreate(PETSC_COMM_SELF, &ksp));
    PetscCallAbort(PETSC_COMM_SELF, KSPSetOperators(ksp, amat, pmat));
    PetscCallAbort(PETSC_COMM_SELF, KSPSetType(ksp, KSPCG));
    PetscCallAbort(PETSC_COMM_SELF, KSPSetNormType(ksp, KSP_NORM_UNPRECONDITIONED));
    PetscCallAbort(PETSC_COMM_SELF, KSPSetTolerances(ksp, rel_tol, 0.0, PETSC_DEFAULT, 10 * n * n));

    PC pc;
    PetscCallAbort(PETSC_COMM_SELF, KSPGetPC(ksp, &pc));
    PetscCallAbort(PETSC_COMM_SELF, PCSetType(pc, pc_type));

    cudaDeviceSynchronize();
    const auto t0 = high_resolution_clock::now();
    PetscCallAbort(PETSC_COMM_SELF, KSPSolve(ksp, b_vec, x_vec));
    cudaDeviceSynchronize();
    const auto t1 = high_resolution_clock::now();
    const duration<double, std::milli> solve_ms = t1 - t0;

    PetscInt its;
    PetscCallAbort(PETSC_COMM_SELF, KSPGetIterationNumber(ksp, &its));

    KSPConvergedReason reason;
    PetscCallAbort(PETSC_COMM_SELF, KSPGetConvergedReason(ksp, &reason));
    EXPECT_GT(reason, 0) << "KSP (" << pc_type << ") failed to converge";
    EXPECT_NEAR((x_tensor - x_exact).norm().item<double>(), 0.0, 1e-5);

    PetscCallAbort(PETSC_COMM_SELF, KSPDestroy(&ksp));
    PetscCallAbort(PETSC_COMM_SELF, VecDestroy(&b_vec));
    PetscCallAbort(PETSC_COMM_SELF, VecDestroy(&x_vec));
    return {its, solve_ms.count()};
  };

  // Jacobi-preconditioned leg: pmat == amat (PCJACOBI only needs the
  // diagonal, which the shell's MatGetDiagonal callback already supplies).
  const auto [jacobi_its, jacobi_ms] = run(PCJACOBI, amat);

  // AMG-preconditioned leg: a real, host-assembled Pmat built from the same
  // coefficients (Section 5's assembleAnisotropicMat), matrix-free Amat
  // unchanged and still CUDA-resident (PR4a's mixed mode). Pmat assembly
  // itself is host-only work, timed separately from the (CUDA) solve.
  const auto t_asm0 = high_resolution_clock::now();
  Mat pmat = assembleAnisotropicMat(n, aniso_Dx, aniso_Dy, aniso_shift);
  const auto t_asm1 = high_resolution_clock::now();
  const duration<double, std::milli> pmat_assembly_ms = t_asm1 - t_asm0;
  const auto [amg_its, amg_ms] = run(PCGAMG, pmat);

  EXPECT_LT(amg_its, jacobi_its);
  Moose::out << "anisotropic CUDA (Dx/Dy=" << (aniso_Dx / aniso_Dy) << "): Jacobi " << jacobi_its
             << " its, " << jacobi_ms << "ms | AMG (PCGAMG) " << amg_its << " its, " << amg_ms
             << "ms solve + " << pmat_assembly_ms.count() << "ms Pmat assembly\n";

  PetscCallAbort(PETSC_COMM_SELF, MatDestroy(&pmat));
  PetscCallAbort(PETSC_COMM_SELF, MatDestroy(&amat));
}

#endif // MARLIN_PETSC_TORCH_CUDA_LEG
