# Periodic Diffusion With PCG And Symmetric RBGS

## Summary

Add a single-field backward-Euler solver for the existing periodic Gaussian diffusion problem. It will solve

```text
(I - dt L) u^(n+1) = u^n
```

matrix-free with preconditioned conjugate gradients (PCG), using one or more symmetric red-black Gauss-Seidel (RBGS) sweeps as the preconditioner. Support serial CPU and distributed MPI in the first milestone, while retaining the existing explicit diffusion test.

## Implementation Changes

- Add `RealSpaceDiffusionPCG`, derived from `TensorSolver` and `IterativeTensorSolverInterface`, with parameters `buffer`, `laplacian`, `relative_tolerance`, `absolute_tolerance`, `max_iterations`, `preconditioner_sweeps`, `substeps`, and `verbose`.
- Restrict this milestone to a constant positive-factor, three-point `FiniteDifferenceLaplacian`, one scalar field, periodic 2D/3D domains, and even global extents. Reject five-point stencils because distance-two couplings do not have the required two-color structure.
- Refactor `FiniteDifferenceLaplacian` to expose a matrix-free `apply(full_tensor)` operation while retaining its existing compute-object behavior.
- Add reusable `TensorProblem::ownedView(tensor)` and `exchangeGhostLayers(tensor, width)` overloads. Keep the named-buffer exchange API as a delegating wrapper. This lets Krylov work vectors use halos without mutating solution buffers.
- Keep all PCG vectors and dot products owned-only. Embed an owned vector into a halo work tensor, exchange it, apply the Laplacian, and return only the owned part of `x - dt * L(x)`.
- Build checkerboard masks from global cell indices obtained through local domain bounds. Exchange the work tensor between every color half-sweep and apply the fixed sequence red, black, black, red from a zero initial correction for every preconditioner call.
- Use `_domain.allreduceSum` for every PCG dot product and norm. Warm-start from the previous solution, fail the timestep on nonconvergence, and recompute the final Laplacian output after convergence.
- Harden `conjugateGradientSolve` for exact initial guesses, zero right-hand sides, combined absolute and relative convergence, nonpositive `p^T A p`, and non-finite values without breaking existing call sites.
- Correct the real-space documentation to state that physical boundary conditions remain unsupported in this milestone and that output halos are not valid stencil results.

## Public Interfaces

- New solver type: `RealSpaceDiffusionPCG`.
- New `TensorProblem` overloads for owned views and halo exchange on temporary tensors.
- New read-only `FiniteDifferenceLaplacian` metadata and application access needed by the solver.
- Existing `RealSpaceForwardEuler`, input files, and named-buffer ghost exchange remain compatible.

## Test Plan

- Extend CG unit tests with exact-initial-guess, zero-right-hand-side, preconditioned SPD, breakdown, and absolute-tolerance cases.
- Add `diffusion_pcg.i` using the same `80 x 60` periodic Gaussian problem, `dt = 0.01`, and final time `1.0`; verify mass conservation, decay, convergence, and regression output.
- Add a periodic Fourier-mode backward-Euler test with a known discrete amplification factor.
- Run the Fourier-mode test on one and four CPU MPI ranks, using even global sizes with odd local offsets to verify global coloring and halo exchanges.
- Add expected-error tests for odd periodic extents, five-point stencils, invalid diffusion factors, and exhausted PCG iterations.
- Run the existing explicit real-space Laplacian and diffusion tests unchanged.

## Assumptions

- Physical Dirichlet and Neumann boundary conditions, variable diffusivity, and the grain-boundary Python problem are follow-on work.
- CPU and CPU-staged MPI are acceptance requirements; other Torch devices remain best-effort in this milestone.
- One symmetric RBGS sweep is the default preconditioner, and PCG starts from the previous timestep solution.
