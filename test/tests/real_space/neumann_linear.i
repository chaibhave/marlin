# Nonzero-flux Neumann boundary condition against a known linear steady profile:
# steady 1D diffusion (Laplacian(u) = 0) with u(0) = g_D (Dirichlet, left) and
# du/dn = flux (Neumann, right) has the exact solution u(x) = flux*x + g_D (the
# b=1,a=0 preset of the same linear-profile family robin_steady_state.i checks
# for a general Robin right face). This is the roadmap's required nonzero-flux
# Neumann test (PR1 plan, Sec. "Verification").
#
# Pseudo-1D domain: ny = 8 with Y periodic (consistent with the other real_space
# steady-state checks); the IC does not match the steady profile, so this is a
# genuine relaxation check, not an instantaneous one.
[Domain]
  dim = 2
  nx = 20
  ny = 8
  parallel_mode = REAL_SPACE
  periodic_directions = 'Y'
  xmin = 0
  xmax = 1
  ymin = -1
  ymax = 1
[]

[TensorComputes]
  [Initialize]
    [u_ic]
      type = ParsedCompute
      buffer = u
      # 0.0001*y (not 0*y): see robin_steady_state.i's comment on the
      # zero-coefficient-collapses-the-axis ParsedCompute pitfall.
      expression = '0.5 + 0.0001 * y'
      extra_symbols = true
    []
  []
  [BCs]
    [left]
      type = TensorDirichletBC
      variable = u
      boundary = left
      value = 1
    []
    [right]
      type = TensorNeumannBC
      variable = u
      boundary = right
      flux = 2
    []
  []
  [Solve]
    [dt]
      type = FiniteDifferenceLaplacian
      input = u
      buffer = dudt
    []
    [u_exact]
      type = ParsedCompute
      buffer = u_exact
      expression = '1 + 2 * x + 0.0001 * y'
      extra_symbols = true
    []
    [diff]
      type = ParsedCompute
      buffer = diff
      expression = 'abs(u - u_exact)'
      inputs = 'u u_exact'
    []
  []
[]

[TensorSolver]
  type = RealSpaceForwardEuler
  buffer = u
  time_derivative = dudt
  substeps = 10
[]

[Postprocessors]
  [max_error]
    type = TensorExtremeValuePostprocessor
    buffer = diff
    value_type = MAX
    execute_on = 'TIMESTEP_END'
  []
[]

[Executioner]
  type = Transient
  # dx=0.05, dy=0.25; same CFL margin as dirichlet_linear.i/robin_steady_state.i.
  # Neumann relaxation is slower here than the Dirichlet/Robin cases (the error
  # mode decays more slowly since both faces either float or have unit-gain
  # feedback rather than clamping to a fixed value), so more steps are needed to
  # flatten max_error.
  dt = 1e-2
  num_steps = 900
[]

[Outputs]
  [out]
    type = CSV
  []
[]
