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
      # Start from a field that is NOT the steady solution and does not match the
      # Dirichlet BCs below pointwise (it depends weakly on y, which the BCs do not).
      # extra_symbols requires every spatial axis to appear in the expression (real-space
      # ghost padding needs it), hence the small 0.001*y term. The point of this test is
      # that pure diffusion under the Dirichlet BCs should erase the y-dependence and the
      # mismatch, driving u toward the exact steady profile u = 1 + 2*x, regardless of IC.
      expression = '1 + 2*x + 0.001*y'
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
      type = TensorDirichletBC
      variable = u
      boundary = right
      value = 3
    []
  []
  [Solve]
    [dt]
      type = FiniteDifferenceLaplacian
      input = u
      buffer = dudt
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
  [max_u]
    type = TensorExtremeValuePostprocessor
    buffer = u
    value_type = MAX
    execute_on = 'INITIAL TIMESTEP_END'
  []
  [min_u]
    type = TensorExtremeValuePostprocessor
    buffer = u
    value_type = MIN
    execute_on = 'INITIAL TIMESTEP_END'
  []
[]

[Executioner]
  type = Transient
  # dx=0.05, dy=0.25; explicit-diffusion CFL limit ~1/(2*(1/dx^2+1/dy^2)) ~= 1.2e-3,
  # substeps=10 above divides dt into 10 explicit sub-updates, so the effective
  # per-substep step is dt/10 and must itself respect the CFL limit.
  dt = 1e-2
  num_steps = 400
[]

[Outputs]
  [out]
    type = CSV
  []
[]
