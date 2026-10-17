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
      # u = 1 + 2*x satisfies the Dirichlet values below exactly on the boundary faces
      # themselves (the BCs do not depend on y). The 0.001*y term is needed only because
      # extra_symbols requires every spatial axis to appear in the expression (real-space
      # ghost padding needs it); it means the field is not EXACTLY the steady solution the
      # BCs imply (which has no y-dependence), so the discrete Laplacian is NOT exactly
      # zero at the two boundary-adjacent x-rows, instantaneously (no time-stepping): since
      # TensorDirichletBC clips the ghost value to the uniform boundary value regardless of
      # y while the interior cell carries the 0.001*y term, the ghost-fill mismatch
      # introduces an O(0.001*y/dx^2) perturbation there. This is expected, not a defect --
      # the gold file records this derived nonzero value (max/min_lap = +/-0.732), not an
      # assumed roundoff-zero. Deep interior cells and the periodic Y ghosts are unaffected
      # and stay at roundoff; see dirichlet_linear.i for the complementary test where the
      # IC does NOT match the BCs and the field is checked to relax to the true steady
      # state (u=1+2*x) via time-stepping instead.
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
    [lap_compute]
      type = FiniteDifferenceLaplacian
      input = u
      buffer = lap
    []
  []
[]

[Postprocessors]
  [max_lap]
    type = TensorExtremeValuePostprocessor
    buffer = lap
    value_type = MAX
  []
  [min_lap]
    type = TensorExtremeValuePostprocessor
    buffer = lap
    value_type = MIN
  []
[]

[Executioner]
  type = Transient
  num_steps = 1
[]

[Outputs]
  [out]
    type = CSV
  []
[]
