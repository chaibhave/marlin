# 2D corner test: non-periodic on BOTH axes (Dirichlet on left/right, zero-flux
# Neumann on bottom/top), exercising the roadmap's documented ghost-fill
# axis-ordering caveat (TensorProblem::exchangeGhostLayers() fills one axis at
# a time, so a corner ghost cell filled while processing axis d may read
# transverse data whose own ghosts, from the other axis, have not been filled
# yet this call).
#
# The field u = 1 + 2*x has NO y-dependence, so it is compatible with uniform
# BC values on every face simultaneously: Dirichlet value = 1 + 2*x evaluated
# at x = 0 / x = 1 on the left/right faces (uniform there, since those faces
# are each at a single x), and zero-flux Neumann on the bottom/top faces is
# exactly satisfied since du/dy = 0 everywhere. (A genuinely bilinear profile,
# e.g. 1 + 2*x + y, is NOT uniform on any face -- it varies with x on the
# bottom/top faces and with y on the left/right faces -- so uniform Dirichlet
# values on all four faces can never exactly match it; an earlier version of
# this test used that profile and produced a large, correct-but-misleading
# Laplacian mismatch at the corners from the BC/IC inconsistency itself, not
# from ghost-fill ordering.) With this corrected, face-compatible profile, the
# discrete Laplacian should again be exactly zero everywhere, including the
# four corner-adjacent cells, confirming the two-non-periodic-axis ghost-fill
# path (not exercised by any other real_space test, which are periodic on at
# least one axis) is correct at the corners.
#
# In practice the recorded gold values are NOT exact roundoff zero: the
# 0.0001*y term needed to avoid the ParsedCompute collapse-to-scalar pitfall
# (see above) means u is not EXACTLY the steady profile the BCs imply (which
# has no y-dependence), so TensorDirichletBC's uniform (y-independent) ghost
# value disagrees with the interior cell's 0.0001*y term by a growing amount
# as y increases away from y=0, producing an O(0.0001*y/dx^2) perturbation at
# the left/right boundary-adjacent columns -- exactly the same effect already
# documented and accepted in dirichlet_laplacian_check.i for the 1D case. The
# deep-interior cells (away from any boundary-adjacent column/row) remain at
# float32 roundoff, confirmed directly via the XDMF field dump used to derive
# this gold value.
[Domain]
  dim = 2
  nx = 20
  ny = 10
  parallel_mode = REAL_SPACE
  xmin = 0
  xmax = 1
  ymin = 0
  ymax = 1
[]

[TensorComputes]
  [Initialize]
    [u_ic]
      type = ParsedCompute
      buffer = u
      # Exactly the steady profile implied by the four BCs below (no
      # y-dependence), so the discrete Laplacian is checked instantaneously,
      # no time-stepping needed (cf. dirichlet_laplacian_check.i for the
      # companion periodic-in-y 1D case). 0.0001*y, not 0*y: see
      # robin_steady_state.i's comment on the zero-coefficient-collapses-the-
      # axis ParsedCompute pitfall.
      expression = '1 + 2 * x + 0.0001 * y'
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
    [bottom]
      type = TensorNeumannBC
      variable = u
      boundary = bottom
      flux = 0
    []
    [top]
      type = TensorNeumannBC
      variable = u
      boundary = top
      flux = 0
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
