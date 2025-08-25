# 1D diffusion solved with ETDRK4 method.
# Analytical solution u(x,t) = sin(x) * exp(-t)

[Domain]
  dim = 1
  nx = 64
  xmax = '${fparse 2*pi}'
[]

[TensorComputes]
  [Initialize]
    [u]
      type = ParsedCompute
      buffer = u
      extra_symbols = true
      expression = 'sin(x)'
      expand = REAL
    []

    [Du]
      type = ReciprocalLaplacianFactor
      factor = 1
      buffer = Du
    []

    [zero]
      type = ConstantTensor
      buffer = zero
      real = 0
    []
  []

  [Solve]
    [source_u_bar]
      type = ForwardFFT
      buffer = source_u_bar
      input = zero
    []
  []

  [Postprocess]
    [exact_u]
      type = ParsedCompute
      buffer = exact_u
      extra_symbols = true
      expression = 'sin(x)*exp(-time)'
      expand = REAL
    []

    [err]
      type = ParsedCompute
      buffer = err
      expression = 'u - exact_u'
      inputs = 'u exact_u'
    []

    [err2]
      type = ParsedCompute
      buffer = err2
      expression = 'err^2'
      inputs = err
    []
  []
[]

[TensorSolver]
  type = ETDRK4Solver
  buffer = u
  linear_reciprocal = Du
  nonlinear_reciprocal = source_u_bar
[]

[Problem]
  type = TensorProblem
[]

[Postprocessors]
  [err_norm]
    type = TensorIntegralPostprocessor
    buffer = err2
  []
[]

[Executioner]
  type = Transient
  dt = 0.1
  num_steps = 1
[]

[Outputs]
  file_base = diffusion_etdrk4
  csv = true
[]
