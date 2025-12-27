# 1D diffusion solved with ETDRK4 method.
# Analytical solution u(x,t) = sin(x) * exp(-t)

[Domain]
  dim = 1
  nx = 64
  xmax = '${fparse 2*pi}'
[]

[GlobalParams]
  expand = REAL
[]

[TensorComputes]
  [Initialize]
    [u]
      type = ParsedCompute
      buffer = u
      extra_symbols = true
      expression = 'sin(x)'
    []

    [Du]
      type = ReciprocalLaplacianFactor
      factor = 1
      buffer = Du
    []

    [zero]
      type = ConstantReciprocalTensor
      buffer = zero
    []
  []

  [Solve]
    [ubar]
      type = ForwardFFT
      buffer = ubar
      input = u
    []

  []

  [Postprocess]
    [exact_u]
      type = ParsedCompute
      buffer = exact_u
      extra_symbols = true
      expression = 'sin(x)*exp(-t)'
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

[TensorOutputs]
    [xdmf]
        type = XDMFTensorOutput
        buffer = 'u exact_u err2 err'
        enable_hdf5 = true
        transpose = false
    []
[]

[TensorSolver]
  type = ETDRK4Solver
  buffer = u
  reciprocal_buffer = ubar
  linear_reciprocal = Du
  nonlinear_reciprocal = zero
[]
# [TensorSolver]
#   type = AdamsBashforthMoulton
#   buffer = u
#   reciprocal_buffer = ubar
#   linear_reciprocal = Du
#   nonlinear_reciprocal = zero
#   corrector_order = 5
#   predictor_order = 5
#   corrector_steps = 10
# []

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
  dt = 10
  num_steps = 1
[]

[Outputs]
  file_base = diffusion_etdrk4
  csv = true
[]
