# Three-component Bateman chain with off-diagonal linear couplings
ss = 2
dt = 0.5
lambda1 = 0.4
lambda2 = 0.2
D = 0.01

[Domain]
  dim = 2
  nx = 100
  xmax = 10
  ny = 100
  ymax = 100
  mesh_mode = DUMMY
[]

[TensorComputes]
  [Initialize]
    [c1]
      type = ConstantTensor
      buffer = c1
      real = '0.0342'
    []
    [c2]
      type = ConstantTensor
      buffer = c2
      real = '0.0113'
    []
    [c3]
      type = ConstantTensor
      buffer = c3
    []
    [zero]
      type = ConstantReciprocalTensor
      buffer = zero
      real = 0.0      
    []

    [L]
      type = ReciprocalLaplacianFactor
      factor = ${D}
      buffer = L
    []

    [lambda_1]
      type = ConstantReciprocalTensor
      buffer = lambda_1
      real = ${lambda1}      
    []
    [lambda_2]
      type = ConstantReciprocalTensor
      buffer = lambda_2
      real = ${lambda2}      
    []

    [L_c1]
      type = ParsedCompute
      buffer = L_c1
      inputs = 'L lambda_1'
      expression = 'L - lambda_1'
    []
    [L_c2]
      type = ParsedCompute
      buffer = L_c2
      inputs = 'L lambda_2'
      expression = 'L - lambda_2'
    []
    [L_c3]
      type = ParsedCompute
      buffer = L_c3
      inputs = L
      expression = 'L'
    []

    [r12_bar]
      type = ConstantReciprocalTensor
      buffer = r12_bar
      real = ${lambda1}
    []
    [r23_bar]
      type = ConstantReciprocalTensor
      buffer = r23_bar
      real = ${lambda2}
    []
  []

  [Solve]
    [c1_bar]
      type = ForwardFFT
      buffer = c1_bar
      input = c1
    []
    [c2_bar]
      type = ForwardFFT
      buffer = c2_bar
      input = c2
    []
    [c3_bar]
      type = ForwardFFT
      buffer = c3_bar
      input = c3
    []
  []
[]

[TensorSolver]
  type = ETDRK4Solver
  buffer = 'c1 c2 c3'
  reciprocal_buffer = 'c1_bar c2_bar c3_bar'
  linear_reciprocal = 'L_c1 L_c2 L_c3'
  linear_offdiag_rows = '1 2'
  linear_offdiag_cols = '0 1'
  linear_offdiag = 'r12_bar r23_bar'
  nonlinear_reciprocal = 'zero zero zero'
  substeps = ${ss}
  # predictor_order = 1
  # corrector_order = 1
  # corrector_steps = 0
[]

[Problem]
  type = TensorProblem
[]

[Postprocessors]
  [c1_int]
    type = TensorIntegralPostprocessor
    buffer = c1
  []
  [c2_int]
    type = TensorIntegralPostprocessor
    buffer = c2
  []
  [c3_int]
    type = TensorIntegralPostprocessor
    buffer = c3
  []
[]

[Executioner]
  type = Transient
  num_steps = 4
  dt = ${dt}
[]

[Outputs]
  file_base = etdrk4_bateman_3c
  csv = true
[]
