interface_width = 2.0
r0 = 3

L = '${fparse 1.0 * 1.6 / ${interface_width}}'

[Domain]
  dim = 2
  nx = 32
  ny = 32
  xmax = 8
  ymax = 8
  mesh_mode = DUMMY
  floating_precision = SINGLE
[]

[TensorComputes]
  [Initialize]
    [gr0]
      type = ParsedCompute
      buffer = 'gr0'
      extra_symbols = 'true'
      expression = 'radius:=sqrt((x-${Domain/xmax}/2)^2+(y-${Domain/ymax}/2)^2);0.5 - 0.5*tanh(2*(radius-${r0})/${interface_width})'
    []
    [gr1]
      type = ParsedCompute
      buffer = 'gr1'
      extra_symbols = 'true'
      expression = 'radius:=sqrt((x-${Domain/xmax}/2)^2+(y-${Domain/ymax}/2)^2);0.5 + 0.5*tanh(2*(radius-${r0})/${interface_width})'
    []
    [smooth]
      type = DeAliasingTensor
      method = HOULI
      buffer = smooth
    []
    # Two grain orientations: identity, and the Sigma-5-like 36.86989765 deg
    # rotation about [001] already used for the pairwise hull data in
    # scripts/gb_energy/generate_gb_hulls.py (R1/R2).
    [orientations]
      type = GrainOrientationsFromInput
      buffer = orientations
      rotation_matrices = '1 0 0 0 1 0 0 0 1;
                           0.8 -0.6 0 0.6 0.8 0 0 0 1'
    []
  []
  [Solve]
    ## FFT
    [gr0_bar]
      type = ForwardFFT
      buffer = gr0_bar
      input = gr0
    []
    [gr1_bar]
      type = ForwardFFT
      buffer = gr1_bar
      input = gr1
    []

    # Gradients
    [grad_gr0]
      type = GradientVector
      buffer = grad_gr0
      input = gr0_bar
      input_is_reciprocal = true
    []
    [grad_gr1]
      type = GradientVector
      buffer = grad_gr1
      input = gr1_bar
      input_is_reciprocal = true
    []

    # O(N) anisotropic GB energy: builds the pairwise hull once (lazily, on
    # the first call, since orientations are not populated before then; see
    # class documentation) and accumulates the nonlinear update, sigma, the
    # shared linear coefficient, and the total free energy density directly.
    [gb_energy]
      type = PolycrystalAnisotropicGBEnergy
      op_buffers = 'gr0 gr1'
      grad_buffers = 'grad_gr0 grad_gr1'
      reciprocal_buffers = 'gr0_bar gr1_bar'
      orientations = orientations
      dealiasing_buffer = smooth
      NL = 'NL_gr0 NL_gr1'
      linear_coefficient = 'linear_coefficient'
      sigma = 'sigma'
      buffer = 'total_energy'
      libtorch_model_file = 'marlin:anisotropic_gb_energy/Ni_GB5DOF.pt'
      interface_width = '${interface_width}'
      L0 = '${L}'
    []
  []
[]

[Postprocessors]
  [total_gb_energy]
    type = TensorIntegralPostprocessor
    buffer = total_energy
  []
[]

[TensorSolver]
  type = AdamsBashforthMoulton
  buffer = 'gr0 gr1'
  reciprocal_buffer = 'gr0_bar gr1_bar'
  linear_reciprocal = 'linear_coefficient linear_coefficient'
  nonlinear_reciprocal = 'NL_gr0 NL_gr1'
  substeps = 1e2
  predictor_order = 1
  corrector_order = 1
  corrector_steps = 1
[]

[TensorOutputs]
  active = ''
  [xdmf]
    type = XDMFTensorOutput
    buffer = 'gr0 gr1 sigma total_energy'
    output_mode = 'NODE NODE NODE NODE'
    enable_hdf5 = true
    transpose = false
  []
[]

[Executioner]
  type = Transient
  dt = 1
  num_steps = 2
[]

[Outputs]
  csv = true
  execute_on = 'INITIAL TIMESTEP_END'
[]
