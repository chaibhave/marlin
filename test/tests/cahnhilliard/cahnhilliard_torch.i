#
# Simple Cahn-Hilliard solve on a 1D grid.
# We use a torch jitted model to evaluate the expression for the time derivative of the variable
#

file_path = '/Users/bhavcv/projects/marlin/data/torch_script_pfm/ch_runner.pt'

[Domain]
    dim = 2
    ny = 50
    ymin = -25
    ymax = 25
    nx = 100
    xmin = -50
    xmax = 50
    mesh_mode = DUMMY
    device_names = 'cpu'
[]

[TensorComputes]
    [Initialize]
        [c]
            type = ParsedCompute
            extra_symbols = 'true'
            buffer = 'c'
            expression = 'exp(-2*x^2/4)'
            expand = REAL
        []
    []

    [Solve]
        [c_hat]
            type = ForwardFFT
            buffer = c_hat
            input = c
        []
        [dc_hat_dt]
            type = LibtorchPhaseFieldModel
            concentration_hat = 'c_hat'
            libtorch_model_file = '${file_path}'
            buffer = 'dc_hat_dt'
            linear = dc_hat_dt_jvp
        []
        [NL]
            type = ParsedCompute
            inputs = 'dc_hat_dt dc_hat_dt_jvp c_hat'
            buffer = 'NL'
            expression = 'dc_hat_dt - dc_hat_dt_jvp * c_hat'
        []
        [dc_dt]
            type = InverseFFT
            buffer = dc_dt
            input = dc_hat_dt
        []
        [NL_real]
            type = InverseFFT
            buffer = NL_real
            input = NL
        []
        [linear]
            type = InverseFFT
            buffer = linear
            input = dc_hat_dt_jvp
        []
    []

[]

# [TensorSolver]
#   type = ForwardEulerSolver
#   time_derivative_reciprocal = dc_hat_dt
#   buffer = c
#   reciprocal_buffer = c_hat
#   substeps = 50
# []

# [TensorSolver]
#   type = AdamsBashforthMoulton
#   buffer = c
#   reciprocal_buffer = c_hat
#   linear_reciprocal = dc_hat_dt_jvp
#   nonlinear_reciprocal = NL
#   substeps = 1
# []

[TensorOutputs]
  [c]
    type = XDMFTensorOutput
    buffer = 'c dc_dt NL_real linear'
    enable_hdf5 = true
  []
[]

[Problem]
  type = TensorProblem
[]

[Executioner]
  type = Transient
#   end_time = ${units 50 s }
  num_steps = 1
  dt = 1e-4
[]

[Outputs]
    perf_graph = True
[]