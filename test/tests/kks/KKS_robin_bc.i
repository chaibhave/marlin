#
# Kim-Kim-Suzuki with no-flux BC imposed using the smooth boundary method (SBM), solved on a 2D grid.
# Mask tensor 'psi' supplies the mask for the solve region to the system.
# Note: c is not directly conserved here - the masked value (psi > 0.0)*c will however be conserved.
#

# Constants for Initial Conditions
l = 4

# Phase-field model parameters
kappa_eta = 5
rho_sq = 2
w = 1
M = 5
L = 5
c0_a = 0.3
c0_b = 0.7

# Expressions for switching function and bulk Gibbs energy
h_eta = '(eta^3*(6*eta^2-15*eta+10))'
F = '${h_eta}*(${rho_sq}*((c - (1-${h_eta})*(${c0_b} - ${c0_a}))-${c0_a})^2) + (1-${h_eta})*(${rho_sq}*((c + (${h_eta})*(${c0_b} - ${c0_a}))-${c0_b})^2 ) + ${w}*(eta^2)*(1-eta)^2'

[Domain]
    dim = 2
    nx = 200
    ny = 200

    xmin = -50
    xmax = 50
    ymin = -50
    ymax = 50

    # run on a CUDA device (adjust this to `cpu` if not available)
    device_names = 'cpu'

    # automatically create a matching mesh
    mesh_mode = DUMMY
[]

[TensorComputes]
    [Initialize]
        [c_IC]
            type = SmoothRectangleCompute
            x1 = -40
            x2 = 0
            y1 = -55
            y2 = 55
            buffer = c
            int_width = ${l}
            profile = COS
            inside = ${c0_a}
            outside = ${c0_b}
        []
        [eta_IC]
            type = SmoothRectangleCompute
            x1 = -40
            x2 = 0
            y1 = -55
            y2 = 55
            buffer = eta
            int_width = ${l}
            profile = COS
            inside = 1
            outside = 0
        []
        [psi_init]
            type = SmoothRectangleCompute
            x1 = -35
            x2 = 35
            y1 = -55
            y2 = 55
            buffer = psi
            int_width = ${fparse ${l} / 2 }
            profile = COS
            inside = 1
            outside = 0
        []
        [zero]
            type = ConstantReciprocalTensor
            buffer = zero
        []
        [zero_real]
            type = ConstantTensor
            buffer = zero_real
        []
        [M]
            type = ConstantTensor
            buffer = M
            real = ${M}
        []
        [two_M]
            type = ConstantTensor
            buffer = two_M
            real = ${fparse 10 * ${M} }
        []
        [L]
            type = ConstantTensor
            buffer = L
            real = ${L}
        []
        [L_kappa]
            type = ConstantTensor
            buffer = L_kappa
            real = '${fparse  ${L} * ${kappa_eta} }'
        []
        [smooth]
            type = DeAliasingTensor
            buffer = smooth
            method = SHARP
        []
        [W_N]
            type = SmoothRectangleCompute
            x1 = -45
            x2 = 0
            y1 = -55
            y2 = 55
            buffer = W_N
            inside = 0
            outside = 1
            int_width = ${l}
            profile = COS
        []
        [B_D]
            type = SmoothRectangleCompute
            x1 = -45
            x2 = 0
            y1 = -55
            y2 = 55
            buffer = B_D
            inside = 0.4
            outside = 0.0
            int_width = ${l}
            profile = COS
        []
    []
    [Solve]
        [cbar]
            type = ForwardFFT
            buffer = cbar
            input = c
        []
        [etabar]
            type = ForwardFFT
            buffer = etabar
            input = eta
        []
        [mu]
            type = ParsedCompute
            buffer = 'mu'
            expression = '${F}'
            inputs = 'c eta'
            derivatives = 'c'
        []
        [div_J]
            type = ReciprocalMatDiffusion
            buffer = 'div_J'
            chemical_potential = mu
            mobility = M
            psi = psi
        []
        [robin_BC]
            type = ReciprocalRobinBC
            buffer = robin_BC
            chemical_potential = mu
            mobility = two_M
            B_D = B_D
            B_N = zero_real
            W_N = W_N
            psi = psi
        []
        [real_robin_out]
            type = InverseFFT
            buffer = real_robin_out
            input = robin_BC
        []
        [NL_c]
            type = ParsedCompute
            buffer = NL_c
            inputs = 'div_J smooth robin_BC'
            expression = 'smooth * (div_J + robin_BC )'
        []
        [domega_chem_deta]
            type = ParsedCompute
            buffer = 'domega_chem_deta'
            expression = '${F} - mu*c'
            inputs = 'mu c eta'
            derivatives = 'eta'
        []
        [AC_bulk]
            type = ReciprocalAllenCahn
            buffer = AC_bulk
            dF_chem_deta = domega_chem_deta
            L = L
            psi = psi
        []
        [kappa_grad_eta]
            type = ReciprocalMatDiffusion
            buffer = 'kappa_grad_eta'
            chemical_potential = 'eta'
            mobility = 'L_kappa'
            psi = psi
        []
        [AC_bar]
            type = ParsedCompute
            buffer = AC_bar
            expression = 'smooth*(kappa_grad_eta + AC_bulk)'
            inputs = 'AC_bulk kappa_grad_eta smooth'
        []
    []
[]

[TensorSolver]
    type = AdamsBashforthMoulton
    buffer = 'c eta'
    reciprocal_buffer = 'cbar etabar'
    linear_reciprocal = 'zero zero'
    nonlinear_reciprocal = 'NL_c AC_bar'
    substeps = 1e3
    predictor_order = 3
    corrector_order = 1
    corrector_steps = 1
[]

[Postprocessors]
    [total_C]
        type = TensorIntegralPostprocessor
        buffer = c
        execute_on = 'INITIAL TIMESTEP_END'
    []
    [total_eta]
        type = TensorIntegralPostprocessor
        buffer = eta
        execute_on = 'INITIAL TIMESTEP_END'
    []
[]

[Problem]
    type = TensorProblem
[]

[TensorOutputs]
    [xdmf]
        type = XDMFTensorOutput
        buffer = 'eta c mu psi domega_chem_deta B_D real_robin_out'
        output_mode = 'node node node node node node node'
        enable_hdf5 = true
        transpose = false
    []
[]

[Executioner]
    type = Transient
    dt = 0.01
    end_time = 100
[]

[Outputs]
    csv = true
    perf_graph = true
    execute_on = 'INITIAL TIMESTEP_END'
[]
