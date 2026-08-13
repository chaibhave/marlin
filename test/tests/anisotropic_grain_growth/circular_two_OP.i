interface_width = 0.8
r0 = 10

gbe_max = '${units 1.3561 J/m^2}'

L = '${fparse 1.0 * 1.6 / ${interface_width} }'

g_gamma0 = '${fparse sqrt(2) / 3 }' # g(gamma=1.5)
f0_gamma0 = 0.1411
kappa = '${fparse gbe_max * interface_width * sqrt(f0_gamma0) / g_gamma0 }'
mu = '${fparse gbe_max  / (g_gamma0 * interface_width * sqrt(f0_gamma0) ) }'

g_to_gamma_func = '(1.0/(-3.0944 * g^8 -1.8169*g^6 + 10.323 * g^4 - 8.1819*g^2 + 2.0033))'
dgamma_dsigma = '(g * (24.7552*g^6 + 10.9014*g^4 - 41.292*g^2 + 16.3638) / (-3.0944*g^8 - 1.8169*g^6 + 10.323*g^4 - 8.1819*g^2 + 2.0033)^2 / sqrt(${kappa} * ${mu}) )'

f0 = '((gr0^4/4 - gr0^2/2) + (gr1^4/4 - gr1^2/2) + gamma_01*gr0^2*gr1^2 + 0.25)'

[Domain]
  dim = 2
  nx = 400
  ny = 400
  xmax = 40
  ymax = 40
  mesh_mode = DUMMY
  device_names = 'cuda'
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
        [kappa_linear_term]
            type = ReciprocalLaplacianFactor
            buffer = kappa_linear_term
            factor = ${fparse ${L} * ${kappa} }
        []
        [kappa_laplacian]
            type = ReciprocalLaplacianFactor
            buffer = kappa_laplacian
            factor = '${kappa}'
        []
        [smooth]
            type = DeAliasingTensor
            method = HOULI
            buffer = smooth
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
            input = gr0 #_bar
            
        []
        [grad_gr1]
            type = GradientVector
            buffer = grad_gr1
            input = gr1 #_bar
        []

        # Get interface energy
        [sigma_gr0_gr1]
            type = PairwiseAnisotropicGBEnergy
            buffer = 'sigma_gr0_gr1'
            dsigma_dgrad_grain1 = 'dsigma_gr0_gr1_dgrad_gr0'
            dsigma_dgrad_grain2 = 'dsigma_gr0_gr1_dgrad_gr1'
            grad_grain1_buffer = 'grad_gr0'
            grad_grain2_buffer = 'grad_gr1'
            interface_width = '${interface_width}'
            grad_gb = grad_gb
            libtorch_model_file = '/scratch/bhavcv/aniso_grain_growth/aniso_pfm_paper/sigma3_N_1e4.pt'
        []

        [torque_01_gr0]
            type = AnisotropyTorque
            buffer = torque_01_gr0
            dmu_dn = dsigma_gr0_gr1_dgrad_gr0
            g = 'coeff_01'
            input = 'gr0'
        []
        [torque_01_gr1]
            type = AnisotropyTorque
            buffer = torque_01_gr1
            dmu_dn = dsigma_gr0_gr1_dgrad_gr1
            g = 'coeff_01'
            input = 'gr1'
        []
        [gr0_bulk_term]
            type = ParsedCompute
            buffer = 'gr0_bulk_term'
            expression = '-${L}*${mu} * (gr0^3 - gr0 + 2*gr0*(gamma_01*gr1^2) - torque_01_gr0 )'
            inputs = 'gr0 gr1 gamma_01 torque_01_gr0'
        []
        [gr1_bulk_term]
            type = ParsedCompute
            buffer = 'gr1_bulk_term'
            expression = '-${L}*${mu} * (gr1^3 - gr1 + 2*gr1*(gamma_01*gr0^2) - torque_01_gr1)'
            inputs = 'gr0 gr1 gamma_01 torque_01_gr1'
        []


        ## build non-linear terms and smooth them
        [NL_gr0]
            type = ForwardFFT
            buffer = NL_gr0
            input = 'gr0_bulk_term'
        []
        [NL_gr1]
            type = ForwardFFT
            buffer = NL_gr1
            input = 'gr1_bulk_term'
        []
        [NL_gr0_smooth]
            type = ParsedCompute
            buffer = NL_gr0_smooth
            expression = 'smooth*NL_gr0'
            inputs = 'smooth NL_gr0'
        []
        [NL_gr1_smooth]
            type = ParsedCompute
            buffer = NL_gr1_smooth
            expression = 'smooth*NL_gr1'
            inputs = 'smooth NL_gr1'
        []

        # Properties for output
        [sigma]
            type = ParsedCompute
            buffer = 'sigma'
            expression = 'gr0^2*gr1^2*sigma_gr0_gr1'
            inputs = 'sigma_gr0_gr1 gr0 gr1'
        []
        [total_energy]
            type = ParsedCompute
            buffer = 'total_energy'
            expression = '${mu} * (${f0}) + kappa_laplacian_gr_i'
            inputs = 'gr0 gr1 gr2 gamma_01 kappa_laplacian_gr_i'
        []
        [kappa_laplacian_gr_i_bar]
            type = ParsedCompute
            buffer = kappa_laplacian_gr_i
            expression = 'kappa_laplacian * (gr0_bar + gr1_bar)'
            inputs = 'gr0_bar gr1_bar kappa_laplacian'
            extra_symbols = true
        []
        [kappa_laplacian_gr_i]
            type = InverseFFT
            buffer = kappa_laplacian_gr_i
            input = kappa_laplacian_gr_i_bar            
        []
    []
[]

[Postprocessors]
    [wall_time]
        type = PerfGraphData
        section_name = "Root"
        data_type = total
    []
    [total_gb_energy]
        type = TensorIntegralPostprocessor
        buffer = total_energy
    []
[]

[TensorSolver]
    type = AdamsBashforthMoulton
    buffer = 'gr0 gr1'
    reciprocal_buffer = 'gr0_bar gr1_bar'
    linear_reciprocal = 'L_kappa_laplacian L_kappa_laplacian'
    nonlinear_reciprocal = 'NL_gr0_smooth NL_gr1_smooth'
    substeps = 1000
    predictor_order = 1
    corrector_order = 1
    corrector_steps = 1
[]

[TensorOutputs]
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
    dt = 0.1
    num_steps = 20
[]

[Outputs]
    csv = true
    perf_graph = true
    execute_on = 'INITIAL TIMESTEP_END'
[]
