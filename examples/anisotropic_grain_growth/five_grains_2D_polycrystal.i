interface_width = 1.6
side = 80

L = '${fparse 1.0 * 1.6 / ${interface_width} }'

# ------------------------------------------------------------------
# Periodic Voronoi-like IC for 5 grains. Identical construction to
# five_grains_2D.i (see that file for the derivation); duplicated here
# so this O(N) example stands on its own.
# ------------------------------------------------------------------
aexp = '${fparse 4.0 / ${interface_width} }'
lp = '${fparse ${side} / pi }'
kp = '${fparse pi / ${side} }'

# seed points (spread out over the periodic 80x80 domain)
d0 = 'sqrt((${lp}*sin(${kp}*(x-8)))^2  + (${lp}*sin(${kp}*(y-6)))^2)'
d1 = 'sqrt((${lp}*sin(${kp}*(x-28)))^2 + (${lp}*sin(${kp}*(y-10)))^2)'
d2 = 'sqrt((${lp}*sin(${kp}*(x-20)))^2 + (${lp}*sin(${kp}*(y-24)))^2)'
d3 = 'sqrt((${lp}*sin(${kp}*(x-4)))^2  + (${lp}*sin(${kp}*(y-28)))^2)'
d4 = 'sqrt((${lp}*sin(${kp}*(x-34)))^2 + (${lp}*sin(${kp}*(y-33)))^2)'

[Domain]
    dim = 2
    nx = 200
    ny = 200
    xmax = ${side}
    ymax = ${side}
    mesh_mode = DUMMY
[]

[TensorComputes]
    [Initialize]
        # unnormalized softmax weights
        [c0]
            type = ParsedCompute
            buffer = c0
            expression = 'exp(-${aexp}*${d0})'
            extra_symbols = true
        []
        [c1]
            type = ParsedCompute
            buffer = c1
            expression = 'exp(-${aexp}*${d1})'
            extra_symbols = true
        []
        [c2]
            type = ParsedCompute
            buffer = c2
            expression = 'exp(-${aexp}*${d2})'
            extra_symbols = true
        []
        [c3]
            type = ParsedCompute
            buffer = c3
            expression = 'exp(-${aexp}*${d3})'
            extra_symbols = true
        []
        [c4]
            type = ParsedCompute
            buffer = c4
            expression = 'exp(-${aexp}*${d4})'
            extra_symbols = true
        []

        # normalized order parameters (partition of unity by construction)
        [gr0]
            type = ParsedCompute
            buffer = gr0
            expression = 'c0/(c0+c1+c2+c3+c4)'
            inputs = 'c0 c1 c2 c3 c4'
        []
        [gr1]
            type = ParsedCompute
            buffer = gr1
            expression = 'c1/(c0+c1+c2+c3+c4)'
            inputs = 'c0 c1 c2 c3 c4'
        []
        [gr2]
            type = ParsedCompute
            buffer = gr2
            expression = 'c2/(c0+c1+c2+c3+c4)'
            inputs = 'c0 c1 c2 c3 c4'
        []
        [gr3]
            type = ParsedCompute
            buffer = gr3
            expression = 'c3/(c0+c1+c2+c3+c4)'
            inputs = 'c0 c1 c2 c3 c4'
        []
        [gr4]
            type = ParsedCompute
            buffer = gr4
            expression = '1 - gr0 - gr1 - gr2 - gr3'
            inputs = 'gr0 gr1 gr2 gr3'
        []

        [smooth]
            type = DeAliasingTensor
            method = HOULI
            buffer = smooth
        []

        # Grain orientations: the same R1..R5 Ni misorientations used in
        # scripts/gb_energy/generate_gb_hulls.py (identity, Sigma-5-like
        # 36.87deg/[001], Sigma-3-like 60deg/[111], Sigma-7-like 38.21deg/[111],
        # Sigma-9-like 38.94deg/[110]).
        [orientations]
            type = GrainOrientationsFromInput
            buffer = orientations
            rotation_matrices = '1 0 0 0 1 0 0 0 1;
                                  0.8 -0.6 0 0.6 0.8 0 0 0 1;
                                  0.6666667 -0.3333333 0.6666667 0.6666667 0.6666667 -0.3333333 -0.3333333 0.6666667 0.6666667;
                                  0.8571429 0.4285714 -0.2857143 -0.2857143 0.8571429 0.4285714 0.4285714 -0.2857143 0.8571429;
                                  0.7777778 0.4444444 0.4444444 -0.4444444 0.8888889 -0.1111111 -0.4444444 -0.1111111 0.8888889'
        []
    []
    [Solve]
        [bnds]
            type = ParsedCompute
            buffer = 'bnds'
            inputs = 'gr0 gr1 gr2 gr3 gr4'
            expression = 'gr0^2 + gr1^2 + gr2^2 + gr3^2 + gr4^2'
        []

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
        [gr2_bar]
            type = ForwardFFT
            buffer = gr2_bar
            input = gr2
        []
        [gr3_bar]
            type = ForwardFFT
            buffer = gr3_bar
            input = gr3
        []
        [gr4_bar]
            type = ForwardFFT
            buffer = gr4_bar
            input = gr4
        []

        [grad_gr0]
            type = GradientVector
            buffer = grad_gr0
            input = gr0
        []
        [grad_gr1]
            type = GradientVector
            buffer = grad_gr1
            input = gr1
        []
        [grad_gr2]
            type = GradientVector
            buffer = grad_gr2
            input = gr2
        []
        [grad_gr3]
            type = GradientVector
            buffer = grad_gr3
            input = gr3
        []
        [grad_gr4]
            type = GradientVector
            buffer = grad_gr4
            input = gr4
        []

        [gb_energy]
            type = PolycrystalAnisotropicGBEnergy
            op_buffers = 'gr0 gr1 gr2 gr3 gr4'
            grad_buffers = 'grad_gr0 grad_gr1 grad_gr2 grad_gr3 grad_gr4'
            reciprocal_buffers = 'gr0_bar gr1_bar gr2_bar gr3_bar gr4_bar'
            orientations = orientations
            dealiasing_buffer = smooth
            NL = 'NL_gr0 NL_gr1 NL_gr2 NL_gr3 NL_gr4'
            linear_coefficient = 'linear_coefficient'
            sigma = 'sigma'
            buffer = 'total_energy'
            libtorch_model_file = 'marlin:anisotropic_gb_energy/Ni_GB5DOF.pt'
            interface_width = '${interface_width}'
            L0 = '${L}'
        []

        # Properties for output
        [grain_ID]
            type = ParsedCompute
            buffer = 'grain_ID'
            expression = 'if(gr0>=max(gr1,max(gr2,max(gr3,gr4))), 0, if(gr1>=max(gr2,max(gr3,gr4)), 1, if(gr2>=max(gr3,gr4), 2, if(gr3>=gr4, 3, 4))))'
            inputs = 'gr0 gr1 gr2 gr3 gr4'
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
    buffer = 'gr0 gr1 gr2 gr3 gr4'
    reciprocal_buffer = 'gr0_bar gr1_bar gr2_bar gr3_bar gr4_bar'
    linear_reciprocal = 'linear_coefficient linear_coefficient linear_coefficient linear_coefficient linear_coefficient'
    nonlinear_reciprocal = 'NL_gr0 NL_gr1 NL_gr2 NL_gr3 NL_gr4'
    substeps = 100
    predictor_order = 1
    corrector_order = 1
    corrector_steps = 1
[]

[TensorOutputs]
    [xdmf]
        type = XDMFTensorOutput
        buffer = 'gr0 gr1 gr2 gr3 gr4 bnds sigma grain_ID'
        output_mode = 'NODE NODE NODE NODE NODE NODE NODE CELL'
        enable_hdf5 = true
        transpose = true
    []
[]

[Executioner]
    type = Transient
    dt = 0.5
    num_steps = 50
[]

[Outputs]
    csv = true
    execute_on = 'INITIAL TIMESTEP_END'
[]
