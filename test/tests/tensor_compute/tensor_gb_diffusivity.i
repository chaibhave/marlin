l = 2
r0 = 6

x_mid = '${fparse ( ${Domain/xmin} + ${Domain/xmax} ) / 2 }'
y_mid = '${fparse ( ${Domain/ymin} + ${Domain/ymax} ) / 2 }'
r_xy = '( sqrt( (x - ${x_mid} )^2 + (y - ${y_mid} )^2 ) )'

circle_IC = '0.5 - 0.5*tanh(2*(${r_xy}-${r0})/${l})'

[Domain]
    dim = 2
    nx = 200
    ny = 200
    xmin = 0
    ymin = 0
    xmax = 20
    ymax = 20
    mesh_mode = DUMMY
    device_names = cpu
[]

[TensorComputes]
    [Initialize]
        [upper]
            type = SmoothRectangleCompute
            buffer = upper
            x1 = '${fparse ${Domain/xmin} - 5}'
            x2 = '${fparse ${Domain/xmax} + 5}'
            y1 = '${fparse ${Domain/ymin} - 5}'
            y2 = '${y_mid}'
            inside = 1
            outside = 0
            profile = TANH
            int_width = ${fparse 2 * ${l} }
        []
        [circle]
            type = ParsedCompute
            buffer = circle
            expression = '${circle_IC}'
            extra_symbols = true
        []

        [eta1]
            type = ParsedCompute
            buffer = 'eta1'
            expression = 'upper*(1-circle)'
            inputs = 'upper circle'
        []
        [eta2]
            type = ParsedCompute
            buffer = 'eta2'
            expression = '(1-upper)*(1-circle)'
            inputs = 'upper circle'
        []
        [eta3]
            type = ParsedCompute
            buffer = 'eta3'
            expression = 'circle'
            inputs = 'circle'
        []

        [gb]
            type = ParsedCompute
            buffer = gb
            expression = '16*(eta1^2 * eta2^2 + eta1^2 * eta3^2 + eta2^2 * eta3^2) + (729 - 432)*(eta1*eta2*eta3)^2'
            inputs = 'eta1 eta2 eta3'
        []
        [D_b]
            type = ConstantTensor
            buffer = D_b
            real = 1
        []
        [D_gb]
            type = ConstantTensor
            buffer = D_gb
            real = 100
        []

        [D]
            type = GBDependentDiffusivity
            D_b = D_b
            D_gb = D_gb
            GB = gb
            buffer = D
        []
        [c]
            type = ParsedCompute
            buffer = 'c'
            expression = 'circle'
            inputs = 'circle'
        []
        [psi]
            type = ConstantTensor
            buffer = psi
            real = 1
        []
        [smooth]
            type = DeAliasingTensor
            method = HOULI
            buffer = smooth
        []

    []
    [Solve]
        [cbar]
            type = ForwardFFT
            buffer = cbar
            input = c
        []
        [div_J]
            type = ReciprocalMatDiffusion
            buffer = 'div_J'
            chemical_potential = c
            mobility = D
            psi = psi
        []
        [NL]
            type = ParsedCompute
            buffer = NL
            inputs = 'div_J smooth'
            expression = 'smooth * div_J'
        []
    []
[]

[TensorSolver]
    type = AdamsBashforthMoulton
    buffer = 'c'
    reciprocal_buffer = 'cbar'
    linear_reciprocal = '0'
    nonlinear_reciprocal = 'NL'
    substeps = 1000
    predictor_order = 3
    corrector_order = 1
    corrector_steps = 2
[]

[Problem]
    type = TensorProblem
[]

[TensorOutputs]
    [xdmf]
        type = XDMFTensorOutput
        buffer = 'gb D upper circle eta1 eta2 eta3 c'
        enable_hdf5 = true
    []
[]

[Executioner]
    type = Transient
    dt = 0.01
    num_steps = 1000
[]

[Outputs]
    perf_graph = true
[]
