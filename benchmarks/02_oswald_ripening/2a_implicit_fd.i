#
# PFHub benchmark 2a (Ostwald ripening, square periodic domain) with the implicit
# finite-difference solver. The problem of 2a.i:
#
#   dc/dt     =  div(M grad(df/dc - kappa_c Laplacian(c)))
#   deta_i/dt = -L (df/deta_i - kappa_eta Laplacian(eta_i)),  i = 1..4
#
# Backward Euler, with the gradient terms implicit and the chemical terms explicit:
#
#   (c - c^n)/dt     + M kappa_c Laplacian(Laplacian(c)) =  M Laplacian(df/dc(c^n, eta^n))
#   (eta_i - eta_i^n)/dt - L kappa_eta Laplacian(eta_i)  = -L df/deta_i(c^n, eta^n)
#
# rho = sqrt(2), c_alpha = 0.3, c_beta = 0.7, w = 1, alpha = 5, kappa_c = kappa_eta = 3,
# M = L = 5. Initial conditions as in the PFHub specification (c0 = 0.5, epsilon = 0.05,
# epsilon_eta = 0.1, psi = 1.5).
#
# The explicit chemical term limits the eta substep: linearised, the backward Euler step
# multiplies a mode by (1 - dt L f'') / (1 + dt L kappa_eta k^2), which at k = 0 needs
# dt L f'' <= 2. The largest positive f'' = d2f/deta_i^2 is about 2 w + 4 alpha = 22
# (eta_i = 0 next to a grain with eta_j = 1), so the substep must stay below
# 2 / (5 * 22) = 0.018. The c equation is not limited: (1 - dt M f''_c k^2) /
# (1 + dt M kappa_c k^4) stays within [-1, 1] for dt <= 2 / (M f''_c^2 / (4 kappa_c)) = 0.3.
#

# substeps per time step (each substep is one linear solve per field); the largest substep
# is dtmax / substeps = 0.01
substeps = 50
dtmax = 0.5
end_time = 1000

fchem = 'fa:=rho^2*(c-ca)^2;
fb:=rho^2*(cb-c)^2;
h:=n1^3*(6*n1^2-15*n1+10) +
   n2^3*(6*n2^2-15*n2+10) +
   n3^3*(6*n3^2-15*n3+10) +
   n4^3*(6*n4^2-15*n4+10);
g:=n1^2*(1-n1)^2 +
   n2^2*(1-n2)^2 +
   n3^2*(1-n3)^2 +
   n4^2*(1-n4)^2 +
alpha*(
n1^2*n2^2 + n1^2*n3^2 + n1^2*n4^2 +
n2^2*n1^2 + n2^2*n3^2 + n2^2*n4^2 +
n3^2*n1^2 + n3^2*n2^2 + n3^2*n4^2 +
n4^2*n1^2 + n4^2*n2^2 + n4^2*n3^2);
(fa*(1-h) + fb*h + w*g)'

nic = 'epsilon*(cos((0.01*idx)*x-4)*cos((0.007+0.01*idx)*y)
       +cos((0.11+0.01*idx)*x)*cos((0.11+0.01*idx)*y)
       +psi*(cos((0.046+0.001*idx)*x+(0.0405+0.001*idx)*y)
       *cos((0.031+0.001*idx)*x-(0.004+0.001*idx)*y))^2)^2'

cnames = 'rho     ca  cb  alpha w L'
cvalues = 'sqrt(2) 0.3 0.7 5     1 5'

[Domain]
  dim = 2
  nx = 200
  ny = 200
  xmax = 200
  ymax = 200
  parallel_mode = REAL_SPACE
  periodic_directions = 'X Y'
[]

[TensorBuffers]
  [c]
  []
  [n1]
  []
  [n2]
  []
  [n3]
  []
  [n4]
  []
  # implicit operator outputs
  [dcdt]
  []
  [bih_c]
  []
  [dn1dt]
  []
  [dn2dt]
  []
  [dn3dt]
  []
  [dn4dt]
  []
  [lap_n1]
  []
  [lap_n2]
  []
  [lap_n3]
  []
  [lap_n4]
  []
  # explicit sources
  [mu_c]
  []
  [rhs_c]
  []
  [rhs_n1]
  []
  [rhs_n2]
  []
  [rhs_n3]
  []
  [rhs_n4]
  []
  # postprocessing
  [lap_c]
  []
  [lap_e1]
  []
  [lap_e2]
  []
  [lap_e3]
  []
  [lap_e4]
  []
  [F]
  []
  [bnds]
  []
[]

[TensorComputes]
  [Initialize]
    [c]
      type = ParsedCompute
      buffer = c
      extra_symbols = true
      expression = 'c0+epsilon*(cos(0.105*x)*cos(0.11*y)+(cos(0.13*x)*cos(0.087*y))^2+cos(0.025*x-0.15*y)*cos(0.07*x-0.02*y))'
      constant_names = 'c0 epsilon'
      constant_expressions = '0.5 0.05'
    []
    [n1]
      type = ParsedCompute
      buffer = n1
      expression = ${nic}
      extra_symbols = true
      constant_names = 'idx epsilon psi'
      constant_expressions = '1 0.1 1.5'
    []
    [n2]
      type = ParsedCompute
      buffer = n2
      expression = ${nic}
      extra_symbols = true
      constant_names = 'idx epsilon psi'
      constant_expressions = '2 0.1 1.5'
    []
    [n3]
      type = ParsedCompute
      buffer = n3
      expression = ${nic}
      extra_symbols = true
      constant_names = 'idx epsilon psi'
      constant_expressions = '3 0.1 1.5'
    []
    [n4]
      type = ParsedCompute
      buffer = n4
      expression = ${nic}
      extra_symbols = true
      constant_names = 'idx epsilon psi'
      constant_expressions = '4 0.1 1.5'
    []
  []

  [Solve]
    # implicit operators
    [dcdt]
      type = TimeDerivativeOperator
      buffer = dcdt
      input = c
    []
    [bih_c]
      type = FiniteDifferenceBiharmonic
      buffer = bih_c
      input = c
      factor = 15 # M * kappa_c
    []
    [dn1dt]
      type = TimeDerivativeOperator
      buffer = dn1dt
      input = n1
    []
    [dn2dt]
      type = TimeDerivativeOperator
      buffer = dn2dt
      input = n2
    []
    [dn3dt]
      type = TimeDerivativeOperator
      buffer = dn3dt
      input = n3
    []
    [dn4dt]
      type = TimeDerivativeOperator
      buffer = dn4dt
      input = n4
    []
    [lap_n1]
      type = FiniteDifferenceLaplacian
      buffer = lap_n1
      input = n1
      factor = -15 # -L * kappa_eta
    []
    [lap_n2]
      type = FiniteDifferenceLaplacian
      buffer = lap_n2
      input = n2
      factor = -15 # -L * kappa_eta
    []
    [lap_n3]
      type = FiniteDifferenceLaplacian
      buffer = lap_n3
      input = n3
      factor = -15 # -L * kappa_eta
    []
    [lap_n4]
      type = FiniteDifferenceLaplacian
      buffer = lap_n4
      input = n4
      factor = -15 # -L * kappa_eta
    []

    # explicit sources, all from c^n and eta^n
    [explicit]
      [mu_c]
        type = ParsedCompute
        buffer = mu_c
        expression = ${fchem}
        constant_names = ${cnames}
        constant_expressions = ${cvalues}
        derivatives = c
        inputs = 'c n1 n2 n3 n4'
      []
      [rhs_c]
        type = FiniteDifferenceLaplacian
        buffer = rhs_c
        input = mu_c
        factor = 5 # M
      []
      [rhs_n1]
        type = ParsedCompute
        buffer = rhs_n1
        expression = '${fchem}*(-L)'
        constant_names = ${cnames}
        constant_expressions = ${cvalues}
        derivatives = n1
        inputs = 'c n1 n2 n3 n4'
      []
      [rhs_n2]
        type = ParsedCompute
        buffer = rhs_n2
        expression = '${fchem}*(-L)'
        constant_names = ${cnames}
        constant_expressions = ${cvalues}
        derivatives = n2
        inputs = 'c n1 n2 n3 n4'
      []
      [rhs_n3]
        type = ParsedCompute
        buffer = rhs_n3
        expression = '${fchem}*(-L)'
        constant_names = ${cnames}
        constant_expressions = ${cvalues}
        derivatives = n3
        inputs = 'c n1 n2 n3 n4'
      []
      [rhs_n4]
        type = ParsedCompute
        buffer = rhs_n4
        expression = '${fchem}*(-L)'
        constant_names = ${cnames}
        constant_expressions = ${cvalues}
        derivatives = n4
        inputs = 'c n1 n2 n3 n4'
      []
    []
  []

  [Postprocess]
    # On the periodic grid sum |grad u|^2 = -sum u Laplacian(u) for the 3-point Laplacian,
    # so each gradient energy (kappa/2)|grad u|^2 integrates to the same total as
    # -(kappa/2) u Laplacian(u).
    [lap_c]
      type = FiniteDifferenceLaplacian
      buffer = lap_c
      input = c
    []
    [lap_e1]
      type = FiniteDifferenceLaplacian
      buffer = lap_e1
      input = n1
    []
    [lap_e2]
      type = FiniteDifferenceLaplacian
      buffer = lap_e2
      input = n2
    []
    [lap_e3]
      type = FiniteDifferenceLaplacian
      buffer = lap_e3
      input = n3
    []
    [lap_e4]
      type = FiniteDifferenceLaplacian
      buffer = lap_e4
      input = n4
    []
    [F]
      type = ParsedCompute
      buffer = F
      expression = '${fchem} - kappa / 2 * (c * lap_c + n1 * lap_e1 + n2 * lap_e2 + n3 * lap_e3 + n4 * lap_e4)'
      # kappa_c = kappa_eta = 3
      constant_names = '${cnames} kappa'
      constant_expressions = '${cvalues} 3'
      inputs = 'c n1 n2 n3 n4 lap_c lap_e1 lap_e2 lap_e3 lap_e4'
    []
    [bnds]
      type = ParsedCompute
      buffer = bnds
      expression = 'n1^2 + n2^2 + n3^2 + n4^2'
      inputs = 'n1 n2 n3 n4'
    []
  []
[]

[TensorSolver]
  type = ImplicitFDSolver
  buffer = 'c n1 n2 n3 n4'
  operators = 'dcdt bih_c; dn1dt lap_n1; dn2dt lap_n2; dn3dt lap_n3; dn4dt lap_n4'
  source = 'rhs_c rhs_n1 rhs_n2 rhs_n3 rhs_n4'
  root_compute = explicit
  substeps = ${substeps}
[]

[Postprocessors]
  [min_c]
    type = TensorExtremeValuePostprocessor
    buffer = c
    value_type = MIN
    execute_on = 'TIMESTEP_END'
  []
  [max_c]
    type = TensorExtremeValuePostprocessor
    buffer = c
    value_type = MAX
    execute_on = 'TIMESTEP_END'
  []
  [F]
    type = TensorIntegralPostprocessor
    buffer = F
  []
  # cumulative wall time of the run (s)
  [wall_time]
    type = PerfGraphData
    section_name = Root
    data_type = TOTAL
  []
[]

[Problem]
  type = TensorProblem
[]

[Executioner]
  type = Transient
  end_time = ${end_time}
  [TimeStepper]
    type = IterationAdaptiveDT
    growth_factor = 1.1
    dt = 0.001
  []
  dtmax = ${dtmax}
[]

[TensorOutputs]
  [xdmf]
    type = XDMFTensorOutput
    buffer = 'c bnds'
    enable_hdf5 = true
  []
[]

[Outputs]
  csv = true
  perf_graph = true
  execute_on = 'TIMESTEP_END'
[]
