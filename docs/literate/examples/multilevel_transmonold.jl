# # Multilevel Transmon

# In this example we will look at a multilevel transmon qubit with a Hamiltonian given by
#
# ```math
# \hat{H}(t) = -\frac{\delta}{2} \hat{n}(\hat{n} - 1) + u_1(t) (\hat{a} + \hat{a}^\dagger) + u_2(t) i (\hat{a} - \hat{a}^\dagger)
# ```
# where $\hat{n} = \hat{a}^\dagger \hat{a}$ is the number operator, $\hat{a}$ is the annihilation operator, $\delta$ is the anharmonicity, and $u_1(t)$ and $u_2(t)$ are control fields.
#
# We will use the following parameter values:
#
# ```math
# \begin{aligned}
# \delta &= 0.2 \text{ GHz}\\
# \abs{u_i(t)} &\leq 0.2 \text{ GHz}\\
# T_0 &= 10 \text{ ns}\\
# \end{aligned}
# ```
#
# For convenience, we have defined the `TransmonSystem` function in the `QuantumSystemTemplates` module, which returns a `QuantumSystem` object for a transmon qubit. We will use this function to define the system.

# ## Setting up the problem

# To begin, let's load the necessary packages, define the system parameters, and create a a `QuantumSystem` object using the `TransmonSystem` function.

using QuantumCollocation
using PiccoloQuantumObjects
using NamedTrajectories
using LinearAlgebra
using SparseArrays
using Random; Random.seed!(123)

using PiccoloPlots
using CairoMakie

## define the time parameters

T₀ = 10     # total time in ns
T = 50      # number of time steps
Δt = T₀ / T # time step

## define the system parameters
levels = 5
δ = 0.2

## add a bound to the controls
a_bound = 0.2

## create the system
sys = TransmonSystem(levels=levels, δ=δ)

## let's look at the parameters of the system
sys.params


# Since this is a multilevel transmon and we want to implement an, let's say, $X$ gate on the qubit subspace, i.e., the first two levels we can utilize the `EmbeddedOperator` type to define the target operator.

## define the target operator
op = EmbeddedOperator(:X, sys)

## show the full operator
op.operator |> sparse

# In this formulation, we also use a subspace identity as the initial state, which looks like

function get_subspace_identity(op::EmbeddedOperator)
    return embed(
        Matrix{ComplexF64}(I(length(op.subspace))),
        op.subspace,
        size(op)[1]
    )
end
get_subspace_identity(op) |> sparse

# We can then pass this embedded operator to the `UnitarySmoothPulseProblem` template to create the problem

## create the problem
prob = UnitarySmoothPulseProblem(sys, op, T, Δt; a_bound=a_bound)

## solve the problem
solve!(prob, max_iter=50)
#=
```julia
solve!(prob; max_iter=50)
```

```@raw html
<pre class="documenter-example-output"><code class="nohighlight hljs ansi">    initializing optimizer...
        applying constraint: timesteps all equal constraint
        applying constraint: initial value of Ũ⃗
        applying constraint: initial value of a
        applying constraint: final value of a
        applying constraint: bounds on a
        applying constraint: bounds on da
        applying constraint: bounds on dda
        applying constraint: bounds on Δt

******************************************************************************
This program contains Ipopt, a library for large-scale nonlinear optimization.
 Ipopt is released as open source code under the Eclipse Public License (EPL).
         For more information visit https://github.com/coin-or/Ipopt
******************************************************************************

This is Ipopt version 3.14.19, running with linear solver MUMPS 5.8.1.

Number of nonzeros in equality constraint Jacobian...:   130578
Number of nonzeros in inequality constraint Jacobian.:        0
Number of nonzeros in Lagrangian Hessian.............:    11223

Total number of variables............................:     2796
                     variables with only lower bounds:        0
                variables with lower and upper bounds:      246
                     variables with only upper bounds:        0
Total number of equality constraints.................:     2695
Total number of inequality constraints...............:        0
        inequality constraints with only lower bounds:        0
   inequality constraints with lower and upper bounds:        0
        inequality constraints with only upper bounds:        0

iter    objective    inf_pr   inf_du lg(mu)  ||d||  lg(rg) alpha_du alpha_pr  ls
   0  6.3299435e-04 9.98e-01 1.21e+01   0.0 0.00e+00    -  0.00e+00 0.00e+00   0
   1  2.0497214e+01 4.88e-01 3.61e+03  -0.6 1.02e+00   2.0 6.37e-01 5.00e-01h  2
   2  1.3491779e+01 1.94e-01 6.08e+03   0.0 9.85e-01   2.4 1.00e+00 5.99e-01h  1
   3  2.9422241e+00 1.38e-01 3.99e+03  -0.3 6.38e-01   2.9 1.00e+00 2.89e-01f  1
   4  3.0923996e+00 1.11e-01 4.06e+03  -1.0 5.09e-01   3.3 1.00e+00 1.96e-01h  1
   5  1.2619020e+01 7.22e-02 2.13e+03  -1.5 4.84e-01   2.8 1.00e+00 3.49e-01h  1
   6  2.0738356e+01 3.25e-02 1.77e+03  -2.4 3.03e-01   3.2 1.00e+00 5.49e-01h  1
   7  2.3369234e+01 2.01e-02 1.86e+03  -3.3 1.62e-01   2.7 1.00e+00 3.82e-01h  1
   8  2.3952415e+01 1.71e-02 1.59e+03  -4.0 9.06e-02   3.2 1.00e+00 1.50e-01h  1
   9  2.6857125e+01 7.48e-04 2.13e+03  -4.0 7.25e-02   2.7 1.00e+00 1.00e+00h  1
iter    objective    inf_pr   inf_du lg(mu)  ||d||  lg(rg) alpha_du alpha_pr  ls
  10  2.6127823e+01 2.30e-05 1.13e+01  -1.8 1.24e-02   2.2 9.89e-01 1.00e+00f  1
  11  2.4112236e+01 1.81e-04 1.92e+00  -3.0 3.46e-02   1.7 1.00e+00 1.00e+00f  1
  12  1.9434757e+01 1.10e-03 1.53e+00  -3.5 8.29e-02   1.3 1.00e+00 1.00e+00f  1
  13  1.3140688e+01 3.22e-03 2.15e+00  -2.8 1.33e-01   0.8 1.00e+00 1.00e+00f  1
  14  8.8917963e+00 5.23e-03 3.35e+00  -2.4 1.41e-01   0.3 9.93e-01 1.00e+00f  1
  15  4.9743382e+00 1.02e-01 1.65e+01  -1.1 1.87e+00  -0.2 6.52e-01 3.53e-01f  1
  16  2.4462880e+00 8.51e-03 8.80e+00  -1.7 2.06e-01   0.3 1.00e+00 1.00e+00h  1
  17  2.3411292e+00 9.99e-03 2.02e+02  -2.1 2.12e-01  -0.2 1.00e+00 1.00e+00h  1
  18  3.4427509e+00 3.89e-03 1.94e+02  -1.9 1.71e-01   0.2 1.00e+00 1.00e+00h  1
  19  2.7849793e+00 7.23e-05 1.49e+01  -3.3 2.13e-02   1.5 1.00e+00 1.00e+00h  1
iter    objective    inf_pr   inf_du lg(mu)  ||d||  lg(rg) alpha_du alpha_pr  ls
  20  2.7136927e+00 1.83e-05 1.22e-01  -4.0 1.06e-02   1.1 1.00e+00 1.00e+00h  1
  21  2.4136342e+00 1.23e-04 1.70e-01  -4.0 2.87e-02   0.6 1.00e+00 1.00e+00f  1
  22  1.6071670e+00 8.06e-04 1.25e+00  -4.0 6.87e-02   0.1 1.00e+00 1.00e+00f  1
  23  1.6204685e+00 6.56e-05 1.33e-01  -4.0 2.15e-02   0.5 1.00e+00 1.00e+00h  1
  24  8.4793547e-01 9.10e-04 1.93e+00  -4.0 5.98e-02   0.1 1.00e+00 1.00e+00f  1
  25  1.0833586e+00 5.82e-05 8.47e-02  -4.0 1.99e-02   0.5 1.00e+00 1.00e+00h  1
  26  1.0463530e+00 7.56e-06 5.74e-02  -4.0 7.08e-03   0.9 1.00e+00 1.00e+00h  1
  27  9.2878482e-01 6.25e-05 1.39e-01  -4.0 1.98e-02   0.4 1.00e+00 1.00e+00f  1
  28  8.9482594e-01 8.44e-06 5.36e-02  -4.0 7.44e-03   0.9 1.00e+00 1.00e+00h  1
  29  7.8556955e-01 6.96e-05 1.76e-01  -4.0 2.08e-02   0.4 1.00e+00 1.00e+00f  1
iter    objective    inf_pr   inf_du lg(mu)  ||d||  lg(rg) alpha_du alpha_pr  ls
  30  7.5566667e-01 9.24e-06 4.98e-02  -4.0 7.77e-03   0.8 1.00e+00 1.00e+00h  1
  31  6.5518011e-01 7.59e-05 2.03e-01  -4.0 2.16e-02   0.3 1.00e+00 1.00e+00f  1
  32  6.2953661e-01 9.87e-06 4.56e-02  -4.0 8.01e-03   0.8 1.00e+00 1.00e+00h  1
  33  5.3771420e-01 8.06e-05 2.37e-01  -4.0 2.23e-02   0.3 1.00e+00 1.00e+00f  1
  34  5.1689444e-01 1.02e-05 4.12e-02  -4.0 8.15e-03   0.7 1.00e+00 1.00e+00h  1
  35  4.3260003e-01 8.35e-05 2.91e-01  -4.0 2.27e-02   0.2 1.00e+00 1.00e+00f  1
  36  4.1787263e-01 1.02e-05 3.79e-02  -4.0 8.14e-03   0.7 1.00e+00 1.00e+00h  1
  37  3.3681225e-01 8.78e-05 3.95e-01  -4.0 2.29e-02   0.2 1.00e+00 1.00e+00f  1
  38  3.3214144e-01 9.79e-06 3.20e-02  -4.0 8.01e-03   0.6 1.00e+00 1.00e+00h  1
  39  2.2896514e-01 1.18e-04 6.96e-01  -4.0 2.36e-02   0.1 1.00e+00 1.00e+00f  1
iter    objective    inf_pr   inf_du lg(mu)  ||d||  lg(rg) alpha_du alpha_pr  ls
  40  2.5836915e-01 9.08e-06 2.82e-02  -4.0 7.92e-03   0.6 1.00e+00 1.00e+00h  1
  41  2.5270638e-01 1.22e-06 2.76e-02  -4.0 2.91e-03   1.0 1.00e+00 1.00e+00h  1
  42  2.3463229e-01 1.04e-05 4.34e-02  -4.0 8.23e-03   0.5 1.00e+00 1.00e+00h  1
  43  2.2893134e-01 1.47e-06 2.57e-02  -4.0 3.05e-03   0.9 1.00e+00 1.00e+00h  1
  44  2.1089191e-01 1.23e-05 5.95e-02  -4.0 8.57e-03   0.4 1.00e+00 1.00e+00h  1
  45  2.0548478e-01 1.77e-06 2.39e-02  -4.0 3.19e-03   0.9 1.00e+00 1.00e+00h  1
  46  1.8762248e-01 1.46e-05 7.53e-02  -4.0 8.84e-03   0.4 1.00e+00 1.00e+00h  1
  47  1.8261332e-01 2.10e-06 2.20e-02  -4.0 3.30e-03   0.8 1.00e+00 1.00e+00h  1
  48  1.6500170e-01 1.70e-05 9.38e-02  -4.0 9.03e-03   0.3 1.00e+00 1.00e+00h  1
  49  1.6056173e-01 2.46e-06 2.00e-02  -4.0 3.38e-03   0.8 1.00e+00 1.00e+00h  1
iter    objective    inf_pr   inf_du lg(mu)  ||d||  lg(rg) alpha_du alpha_pr  ls
  50  1.4312848e-01 1.93e-05 1.19e-01  -4.0 9.10e-03   0.3 1.00e+00 1.00e+00h  1

Number of Iterations....: 50

                                   (scaled)                 (unscaled)
Objective...............:   1.4312847869180015e-01    1.4312847869180015e-01
Dual infeasibility......:   1.1926413598927255e-01    1.1926413598927255e-01
Constraint violation....:   1.9342550412471127e-05    1.9342550412471127e-05
Variable bound violation:   0.0000000000000000e+00    0.0000000000000000e+00
Complementarity.........:   9.9999999971021043e-05    9.9999999971021043e-05
Overall NLP error.......:   1.1926413598927255e-01    1.1926413598927255e-01


Number of objective function evaluations             = 54
Number of objective gradient evaluations             = 51
Number of equality constraint evaluations            = 54
Number of inequality constraint evaluations          = 0
Number of equality constraint Jacobian evaluations   = 51
Number of inequality constraint Jacobian evaluations = 0
Number of Lagrangian Hessian evaluations             = 50
Total seconds in IPOPT                               = 367.274

EXIT: Maximum Number of Iterations Exceeded.
</code><button class="copy-button fa-solid fa-copy" aria-label="Copy this code block" title="Copy"></button></pre>
```
=#
save("multilevel_transmon_example_25e3be.jld2", prob.trajectory)
# load_path = joinpath(dirname(Base.active_project()), "data/multilevel_transmon_example_25e3be.jld2") # hide
# prob.trajectory = load_traj(load_path) # hide
# nothing # hide

# Let's look at the fidelity in the subspace

fid = unitary_rollout_fidelity(prob.trajectory, sys; subspace=op.subspace, drive_name=:a)
println("Fidelity: ", fid)
@assert fid > 0.99

# and plot the result using the `plot_unitary_populations` function.

plot_unitary_populations(prob.trajectory; fig_size=(900, 700), control_name=:a)

# ## Leakage suppresion
# As can be seen from the above plot, there is a substantial amount of leakage into the higher levels during the evolution. To mitigate this, we have implemented a constraint to avoid populating the leakage levels, which should ideally drive those leakage populations down to zero.
# To implement this, pass `leakage_constraint=true` and set `leakage_constraint_value={value}` and `leakage_cost={value}` to the `PiccoloOptions` instance passed to the `UnitarySmoothPulseProblem` template.

## create the a leakage suppression problem, initializing with the previous solution

prob_leakage = UnitarySmoothPulseProblem(sys, op, T, Δt;
    a_bound=a_bound,
    a_guess=prob.trajectory.a[:, :],
    piccolo_options=PiccoloOptions(
        leakage_constraint=true,
        leakage_constraint_value=1e-2,
        leakage_cost=1e-2,
    ),
)

## solve the problem
load_path = joinpath(dirname(Base.active_project()), "data/multilevel_transmon_example_leakage_25e3be.jld2") # hide
prob_leakage.trajectory = load_traj(load_path) # hide
nothing # hide

#=
```julia
solve!(prob_leakage; max_iter=250)
```

```@raw html
<pre class="documenter-example-output"><code class="nohighlight hljs ansi">    initializing optimizer...
        applying constraint: timesteps all equal constraint
        applying constraint: initial value of Ũ⃗
        applying constraint: initial value of a
        applying constraint: final value of a
        applying constraint: bounds on a
        applying constraint: bounds on da
        applying constraint: bounds on dda
        applying constraint: bounds on Δt
This is Ipopt version 3.14.19, running with linear solver MUMPS 5.8.1.

Number of nonzeros in equality constraint Jacobian...:   130578
Number of nonzeros in inequality constraint Jacobian.:    58800
Number of nonzeros in Lagrangian Hessian.............:   196198

Total number of variables............................:     2796
                     variables with only lower bounds:        0
                variables with lower and upper bounds:      246
                     variables with only upper bounds:        0
Total number of equality constraints.................:     2695
Total number of inequality constraints...............:     1200
        inequality constraints with only lower bounds:        0
   inequality constraints with lower and upper bounds:        0
        inequality constraints with only upper bounds:     1200

iter    objective    inf_pr   inf_du lg(mu)  ||d||  lg(rg) alpha_du alpha_pr  ls
   0  1.4950134e-01 1.85e-01 2.29e-01   0.0 0.00e+00    -  0.00e+00 0.00e+00   0
   1  5.0824652e-01 1.69e-01 1.96e+02  -1.4 7.25e-01   0.0 3.87e-01 1.23e-01h  1
   2  3.5258495e-01 1.54e-01 1.78e+02  -2.6 9.20e-01    -  1.15e-01 9.37e-02h  1
   3  2.4017237e-01 1.47e-01 1.70e+02  -4.0 1.52e+00    -  8.65e-02 4.27e-02h  1
   4  5.0460395e-02 1.34e-01 4.00e+01  -2.4 1.81e+00    -  9.54e-02 8.65e-02h  1
   5  1.7890116e-01 1.29e-01 3.82e+01  -1.8 3.37e+00    -  3.46e-02 4.02e-02h  1
   6  4.0032329e-01 1.24e-01 3.70e+01  -1.7 4.63e+00    -  3.89e-02 2.77e-02h  1
   7  7.2440848e-01 1.20e-01 3.60e+01  -1.5 1.73e+01    -  1.70e-02 2.38e-02f  1
   8  9.8420024e-01 1.19e-01 3.56e+01  -1.2 2.43e+01    -  1.14e-02 1.04e-02f  1
   9  1.4377510e+00 1.13e-01 3.40e+01  -1.2 5.32e+00    -  1.96e-02 4.60e-02f  1
iter    objective    inf_pr   inf_du lg(mu)  ||d||  lg(rg) alpha_du alpha_pr  ls
  10  2.1041138e+00 1.06e-01 3.46e+01  -1.2 2.30e+00    -  3.22e-02 6.45e-02f  1
  11  2.6999909e+00 1.00e-01 3.24e+01  -1.2 1.84e+00    -  6.19e-02 5.01e-02h  1
  12  3.8962937e+00 9.04e-02 3.59e+01  -1.1 1.82e+00    -  5.51e-02 9.60e-02h  1
  13  4.5923397e+00 8.07e-02 3.88e+01  -1.2 4.06e+00    -  1.24e-01 1.06e-01h  1
  14  7.0404696e+00 7.37e-02 4.82e+01  -0.3 3.21e+00    -  3.20e-02 5.40e-02f  1
  15  1.0716498e+01 5.84e-02 4.80e+01  -1.2 8.41e-01  -0.5 1.39e-01 2.37e-01h  1
  16  1.0796271e+01 5.75e-02 5.33e+01  -4.0 1.30e+00  -1.0 1.32e-01 1.61e-02h  1
  17  1.0758077e+01 5.00e-02 3.96e+01  -2.4 2.89e+00  -1.4 1.15e-01 1.31e-01h  1
  18  1.0750479e+01 4.74e-02 4.62e+01  -1.6 2.13e+00  -1.0 1.07e-01 4.75e-02h  1
  19  9.8188360e+00 4.66e-02 4.56e+01  -0.7 6.34e+01  -1.5 1.07e-02 1.07e-02f  1
iter    objective    inf_pr   inf_du lg(mu)  ||d||  lg(rg) alpha_du alpha_pr  ls
  20  9.8655313e+00 4.56e-02 6.89e+01  -4.0 2.38e+00  -1.1 2.79e-01 2.26e-02h  1
  21  1.5262766e+01 2.97e-02 1.26e+02  -1.4 1.38e+00  -0.6 2.32e-01 5.11e-01h  1
  22  1.5345194e+01 2.79e-02 1.20e+02  -4.0 7.86e-01  -1.1 2.67e-01 7.06e-02h  1
  23  1.5098998e+01 2.70e-02 1.24e+02  -1.9 4.61e+00  -1.6 2.68e-01 9.91e-02f  1
  24  1.5661951e+01 1.98e-02 9.29e+01  -4.0 7.13e-01  -1.2 5.67e-01 2.96e-01h  1
  25  1.8642805e+01 1.01e-02 5.04e+01  -1.6 3.78e-01  -0.7 8.45e-01 5.51e-01h  1
  26  2.1623972e+01 1.18e-02 1.00e+01  -1.4 4.91e-01  -1.2 6.85e-01 8.37e-01h  1
  27  2.0701492e+01 2.84e-02 4.29e+00  -1.7 1.62e+00  -1.7 5.92e-01 5.66e-01f  1
  28  2.0358386e+01 9.81e-03 8.29e+00  -1.7 6.52e-01  -1.3 6.64e-01 7.50e-01h  1
  29  1.9575393e+01 1.29e-02 5.86e+00  -4.0 1.39e+00  -1.7 3.13e-01 3.01e-01f  1
iter    objective    inf_pr   inf_du lg(mu)  ||d||  lg(rg) alpha_du alpha_pr  ls
  30  2.0949644e+01 3.60e-02 1.38e+01  -1.5 9.70e-01  -1.3 7.82e-01 1.00e+00f  1
  31  1.9665655e+01 5.46e-03 1.42e+00  -1.6 5.44e-01  -0.9 1.00e+00 1.00e+00h  1
  32  1.8921625e+01 2.05e-02 5.66e+00  -1.7 2.78e+00    -  3.09e-01 2.38e-01f  1
  33  1.9869663e+01 4.95e-02 5.19e+00  -1.4 9.77e-01  -1.4 9.48e-01 1.00e+00f  1
  34  1.8782254e+01 8.33e-02 4.54e+00  -1.5 1.13e+01  -1.8 1.06e-01 8.81e-02f  1
  35  1.8006443e+01 5.99e-02 1.12e+01  -1.5 1.90e+00  -1.4 5.89e-01 4.05e-01f  1
  36  1.6376032e+01 1.10e-02 1.28e+00  -1.7 6.36e-01  -1.0 9.98e-01 1.00e+00h  1
  37  1.6483636e+01 4.20e-02 1.44e+00  -1.5 1.69e+00    -  4.60e-01 4.64e-01f  1
  38  1.5397169e+01 3.19e-02 2.70e+00  -1.7 1.09e+00  -1.5 7.27e-01 8.50e-01f  1
  39  1.5084121e+01 3.54e-02 2.95e+01  -4.0 7.02e+00  -1.9 1.17e-01 4.19e-02f  1
iter    objective    inf_pr   inf_du lg(mu)  ||d||  lg(rg) alpha_du alpha_pr  ls
  40  1.4056991e+01 7.26e-02 1.53e+01  -1.9 2.39e+00  -1.5 4.40e-01 4.61e-01f  1
  41  1.6270302e+01 5.23e-02 2.84e+01  -1.0 2.21e+00  -2.0 2.21e-01 2.76e-01f  1
  42  1.4885369e+01 6.97e-02 2.56e+01  -1.5 1.83e+00  -1.6 1.61e-01 6.30e-01f  1
  43  1.4891578e+01 3.02e-02 1.76e+02  -1.5 8.85e-01  -1.1 1.41e-01 1.00e+00f  1
  44  1.3989306e+01 1.03e-02 4.48e+01  -1.6 5.00e-01  -1.6 1.00e+00 7.49e-01h  1
  45  1.7030165e+01 5.35e-02 3.22e+01  -0.8 4.08e+00  -2.1 2.74e-01 2.54e-01f  1
  46  1.4661680e+01 1.59e-01 1.52e+01  -1.3 4.11e+00  -1.7 4.24e-01 6.61e-01f  1
  47  1.5001321e+01 4.47e-02 3.99e+00  -1.3 1.33e+00  -1.2 1.00e+00 1.00e+00h  1
  48  1.4406515e+01 1.61e-02 5.42e+00  -1.3 1.04e+00    -  1.00e+00 1.00e+00h  1
  49  1.3303759e+01 9.99e-02 2.02e+01  -1.3 1.95e+00  -1.7 7.36e-01 1.00e+00f  1
iter    objective    inf_pr   inf_du lg(mu)  ||d||  lg(rg) alpha_du alpha_pr  ls
  50  1.3258639e+01 3.60e-02 1.67e+00  -1.3 1.37e+00  -1.3 1.00e+00 1.00e+00h  1
  51  1.0893455e+01 2.08e-02 1.46e+01  -1.8 1.01e+00    -  9.71e-01 6.04e-01f  1
  52  1.1281526e+01 1.35e-01 5.04e+00  -1.4 2.09e+00  -1.8 5.60e-01 6.98e-01f  1
  53  9.5854177e+00 4.05e-02 1.95e+00  -1.6 1.26e+00  -1.3 1.00e+00 1.00e+00h  1
  54  9.2755037e+00 4.43e-02 2.84e+00  -1.6 1.46e+00    -  5.53e-01 5.97e-01h  1
  55  9.1341181e+00 7.28e-02 1.57e+01  -1.3 5.97e+00  -1.8 3.09e-01 1.53e-01f  1
  56  8.0380719e+00 3.71e-02 1.91e+01  -2.1 8.82e-01  -1.4 1.00e+00 5.52e-01h  1
  57  7.8238845e+00 2.58e-02 1.44e+01  -1.8 2.12e+00  -1.9 2.49e-01 2.48e-01h  1
  58  6.8057837e+00 2.61e-02 1.40e+00  -2.0 7.42e-01  -1.4 1.00e+00 1.00e+00h  1
  59  6.7413190e+00 2.58e-02 1.01e+01  -1.5 4.74e+01    -  7.67e-03 3.35e-03f  2
iter    objective    inf_pr   inf_du lg(mu)  ||d||  lg(rg) alpha_du alpha_pr  ls
  60  6.5380671e+00 3.58e-02 7.32e+00  -2.0 2.56e+00  -1.9 1.75e-01 2.11e-01h  1
  61  6.4801158e+00 3.56e-02 8.94e+00  -4.0 2.98e+01  -2.4 9.42e-03 5.71e-03h  1
  62  6.3851709e+00 2.95e-03 5.20e-01  -2.1 3.23e-01  -1.1 9.99e-01 1.00e+00f  1
  63  6.0529438e+00 2.14e-02 6.89e-01  -3.2 3.60e+00    -  1.73e-01 1.67e-01f  1
  64  5.8240147e+00 2.03e-02 6.44e+00  -2.2 9.45e-01  -1.6 7.23e-01 5.12e-01h  1
  65  5.4832742e+00 1.19e-02 3.36e+00  -2.3 3.85e-01  -1.1 1.00e+00 8.41e-01h  1
  66  5.3219359e+00 1.49e-02 1.47e+01  -2.4 1.24e+00  -1.6 5.53e-01 2.38e-01h  1
  67  5.2153211e+00 1.63e-02 1.79e+01  -2.1 3.11e+00  -2.1 1.72e-01 1.16e-01f  1
  68  5.0405267e+00 2.21e-02 1.37e+01  -2.8 8.21e+00  -2.6 2.89e-02 6.25e-02f  1
  69  4.9219006e+00 2.06e-02 1.25e+01  -4.0 2.32e+00  -2.1 1.48e-01 9.10e-02h  1
iter    objective    inf_pr   inf_du lg(mu)  ||d||  lg(rg) alpha_du alpha_pr  ls
  70  4.8600210e+00 1.48e-02 1.23e+01  -2.4 7.82e-01  -1.7 4.36e-01 2.63e-01h  1
  71  4.6115393e+00 5.99e-03 3.44e-01  -2.7 2.75e-01  -1.3 1.00e+00 1.00e+00h  1
  72  4.5154096e+00 1.20e-02 1.40e+00  -2.8 1.94e+00  -1.8 2.76e-01 2.31e-01h  1
  73  4.6161788e+00 9.71e-03 1.47e+00  -2.3 4.46e-01  -1.3 9.01e-01 1.00e+00f  1
  74  4.4634690e+00 1.46e-02 8.40e+00  -2.5 1.34e+00  -1.8 4.91e-01 2.36e-01h  1
  75  4.2876254e+00 9.54e-03 9.51e+00  -2.8 3.23e-01  -1.4 1.00e+00 4.21e-01h  1
  76  4.1391518e+00 7.61e-03 4.72e+00  -4.0 1.14e+00  -1.9 9.75e-02 2.13e-01h  1
  77  4.1593861e+00 6.11e-03 3.41e+00  -2.4 4.07e-01  -1.4 5.28e-01 3.51e-01h  1
  78  4.0895377e+00 7.12e-03 3.12e+00  -4.0 2.06e+00  -1.9 9.47e-02 1.10e-01h  1
  79  4.1649214e+00 1.08e-02 6.69e+00  -2.1 1.02e+00  -1.5 4.70e-01 3.00e-01f  1
iter    objective    inf_pr   inf_du lg(mu)  ||d||  lg(rg) alpha_du alpha_pr  ls
  80  4.0760901e+00 1.52e-02 6.24e+00  -2.4 4.29e+00  -2.0 6.65e-02 6.72e-02h  1
  81  3.8250540e+00 1.56e-02 4.38e+00  -2.9 7.19e-01  -1.5 6.13e-01 4.43e-01h  1
  82  3.7793156e+00 1.45e-02 3.98e+00  -2.2 6.08e+00  -2.0 7.97e-02 7.44e-02h  1
  83  3.5840802e+00 1.02e-02 3.12e+00  -4.0 6.74e-01  -1.6 3.94e-01 4.67e-01h  1
  84  3.5364298e+00 1.34e-02 3.09e+00  -2.1 4.10e+01  -2.1 6.85e-03 6.92e-03f  1
  85  3.6069081e+00 1.17e-02 1.70e+00  -2.2 1.52e+00  -1.6 4.51e-01 4.50e-01f  1
  86  3.4389774e+00 5.08e-03 2.54e-01  -2.5 2.78e-01  -1.2 1.00e+00 1.00e+00h  1
  87  3.3151710e+00 1.07e-02 9.77e+00  -2.5 6.17e-01  -1.7 8.84e-01 5.01e-01h  1
  88  3.1987678e+00 1.08e-02 6.44e+00  -4.0 2.37e+00  -2.2 8.38e-02 1.44e-01h  1
  89  3.3174012e+00 2.26e-02 1.55e+01  -0.7 4.27e+01  -2.6 5.05e-03 1.12e-02f  1
iter    objective    inf_pr   inf_du lg(mu)  ||d||  lg(rg) alpha_du alpha_pr  ls
  90  3.2347348e+00 2.06e-02 1.17e+01  -2.5 1.70e+00  -2.2 1.48e-01 1.26e-01h  1
  91  3.1265478e+00 1.61e-02 8.96e+00  -2.5 1.33e+00  -1.8 2.00e-01 2.37e-01h  1
  92  2.9645703e+00 1.10e-02 4.33e+00  -2.5 4.34e-01  -1.4 1.00e+00 7.71e-01f  1
  93  3.4576453e+00 1.38e-02 4.15e+00  -1.4 3.60e+00    -  1.54e-01 1.58e-01f  1
  94  3.2084310e+00 1.58e-02 2.28e+00  -2.1 1.26e+00  -1.8 3.79e-01 4.29e-01h  1
  95  3.0179091e+00 2.08e-02 2.28e+01  -2.1 4.75e-01  -1.4 5.53e-01 1.00e+00h  1
  96  2.9235911e+00 1.54e-02 1.38e+01  -2.1 6.75e-01  -1.9 6.84e-01 4.76e-01h  1
  97  2.8304039e+00 1.72e-02 1.26e+01  -2.9 6.35e+00    -  7.36e-02 6.36e-02h  1
  98  2.5743595e+00 5.49e-03 2.55e-01  -2.4 3.58e-01  -1.5 1.00e+00 1.00e+00h  1
  99  2.4504387e+00 2.38e-02 6.56e-01  -2.4 1.31e+00  -1.9 3.90e-01 3.63e-01h  1
iter    objective    inf_pr   inf_du lg(mu)  ||d||  lg(rg) alpha_du alpha_pr  ls
 100  2.2689192e+00 1.53e-02 8.46e-01  -2.4 5.12e-01  -1.5 9.84e-01 9.32e-01h  1
 101  2.1632226e+00 2.34e-02 4.90e+00  -2.4 1.59e+00  -2.0 4.75e-01 2.90e-01h  1
 102  2.0690826e+00 2.35e-02 4.54e+00  -3.2 5.93e+00  -2.5 8.75e-02 7.47e-02h  1
 103  1.9445123e+00 2.26e-02 3.39e+00  -4.0 1.03e+00  -2.0 2.44e-01 2.53e-01h  1
 104  1.8550110e+00 6.42e-03 3.57e+00  -2.9 2.17e-01  -1.6 5.66e-01 6.61e-01h  1
 105  1.8631009e+00 2.27e-02 5.11e+00  -2.0 9.12e+00  -2.1 8.29e-02 5.97e-02f  1
 106  1.8034826e+00 1.41e-02 2.26e+00  -2.7 5.19e-01  -1.7 7.98e-01 6.70e-01h  1
 107  1.6976508e+00 5.51e-03 9.18e+00  -4.0 2.79e-01  -2.1 4.21e-01 6.03e-01h  1
 108  1.6475565e+00 2.61e-03 2.07e+00  -3.5 2.46e-01  -1.7 8.80e-01 7.94e-01h  1
 109  1.6307314e+00 6.92e-04 3.36e-02  -3.5 1.20e-01  -1.3 9.99e-01 1.00e+00h  1
iter    objective    inf_pr   inf_du lg(mu)  ||d||  lg(rg) alpha_du alpha_pr  ls
 110  1.6108925e+00 2.23e-03 8.79e-02  -4.0 7.34e-01  -1.8 3.44e-01 3.11e-01h  1
 111  1.5903064e+00 1.50e-03 5.45e-01  -4.0 1.96e-01  -1.3 9.99e-01 6.31e-01h  1
 112  1.5783713e+00 2.01e-04 3.29e-02  -4.0 4.20e-02  -0.9 1.00e+00 9.54e-01h  1
 113  1.5669420e+00 2.06e-03 4.98e-02  -3.7 1.33e-01  -1.4 9.90e-01 1.00e+00h  1
 114  1.5489283e+00 3.02e-04 9.23e-03  -4.0 5.31e-02  -1.0 1.00e+00 1.00e+00h  1
 115  1.5399325e+00 5.83e-04 2.75e+00  -3.7 2.31e-01  -1.4 1.00e+00 2.65e-01h  1
 116  1.5246440e+00 3.36e-04 1.91e-02  -4.0 6.06e-02  -1.0 1.00e+00 1.00e+00h  1
 117  1.5112277e+00 1.28e-03 2.72e+00  -3.4 3.12e-01  -1.5 7.95e-01 3.53e-01h  1
 118  1.4967242e+00 1.90e-03 3.21e+00  -2.8 4.19e+01  -2.0 4.32e-03 3.21e-03f  1
 119  1.5261752e+00 1.56e-03 3.40e+00  -2.9 3.18e-01  -1.6 5.81e-01 4.40e-01f  1
iter    objective    inf_pr   inf_du lg(mu)  ||d||  lg(rg) alpha_du alpha_pr  ls
 120  1.5074801e+00 2.89e-03 1.02e+01  -3.1 1.27e+00  -2.0 4.12e-01 1.45e-01h  1
 121  1.4666454e+00 8.72e-03 2.49e+00  -3.1 3.51e-01  -1.6 8.10e-01 1.00e+00f  1
 122  1.4166400e+00 1.30e-02 2.49e+00  -4.0 1.96e+00  -2.1 2.12e-01 1.94e-01h  1
 123  1.3800609e+00 1.23e-02 1.75e+00  -3.9 7.02e-01  -1.7 1.96e-01 2.27e-01h  1
 124  1.3287553e+00 4.96e-03 1.16e+00  -3.5 1.92e-01  -1.2 1.00e+00 7.12e-01h  1
 125  1.4449368e+00 6.57e-03 7.88e+00  -2.1 2.22e+00  -1.7 3.28e-01 1.84e-01f  1
 126  1.3586007e+00 2.71e-02 7.09e+00  -2.7 3.59e+00  -2.2 1.36e-01 1.53e-01h  1
 127  1.3408499e+00 2.62e-02 5.06e+00  -2.7 2.76e+00    -  8.89e-02 3.63e-02h  1
 128  1.3205208e+00 2.50e-02 4.58e+00  -2.7 1.24e+01    -  3.88e-02 2.85e-02h  1
 129  1.2486748e+00 1.68e-02 3.25e+00  -4.0 2.33e+00    -  1.33e-01 1.79e-01h  1
iter    objective    inf_pr   inf_du lg(mu)  ||d||  lg(rg) alpha_du alpha_pr  ls
 130  1.2034054e+00 1.25e-02 2.76e+00  -4.0 1.59e+00    -  1.64e-01 1.57e-01h  1
 131  1.1740838e+00 1.27e-02 2.39e+00  -3.3 2.49e+00  -2.7 1.09e-01 1.28e-01h  1
 132  1.1324880e+00 1.59e-02 3.21e+00  -3.8 2.96e+00  -2.2 3.97e-02 1.54e-01h  1
 133  1.1184144e+00 1.37e-02 2.28e+00  -3.5 5.31e-01  -1.8 5.89e-01 1.36e-01h  1
 134  1.0498719e+00 1.53e-03 5.64e-01  -3.7 1.96e-01  -1.4 1.00e+00 8.90e-01h  1
 135  1.0675790e+00 1.40e-02 1.11e+00  -2.8 1.05e+00  -1.9 6.71e-01 5.95e-01f  1
 136  1.0199401e+00 1.92e-02 2.75e+00  -3.1 2.95e+00  -2.3 5.13e-02 1.24e-01h  1
 137  9.8063680e-01 1.84e-02 1.61e+00  -4.0 1.39e+00  -1.9 1.88e-01 1.45e-01h  1
 138  9.0828734e-01 7.87e-03 7.96e-01  -4.0 2.31e-01  -1.5 6.69e-01 5.85e-01h  1
 139  8.9870687e-01 7.95e-03 7.77e-01  -4.0 3.16e+01  -2.0 5.15e-03 5.23e-03h  1
iter    objective    inf_pr   inf_du lg(mu)  ||d||  lg(rg) alpha_du alpha_pr  ls
 140  8.9317084e-01 6.09e-03 2.13e+00  -3.0 5.63e-01  -1.5 9.98e-01 5.91e-01h  1
 141  8.6599961e-01 8.62e-03 4.77e+00  -3.1 1.53e+00  -2.0 2.87e-01 1.67e-01h  1
 142  8.3647667e-01 7.99e-03 4.14e+00  -4.0 1.86e+00    -  4.44e-02 1.36e-01h  1
 143  8.1861058e-01 5.29e-03 5.72e+00  -3.1 2.07e-01  -1.6 6.15e-01 1.00e+00f  1
 144  8.1742517e-01 7.15e-03 5.71e+00  -2.5 4.02e+00    -  1.48e-01 1.09e-01f  1
 145  7.9300004e-01 5.78e-03 4.97e+00  -4.0 1.55e+00    -  3.77e-02 1.40e-01h  1
 146  8.3341918e-01 7.55e-03 3.84e+00  -2.7 3.83e-01  -2.1 4.42e-01 5.69e-01f  1
 147  7.9734410e-01 5.90e-03 3.83e+00  -2.9 1.40e+00    -  1.60e-01 2.55e-01h  1
 148  7.6707696e-01 1.71e-02 3.43e+00  -2.9 4.97e+00  -2.5 9.80e-02 9.40e-02h  1
 149  7.3612001e-01 1.84e-02 2.27e+00  -2.8 7.32e-01  -2.1 7.07e-01 5.42e-01f  1
iter    objective    inf_pr   inf_du lg(mu)  ||d||  lg(rg) alpha_du alpha_pr  ls
 150  7.3513738e-01 1.85e-02 8.65e+00  -2.5 4.32e+00  -2.6 9.99e-02 2.17e-02h  1
 151  6.8290870e-01 1.41e-02 5.30e+00  -3.6 3.65e-01  -2.2 4.23e-01 3.71e-01h  1
 152  6.1909257e-01 4.39e-03 2.76e-01  -3.5 1.82e-01  -1.7 1.00e+00 8.31e-01h  1
 153  5.9451308e-01 4.59e-03 4.95e-01  -4.0 1.08e+00  -2.2 2.39e-01 3.40e-01h  1
 154  5.7692482e-01 4.12e-03 1.41e+00  -3.7 2.12e-01  -1.8 1.00e+00 6.22e-01h  1
 155  5.7663186e-01 4.30e-03 5.82e+00  -2.9 1.03e+00  -2.3 4.58e-01 1.37e-01h  1
 156  5.5635097e-01 2.89e-03 2.58e+00  -3.8 2.77e-01  -1.8 8.44e-01 6.52e-01h  1
 157  5.3697552e-01 8.40e-04 2.87e-02  -4.0 7.24e-02  -1.4 1.00e+00 1.00e+00h  1
 158  5.3356552e-01 1.53e-04 1.24e-02  -4.0 3.98e-02  -1.0 1.00e+00 1.00e+00h  1
 159  5.2459315e-01 1.77e-03 2.24e-01  -4.0 2.11e-01  -1.5 1.00e+00 7.37e-01h  1
iter    objective    inf_pr   inf_du lg(mu)  ||d||  lg(rg) alpha_du alpha_pr  ls
 160  5.2117304e-01 2.04e-03 1.02e+00  -4.0 2.99e+00  -1.9 5.90e-02 2.86e-02h  1
 161  5.2202500e-01 2.23e-03 5.58e+00  -3.4 4.46e-01  -1.5 8.84e-01 2.84e-01f  1
 162  5.1517931e-01 2.49e-03 6.51e+00  -3.5 4.06e+00  -2.0 1.17e-01 4.29e-02h  1
 163  4.9132399e-01 3.27e-03 1.86e+00  -3.8 4.15e-01  -1.6 9.24e-01 6.88e-01f  1
 164  4.7567401e-01 4.41e-03 1.76e+00  -4.0 2.73e+00  -2.0 1.21e-01 1.07e-01h  1
 165  4.7623353e-01 3.54e-03 3.19e+00  -3.2 2.84e-01  -1.6 9.11e-01 2.31e-01h  1
 166  4.6430303e-01 3.04e-03 1.55e+00  -4.0 4.59e-01  -2.1 2.52e-01 1.52e-01h  1
 167  4.3978139e-01 2.79e-03 1.63e+00  -4.0 1.55e+00  -2.6 6.42e-02 1.31e-01h  1
 168  4.1020998e-01 5.69e-03 1.25e+00  -4.0 1.11e+00  -2.1 4.44e-01 3.30e-01h  1
 169  3.8978815e-01 3.35e-03 1.15e+00  -3.6 3.08e-01  -1.7 9.47e-01 7.50e-01h  1
iter    objective    inf_pr   inf_du lg(mu)  ||d||  lg(rg) alpha_du alpha_pr  ls
 170  3.6252698e-01 6.37e-03 9.45e-01  -4.0 2.01e+00  -2.2 1.40e-01 2.17e-01h  1
 171  3.4943080e-01 4.97e-03 2.56e+00  -4.0 4.51e-01  -1.8 5.54e-01 2.77e-01h  1
 172  3.4160681e-01 2.11e-03 1.71e+00  -3.8 1.04e-01  -1.3 1.00e+00 5.62e-01h  1
 173  3.4704460e-01 1.43e-03 1.60e+00  -3.2 3.38e-01  -1.8 3.71e-01 3.37e-01h  1
 174  3.4293738e-01 1.40e-03 5.23e+00  -3.2 8.00e-01  -2.3 4.65e-01 1.44e-01h  1
 175  3.2601445e-01 1.11e-03 2.30e+00  -4.0 2.84e-01  -1.9 8.27e-01 4.79e-01h  1
 176  3.5822389e-01 5.80e-03 1.77e+00  -2.6 2.46e+00  -2.4 1.97e-01 2.07e-01f  1
 177  3.5087149e-01 6.03e-03 3.07e+00  -3.3 3.36e+01  -2.8 2.83e-03 7.63e-03h  1
 178  3.4138701e-01 8.66e-03 2.26e+00  -3.3 2.01e+00  -2.4 1.30e-01 1.15e-01h  1
 179  3.2162557e-01 7.82e-03 1.08e+00  -3.3 4.29e-01  -2.0 7.82e-01 6.52e-01f  1
iter    objective    inf_pr   inf_du lg(mu)  ||d||  lg(rg) alpha_du alpha_pr  ls
 180  3.1832376e-01 8.10e-03 1.80e+00  -4.0 1.57e+01  -2.5 1.44e-02 8.43e-03h  1
 181  2.8092904e-01 1.42e-03 8.39e+00  -4.0 1.99e-01  -2.0 3.94e-01 8.68e-01h  1
 182  2.7381242e-01 7.21e-04 6.48e-02  -4.0 9.71e-02  -1.6 1.00e+00 1.00e+00h  1
 183  2.7559741e-01 2.02e-03 9.23e-01  -3.3 9.45e-01  -2.1 5.46e-01 1.57e-01h  1
 184  2.7595600e-01 3.06e-03 4.60e-02  -3.6 1.97e-01  -1.7 1.00e+00 1.00e+00h  1
 185  2.6099605e-01 1.04e-02 2.97e-01  -3.6 7.90e-01  -2.1 4.88e-01 4.24e-01h  1
 186  2.4967995e-01 6.82e-03 8.56e-01  -4.0 1.67e-01  -1.7 1.00e+00 3.97e-01h  1
 187  2.4219269e-01 6.62e-03 2.35e+00  -3.6 9.83e-01  -2.2 4.94e-01 3.63e-01h  1
 188  2.2154620e-01 3.60e-03 6.37e-02  -4.0 2.19e-01  -1.8 1.00e+00 1.00e+00h  1
 189  2.1882135e-01 1.10e-02 1.65e-01  -3.1 1.89e+00  -2.2 2.15e-01 2.09e-01h  1
iter    objective    inf_pr   inf_du lg(mu)  ||d||  lg(rg) alpha_du alpha_pr  ls
 190  2.0667735e-01 1.32e-02 5.01e-01  -2.4 3.59e+01  -2.7 1.52e-02 9.48e-03f  1
 191  1.9466462e-01 1.17e-02 2.63e+00  -3.2 8.66e-01  -2.3 9.56e-01 2.87e-01h  1
 192  1.9343140e-01 1.16e-02 2.67e+00  -4.0 1.85e+01    -  2.49e-03 1.04e-02h  1
 193  1.8608520e-01 1.06e-02 3.82e+00  -4.0 8.02e-01    -  2.30e-01 8.60e-02h  1
 194  1.6823635e-01 2.02e-02 3.44e+00  -3.4 4.68e+00    -  1.16e-01 1.02e-01h  1
 195  1.5986621e-01 1.68e-02 2.82e+00  -3.5 1.87e+00  -2.8 2.88e-01 1.69e-01h  1
 196  1.5718617e-01 1.61e-02 3.22e+00  -4.0 4.82e+00  -3.2 9.45e-02 4.23e-02h  1
 197  1.5508057e-01 1.35e-02 2.24e+00  -3.5 1.45e+00  -2.8 3.68e-01 1.60e-01h  1
 198  1.4981020e-01 1.41e-02 1.94e+00  -4.0 1.51e+00    -  6.01e-02 1.27e-01h  1
 199  1.3837356e-01 3.83e-03 2.17e-01  -3.7 2.37e-01  -2.4 1.00e+00 9.31e-01h  1
iter    objective    inf_pr   inf_du lg(mu)  ||d||  lg(rg) alpha_du alpha_pr  ls
 200  1.2945490e-01 4.56e-04 4.55e-02  -3.9 7.37e-02  -2.0 9.99e-01 1.00e+00h  1
 201  1.3699770e-01 4.29e-03 1.09e+00  -3.4 7.14e-01  -2.4 5.93e-01 3.64e-01h  1
 202  1.3403040e-01 3.57e-03 2.76e+00  -3.6 1.05e+00    -  8.62e-02 1.77e-01h  1
 203  1.3109015e-01 9.25e-04 5.15e-02  -3.6 1.66e-01  -2.0 1.00e+00 1.00e+00f  1
 204  1.2327610e-01 5.25e-03 1.06e+00  -3.6 7.96e-01  -2.5 2.10e-01 4.80e-01h  1
 205  1.2412502e-01 4.91e-03 2.83e+00  -3.0 1.43e+00    -  2.14e-01 1.01e-01f  1
 206  1.1676102e-01 2.00e-02 2.33e+00  -3.5 5.02e+00  -3.0 1.50e-02 7.94e-02h  1
 207  1.0166567e-01 1.90e-02 1.26e+00  -4.0 7.22e-01  -2.5 5.48e-01 3.67e-01h  1
 208  9.2833890e-02 3.84e-03 6.17e-02  -3.6 2.19e-01  -2.1 9.97e-01 1.00e+00h  1
 209  8.9518252e-02 1.21e-02 3.67e-01  -3.4 9.87e-01  -2.6 5.51e-01 4.26e-01h  1
iter    objective    inf_pr   inf_du lg(mu)  ||d||  lg(rg) alpha_du alpha_pr  ls
 210  7.1285927e-02 7.83e-03 5.40e-02  -3.7 2.83e-01  -2.2 1.00e+00 1.00e+00h  1
 211  6.1269420e-02 9.30e-03 2.31e-01  -4.0 9.54e-01  -2.6 2.82e-01 2.06e-01h  1
 212  5.5685810e-02 6.39e-03 1.54e+00  -3.7 4.06e-01  -2.2 1.00e+00 3.86e-01h  1
 213  5.3021777e-02 7.99e-03 1.23e+00  -4.0 9.48e+00    -  2.82e-02 4.39e-02h  1
 214  4.5978236e-02 1.63e-02 7.22e-01  -4.0 3.30e+00    -  4.39e-02 1.28e-01h  1
 215  3.7020024e-02 9.44e-03 6.97e-01  -4.0 5.67e+00    -  4.07e-02 6.03e-02h  1
 216  3.2387282e-02 1.09e-02 6.66e-01  -4.0 1.76e+01    -  1.95e-02 1.45e-02h  1
 217  2.8367819e-02 8.52e-03 1.50e+00  -3.8 4.80e-01  -2.7 8.07e-01 3.25e-01h  1
 218  2.5623622e-02 1.24e-02 1.31e+00  -4.0 1.04e+01    -  4.21e-03 2.84e-02h  1
 219  2.1004861e-02 1.55e-02 8.56e-01  -4.0 8.68e-01  -3.2 2.36e-01 3.59e-01h  1
iter    objective    inf_pr   inf_du lg(mu)  ||d||  lg(rg) alpha_du alpha_pr  ls
 220  1.8653237e-02 1.42e-02 9.53e-01  -4.0 2.70e+00  -3.6 6.23e-02 1.37e-01h  1
 221  1.5748032e-02 2.54e-02 1.36e+00  -4.0 1.45e+00  -3.2 1.32e-01 3.67e-01h  1
 222  1.8224927e-02 1.70e-02 6.66e-01  -3.5 5.28e-01  -2.8 6.46e-01 5.34e-01h  1
 223  8.5550831e-03 1.73e-02 7.67e-01  -4.0 6.05e-01  -3.3 4.95e-01 7.92e-01h  1
 224  8.4523103e-03 1.70e-02 6.99e-01  -4.0 1.33e+00  -3.8 1.51e-01 2.95e-01h  1
 225  7.7031464e-03 3.69e-02 8.02e-01  -4.0 1.25e+01  -4.2 4.43e-02 5.61e-02h  1
 226  1.0242195e-02 2.54e-02 1.98e+02  -4.0 1.32e+00    -  1.98e-01 2.93e-01h  1
 227  5.8870294e-03 2.47e-02 4.65e+00  -3.9 1.89e+01  -2.9 1.72e-02 2.35e-02h  1
 228  9.7611305e-03 1.01e-02 1.97e+02  -4.0 2.59e-01  -2.5 4.35e-01 6.46e-01h  1
 229  5.1713891e-03 9.06e-03 1.99e+01  -4.0 3.47e+00    -  1.02e-01 9.26e-02h  1
iter    objective    inf_pr   inf_du lg(mu)  ||d||  lg(rg) alpha_du alpha_pr  ls
 230  4.9177645e-03 5.62e-03 1.87e+02  -4.0 3.75e-01  -2.0 4.53e-01 3.95e-01h  1
 231  8.7283730e-03 5.20e-03 3.65e+01  -2.9 1.54e+00  -2.5 1.00e+00 1.31e-01h  1
 232  3.6487763e-03 2.46e-02 1.99e+02  -3.0 6.91e-01    -  5.41e-01 1.00e+00h  1
 233  1.0342111e-02 2.17e-02 1.40e+01  -2.7 5.12e+00  -3.0 1.09e-01 7.04e-02H  1
 234  1.4242643e-02 2.05e-02 1.14e+01  -3.0 2.93e+00    -  2.39e-01 1.81e-01h  2
 235  2.7907144e-02 1.79e-02 1.03e+01  -3.0 1.84e+00  -3.5 1.51e-01 1.01e-01H  1
 236  2.4785154e-02 2.53e-02 9.02e+00  -3.0 1.61e+00  -3.1 2.68e-01 1.24e-01H  1
 237  2.2369582e-02 2.36e-02 8.11e+00  -3.0 3.89e+00    -  5.40e-02 1.01e-01h  1
 238  1.3371702e-02 1.98e-02 4.05e+00  -3.0 1.01e+00  -3.5 2.43e-01 5.00e-01f  2
 239  1.6879573e-02 6.57e-02 3.95e+00  -3.0 4.29e+01    -  1.78e-02 2.35e-02h  2
iter    objective    inf_pr   inf_du lg(mu)  ||d||  lg(rg) alpha_du alpha_pr  ls
 240  1.1879330e-02 2.64e-02 2.12e+00  -3.0 7.66e-01  -2.2 7.33e-01 6.14e-01h  1
 241  4.8020537e-03 1.40e-02 1.09e+00  -3.0 1.05e+00  -2.7 7.59e-01 4.87e-01h  1
 242  1.0127074e-02 7.32e-03 1.99e+02  -3.5 3.42e-01    -  1.00e+00 1.00e+00h  1
 243  4.8797143e-03 4.07e-02 1.40e+02  -4.0 1.38e+00    -  4.17e-01 7.05e-01h  1
 244  3.1555319e-03 2.19e-02 7.40e+01  -3.7 4.09e-01   1.4 1.38e-01 4.71e-01h  1
 245  2.6470030e-02 4.42e-03 1.84e+02  -4.0 2.33e-01   0.9 8.82e-02 8.03e-01h  1
 246  1.3520190e-02 1.49e-04 1.99e+02  -3.8 4.66e-02   0.4 7.77e-01 1.00e+00h  1
 247  5.8888308e-03 3.32e-05 1.99e+02  -3.0 9.21e-03   1.7 1.00e+00 1.00e+00h  1
 248  1.4234003e-02 1.99e-06 1.99e+02  -3.2 2.89e-03   1.3 1.00e+00 1.00e+00h  1
 249  9.8726185e-03 1.40e-06 2.45e+00  -4.0 2.56e-03   1.7 1.00e+00 1.00e+00h  1
iter    objective    inf_pr   inf_du lg(mu)  ||d||  lg(rg) alpha_du alpha_pr  ls
 250  6.4249739e-03 2.64e-06 4.21e-02  -4.0 2.62e-03   1.2 1.00e+00 1.00e+00h  1

Number of Iterations....: 250

                                   (scaled)                 (unscaled)
Objective...............:   6.4249738858204513e-03    6.4249738858204513e-03
Dual infeasibility......:   4.2110842576720382e-02    4.2110842576720382e-02
Constraint violation....:   2.6385807075067231e-06    2.6385807075067231e-06
Variable bound violation:   0.0000000000000000e+00    0.0000000000000000e+00
Complementarity.........:   1.0381548778164065e-04    1.0381548778164065e-04
Overall NLP error.......:   4.2110842576720382e-02    4.2110842576720382e-02


Number of objective function evaluations             = 264
Number of objective gradient evaluations             = 251
Number of equality constraint evaluations            = 264
Number of inequality constraint evaluations          = 264
Number of equality constraint Jacobian evaluations   = 251
Number of inequality constraint Jacobian evaluations = 251
Number of Lagrangian Hessian evaluations             = 250
Total seconds in IPOPT                               = 1997.776

EXIT: Maximum Number of Iterations Exceeded.
</code><button class="copy-button fa-solid fa-copy" aria-label="Copy this code block" title="Copy"></button></pre>
```
=#

# Let's look at the fidelity in the subspace

fid_leakage = unitary_rollout_fidelity(prob_leakage.trajectory, sys; subspace=op.subspace, drive_name=:a)
println("Fidelity: ", fid_leakage)
@assert fid_leakage > 0.99

# and plot the result using the `plot_unitary_populations` function.

plot_unitary_populations(prob_leakage.trajectory; fig_size=(900, 700), control_name=:a)

# Here we can see that the leakage populations have been driven substantially down.
