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

T₀ = 10.0   # total time in ns
N = 50      # number of time steps
Δt = T₀ / N # time step

## define the system parameters
levels = 5
δ = 0.2

## add a bound to the controls
u_bounds = [0.2, 0.2]

## create the system
sys = TransmonSystem(levels=levels, δ=δ, T_max=T₀, u_bounds=u_bounds)

## let's look at a drive
get_drives(sys)[1] |> sparse


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
prob = UnitarySmoothPulseProblem(sys, op, N, Δt)

## solve the problem
load_path = joinpath(dirname(Base.active_project()), "data/multilevel_transmon_example_89ee72.jld2") # hide
prob.trajectory = load_traj(load_path) # hide
nothing # hide

#=
```julia
solve!(prob; max_iter=50)
```

```@raw html
<pre class="documenter-example-output"><code class="nohighlight hljs ansi">    initializing optimizer...
        applying constraint: timesteps all equal constraint
        applying constraint: initial value of Ũ⃗
        applying constraint: initial value of u
        applying constraint: final value of u
        applying constraint: bounds on u
        applying constraint: bounds on du
        applying constraint: bounds on ddu
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
   1  1.7461331e+01 4.87e-01 3.66e+03  -0.6 1.02e+00   2.0 6.32e-01 5.00e-01h  2
   2  1.1690187e+01 1.94e-01 6.11e+03   0.0 9.75e-01   2.4 1.00e+00 6.00e-01h  1
   3  1.0956380e+00 1.36e-01 4.05e+03  -0.3 6.45e-01   2.9 1.00e+00 3.00e-01f  1
   4  3.9110348e+00 1.13e-01 3.98e+03  -1.0 5.07e-01   3.3 1.00e+00 1.68e-01h  1

< ...snip... >

  45  3.3045607e-01 3.45e-06 9.48e-02  -4.0 4.85e-03   0.9 1.00e+00 1.00e+00f  1
  46  2.9815119e-01 2.50e-05 8.55e-02  -4.0 1.35e-02   0.4 1.00e+00 1.00e+00h  1
  47  2.8948071e-01 3.32e-06 3.25e-02  -4.1 4.88e-03   0.8 1.00e+00 1.00e+00h  1
  48  2.5998426e-01 2.55e-05 6.31e-02  -4.0 1.35e-02   0.3 1.00e+00 1.00e+00h  1
  49  2.5126499e-01 3.58e-06 2.93e-02  -4.1 4.94e-03   0.8 1.00e+00 1.00e+00h  1
iter    objective    inf_pr   inf_du lg(mu)  ||d||  lg(rg) alpha_du alpha_pr  ls
  50  2.2337171e-01 2.87e-05 8.86e-02  -4.0 1.34e-02   0.3 1.00e+00 1.00e+00h  1

Number of Iterations....: 50

                                   (scaled)                 (unscaled)
Objective...............:   2.2337171204275508e-01    2.2337171204275508e-01
Dual infeasibility......:   8.8610846418504252e-02    8.8610846418504252e-02
Constraint violation....:   2.8712906816219519e-05    2.8712906816219519e-05
Variable bound violation:   0.0000000000000000e+00    0.0000000000000000e+00
Complementarity.........:   1.0048178594192366e-04    1.0048178594192366e-04
Overall NLP error.......:   8.8610846418504252e-02    8.8610846418504252e-02


Number of objective function evaluations             = 55
Number of objective gradient evaluations             = 51
Number of equality constraint evaluations            = 55
Number of inequality constraint evaluations          = 0
Number of equality constraint Jacobian evaluations   = 51
Number of inequality constraint Jacobian evaluations = 0
Number of Lagrangian Hessian evaluations             = 50
Total seconds in IPOPT                               = 330.056

EXIT: Maximum Number of Iterations Exceeded.
</code><button class="copy-button fa-solid fa-copy" aria-label="Copy this code block" title="Copy"></button></pre>
```
=#


# Let's look at the fidelity in the subspace

fid = unitary_rollout_fidelity(prob.trajectory, sys; subspace=op.subspace)
println("Fidelity: ", fid)
@assert fid > 0.99

# and plot the result using the `plot_unitary_populations` function.

plot_unitary_populations(prob.trajectory; fig_size=(900, 700))

# ## Leakage suppresion
# As can be seen from the above plot, there is a substantial amount of leakage into the higher levels during the evolution. To mitigate this, we have implemented a constraint to avoid populating the leakage levels, which should ideally drive those leakage populations down to zero.
# To implement this, pass `leakage_constraint=true` and set `leakage_constraint_value={value}` and `leakage_cost={value}` to the `PiccoloOptions` instance passed to the `UnitarySmoothPulseProblem` template.

## create the a leakage suppression problem, initializing with the previous solution

prob_leakage = UnitarySmoothPulseProblem(sys, op, N, Δt;
    u_guess=prob.trajectory.u[:, :],
    piccolo_options=PiccoloOptions(
        leakage_constraint=true,
        leakage_constraint_value=1e-2,
        leakage_cost=1e-2,
    ),
)

## solve the problem
load_path = joinpath(dirname(Base.active_project()), "data/multilevel_transmon_example_leakage_89ee72.jld2") # hide
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
        applying constraint: initial value of u
        applying constraint: final value of u
        applying constraint: bounds on u
        applying constraint: bounds on du
        applying constraint: bounds on ddu
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
   0  2.2434810e-01 1.80e-01 2.30e-01   0.0 0.00e+00    -  0.00e+00 0.00e+00   0
   1  5.1360583e-01 1.65e-01 1.95e+02  -1.4 7.80e-01   0.0 3.96e-01 1.19e-01h  1
   2  3.3978388e-01 1.50e-01 1.74e+02  -2.6 9.24e-01    -  1.15e-01 1.04e-01h  1
   3  1.5298834e-01 1.41e-01 1.63e+02  -1.4 1.88e+00    -  1.06e-01 6.31e-02f  1
   4  1.7494458e-01 1.29e-01 4.47e+01  -2.1 1.72e+00    -  9.03e-02 8.41e-02h  1

<...snip...>

 245  1.7381917e-03 3.01e-05 1.70e-02  -4.0 1.37e-02  -2.4 1.00e+00 1.00e+00h  1
 246  1.7355438e-03 1.21e-05 1.17e-02  -4.0 1.11e-02  -2.0 1.00e+00 1.00e+00h  1
 247  1.7084108e-03 5.16e-05 2.32e-02  -4.0 1.95e-02  -2.4 1.00e+00 1.00e+00h  1
 248  1.7080428e-03 2.11e-05 1.73e-02  -4.0 1.68e-02  -2.0 1.00e+00 1.00e+00h  1
  
 250  1.6487637e-03 4.14e-05 2.81e-02  -4.0 2.77e-02  -2.1 1.00e+00 1.00e+00h  1

Number of Iterations....: 250

                                   (scaled)                 (unscaled)
Objective...............:   1.6487637480209457e-03    1.6487637480209457e-03
Dual infeasibility......:   2.8069428563796350e-02    2.8069428563796350e-02
Constraint violation....:   4.1350575226820063e-05    4.1350575226820063e-05
Variable bound violation:   0.0000000000000000e+00    0.0000000000000000e+00
Complementarity.........:   1.0000000000000002e-04    1.0000000000000002e-04
Overall NLP error.......:   2.8069428563796350e-02    2.8069428563796350e-02


Number of objective function evaluations             = 270
Number of objective gradient evaluations             = 251
Number of equality constraint evaluations            = 270
Number of inequality constraint evaluations          = 270
Number of equality constraint Jacobian evaluations   = 251
Number of inequality constraint Jacobian evaluations = 251
Number of Lagrangian Hessian evaluations             = 250
Total seconds in IPOPT                               = 1864.901

EXIT: Maximum Number of Iterations Exceeded.
</code><button class="copy-button fa-solid fa-copy" aria-label="Copy this code block" title="Copy"></button></pre>
```
=#

# Let's look at the fidelity in the subspace

fid_leakage = unitary_rollout_fidelity(prob_leakage.trajectory, sys; subspace=op.subspace)
println("Fidelity: ", fid_leakage)
@assert fid_leakage > 0.99

# and plot the result using the `plot_unitary_populations` function.

plot_unitary_populations(prob_leakage.trajectory; fig_size=(900, 700))

# Here we can see that the leakage populations have been driven substantially down.
