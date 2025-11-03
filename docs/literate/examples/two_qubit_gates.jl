# # Two Qubit Gates

# In this example we will solve for a selection of two-qubit gates using a simple two-qubit system. We will use the [`UnitarySmoothPulseProblem`](@ref) template to solve for the optimal control fields.

# ## Defining our Hamiltonian

# In quantum optimal control we work with Hamiltonians of the form

# ```math
# H(t) = H_{\text{drift}} + \sum_{j} u^j(t) H_{\text{drive}}^j,
# ```

# Specifically, for a simple two-qubit system in a rotating frame, we have

# ```math
# H = J_{12} \sigma_1^x \sigma_2^x + \sum_{i \in {1,2}} a_i^R(t) {\sigma^x_i \over 2} + a_i^I(t) {\sigma^y_i \over 2}.
# ```

# where

# ```math
# \begin{align*}
# J_{12} &= 0.001 \text{ GHz}, \\
# |a_i^R(t)| &\leq 0.1 \text{ GHz} \\
# \end{align*}
# ```

# And the duration of the gate will be capped at $400 \ \mu s$.

# Let's now set this up using some of the convenience functions available in QuantumCollocation.jl.

using QuantumCollocation
using PiccoloQuantumObjects
using NamedTrajectories
using LinearAlgebra

using PiccoloPlots
using CairoMakie

⊗(a, b) = kron(a, b)

## Define our operators
σx = GATES[:X]
σy = GATES[:Y]
Id = GATES[:I]

## Lift the operators to the two-qubit Hilbert space
σx_1 = σx ⊗ Id
σx_2 = Id ⊗ σx

σy_1 = σy ⊗ Id
σy_2 = Id ⊗ σy

## Define the parameters of the Hamiltonian
J_12 = 0.001 # GHz
a_bound = 0.100 # GHz

## Define the drift (coupling) Hamiltonian
H_drift = J_12 * (σx ⊗ σx)

## Define the control Hamiltonians
H_drives = [σx_1 / 2, σy_1 / 2, σx_2 / 2, σy_2 / 2]

## Define control (and higher derivative) bounds
a_bound = 0.1
da_bound = 0.0005
dda_bound = 0.0025

## Scale the Hamiltonians by 2π
H_drift *= 2π
H_drives .*= 2π

## Define the time parameters
T = 100 # timesteps
T_max = 1.0 # max evolution time
u_bounds = [
    (-1.0, 1.0),
    (-1.0, 1.0),
    (-1.0, 1.0),
    (-1.0, 1.0)
]
duration = 100 # μs
Δt = duration / T
Δt_max = 400 / T

## Define the system
sys = QuantumSystem(H_drift, H_drives, T_max, u_bounds)

# ## SWAP gate

## Define the goal operation
U_goal = [
    1 0 0 0;
    0 0 1 0;
    0 1 0 0;
    0 0 0 1
] |> Matrix{ComplexF64}

## Set up the problem
prob = UnitarySmoothPulseProblem(
    sys,
    U_goal,
    T,
    Δt;
    a_bound=a_bound,
    da_bound=da_bound,
    dda_bound=dda_bound,
    R_da=0.01,
    R_dda=0.01,
    Δt_max=Δt_max,
    piccolo_options=PiccoloOptions(bound_state=true),
)
fid_init = unitary_rollout_fidelity(prob.trajectory, sys, drive_name=:a)
println(fid_init)

# Solve the problem

#=
```julia
solve!(prob; max_iter=100)
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
        applying constraint: bounds on Ũ⃗

******************************************************************************
This program contains Ipopt, a library for large-scale nonlinear optimization.
 Ipopt is released as open source code under the Eclipse Public License (EPL).
         For more information visit https://github.com/coin-or/Ipopt
******************************************************************************

This is Ipopt version 3.14.19, running with linear solver MUMPS 5.8.1.

Number of nonzeros in equality constraint Jacobian...:   122590
Number of nonzeros in inequality constraint Jacobian.:        0
Number of nonzeros in Lagrangian Hessian.............:    22843

Total number of variables............................:     4460
                     variables with only lower bounds:        0
                variables with lower and upper bounds:     4460
                     variables with only upper bounds:        0
Total number of equality constraints.................:     4059
Total number of inequality constraints...............:        0
        inequality constraints with only lower bounds:        0
   inequality constraints with lower and upper bounds:        0
        inequality constraints with only upper bounds:        0

iter    objective    inf_pr   inf_du lg(mu)  ||d||  lg(rg) alpha_du alpha_pr  ls
   0  1.9966690e+00 4.09e-01 1.50e+01   0.0 0.00e+00    -  0.00e+00 0.00e+00   0
   1  1.8868533e+00 4.08e-01 1.49e+01  -2.2 7.79e-01    -  2.56e-03 2.67e-03f  1
   2  1.5589426e+00 4.05e-01 1.49e+01  -2.2 7.02e-01    -  2.93e-03 7.80e-03f  1
   3  1.2080227e+00 4.00e-01 1.48e+01  -2.2 6.76e-01    -  6.93e-03 1.10e-02f  1
   4  1.0979090e+00 3.98e-01 1.90e+01  -2.0 6.91e-01    -  3.70e-02 5.96e-03f  1

<...snip...>

  95  5.0246578e-04 8.45e-07 1.60e-03  -5.4 3.00e-03  -0.5 1.00e+00 1.00e+00h  1
  96  4.1531633e-04 6.13e-07 7.07e+01  -5.4 3.03e-03  -0.9 1.00e+00 1.00e+00h  1
  97  3.3533437e-04 5.91e-06 4.93e+01  -4.0 9.44e-03  -1.4 1.00e+00 3.03e-01h  2
  98  3.9456259e-04 5.21e-06 2.76e+01  -4.0 9.13e-03  -1.9 1.00e+00 1.25e-01h  4
  99  3.3591342e-04 4.60e-06 4.66e+01  -4.0 1.34e-02  -2.4 1.00e+00 1.25e-01h  4
iter    objective    inf_pr   inf_du lg(mu)  ||d||  lg(rg) alpha_du alpha_pr  ls
 100  3.3633429e-04 4.46e-06 2.56e+01  -4.1 2.62e-02  -2.8 1.00e+00 3.12e-02h  6

Number of Iterations....: 100

                                   (scaled)                 (unscaled)
Objective...............:   3.3633429041403043e-04    3.3633429041403043e-04
Dual infeasibility......:   2.5590761688032366e+01    2.5590761688032366e+01
Constraint violation....:   4.4603128525290609e-06    4.4603128525290609e-06
Variable bound violation:   0.0000000000000000e+00    0.0000000000000000e+00
Complementarity.........:   1.1727909811733182e-04    1.1727909811733182e-04
Overall NLP error.......:   2.5590761688032366e+01    2.5590761688032366e+01


Number of objective function evaluations             = 216
Number of objective gradient evaluations             = 101
Number of equality constraint evaluations            = 216
Number of inequality constraint evaluations          = 0
Number of equality constraint Jacobian evaluations   = 101
Number of inequality constraint Jacobian evaluations = 0
Number of Lagrangian Hessian evaluations             = 100
Total seconds in IPOPT                               = 470.209

EXIT: Maximum Number of Iterations Exceeded.
</code><button class="copy-button fa-solid fa-copy" aria-label="Copy this code block" title="Copy"></button></pre>
```
=#

load_path = joinpath(dirname(Base.active_project()), "data/two_qubit_gates_049034.jld2") # hide
prob.trajectory = load_traj(load_path) # hide
nothing # hide

## Let's take a look at the final fidelity
fid_final = unitary_rollout_fidelity(prob.trajectory, sys, drive_name=:a)
println(fid_final)
@assert fid_final > 0.99

# Looks good!

# Now let's plot the pulse and the population trajectories for the first two columns of the unitary, i.e. initial state of $\ket{00}$ and $\ket{01}$. For this we provide the function [`plot_unitary_populations`](@ref).
plot_unitary_populations(prob.trajectory, control_name=:a)


# For fun, let's look at a minimum time pulse for this problem
min_time_prob = UnitaryMinimumTimeProblem(prob, U_goal; final_fidelity=.995)

#=
```julia
solve!(min_time_prob; max_iter=300)
```

```@raw html<pre class="documenter-example-output"><code class="nohighlight hljs ansi">    initializing optimizer...
        applying constraint: timesteps all equal constraint
        applying constraint: initial value of Ũ⃗
        applying constraint: initial value of a
        applying constraint: final value of a
        applying constraint: bounds on a
        applying constraint: bounds on da
        applying constraint: bounds on dda
        applying constraint: bounds on Δt
        applying constraint: bounds on Ũ⃗
This is Ipopt version 3.14.19, running with linear solver MUMPS 5.8.1.

Number of nonzeros in equality constraint Jacobian...:    47302
Number of nonzeros in inequality constraint Jacobian.:       32
Number of nonzeros in Lagrangian Hessian.............:    23371

Total number of variables............................:     4460
                     variables with only lower bounds:        0
                variables with lower and upper bounds:     4460
                     variables with only upper bounds:        0
Total number of equality constraints.................:     4059
Total number of inequality constraints...............:        1
        inequality constraints with only lower bounds:        0
   inequality constraints with lower and upper bounds:        0
        inequality constraints with only upper bounds:        1

iter    objective    inf_pr   inf_du lg(mu)  ||d||  lg(rg) alpha_du alpha_pr  ls
   0  3.7646707e+04 9.71e-03 1.00e+02   0.0 0.00e+00    -  0.00e+00 0.00e+00   0
   1  3.7636990e+04 9.71e-03 8.01e+03   1.4 7.82e+01    -  1.83e-02 2.83e-05f  2
   2  2.9334248e+04 9.65e-03 4.89e+04   1.4 1.49e+02    -  2.82e-02 5.65e-03f  1
   3  2.7936295e+04 9.48e-03 7.07e+05   1.4 8.45e+00    -  1.00e+00 1.67e-02f  1
   4  2.8403697e+04 5.00e-03 2.63e+05   1.2 3.73e-01    -  6.91e-01 6.10e-01H  1

<...snip...>

 295  3.5549085e+04 8.00e-07 2.78e-01  -4.1 9.20e-04   0.1 1.00e+00 1.00e+00h  1
 296  3.5549085e+04 1.09e-06 6.13e-01  -4.1 1.31e-03  -0.4 1.00e+00 1.00e+00h  1
 297  3.5549085e+04 2.18e-06 1.88e+00  -4.1 2.13e-03  -0.9 1.00e+00 1.00e+00h  1
 298  3.5549085e+04 9.05e-07 3.05e-01  -4.1 1.08e-03  -0.5 1.00e+00 1.00e+00h  1
 299  3.5549085e+04 3.69e-06 1.98e+00  -4.1 2.65e-03  -1.0 1.00e+00 1.00e+00h  1
iter    objective    inf_pr   inf_du lg(mu)  ||d||  lg(rg) alpha_du alpha_pr  ls
 300  3.5549085e+04 1.52e-06 2.65e-01  -4.1 1.29e-03  -0.5 1.00e+00 1.00e+00h  1

Number of Iterations....: 300

                                   (scaled)                 (unscaled)
Objective...............:   3.5549083994173525e+04    3.5549085239436019e+04
Dual infeasibility......:   2.6456416158024953e-01    2.6456417084776901e-01
Constraint violation....:   1.5174279920371347e-06    1.5174279920371347e-06
Variable bound violation:   0.0000000000000000e+00    0.0000000000000000e+00
Complementarity.........:   8.0239910283720243e-05    8.0239913094474829e-05
Overall NLP error.......:   3.8843992327658324e-03    2.6456417084776901e-01


Number of objective function evaluations             = 405
Number of objective gradient evaluations             = 301
Number of equality constraint evaluations            = 405
Number of inequality constraint evaluations          = 405
Number of equality constraint Jacobian evaluations   = 301
Number of inequality constraint Jacobian evaluations = 301
Number of Lagrangian Hessian evaluations             = 300
Total seconds in IPOPT                               = 1342.144

EXIT: Maximum Number of Iterations Exceeded.
</code><button class="copy-button fa-solid fa-copy" aria-label="Copy this code block" title="Copy"></button></pre>
```
=#

load_path = joinpath(dirname(Base.active_project()), "data/two_qubit_gates_min_time_049034.jld2") # hide
min_time_prob.trajectory = load_traj(load_path) # hide
nothing # hide

fid_final_min_time = unitary_rollout_fidelity(min_time_prob.trajectory, sys, drive_name=:a)
println(fid_final_min_time)
@assert fid_final_min_time > 0.99

# And let's plot this solution
plot_unitary_populations(min_time_prob.trajectory, control_name=:a)

# It looks like our pulse derivative bounds are holding back the solution, but regardless, the duration has decreased:
duration = get_duration(prob.trajectory)
min_time_duration = get_duration(min_time_prob.trajectory)
println(duration, " - ", min_time_duration, " = ", duration - min_time_duration)




# ## Mølmer–Sørensen gate

# Here we will solve for a [Mølmer–Sørensen gate](https://en.wikipedia.org/wiki/M%C3%B8lmer%E2%80%93S%C3%B8rensen_gate) between two. The gate is generally described, for N qubits, by the unitary matrix

# ```math
# U_{\text{MS}}(\vec\theta) = \exp\left(i\sum_{j=1}^{N-1}\sum_{k=j+1}^{N}\theta_{jk}\sigma_j^x\sigma_k^x\right),
# ```

# where $\sigma_j^x$ is the Pauli-X operator acting on the $j$-th qubit, and $\vec\theta$ is a vector of real parameters. The Mølmer–Sørensen gate is a two-qubit gate that is particularly well-suited for trapped-ion qubits, where the interaction between qubits is mediated.

# Here we will focus on the simplest case of a Mølmer–Sørensen gate between two qubits. The gate is described by the unitary matrix

# ```math
# U_{\text{MS}}\left({\pi \over 4}\right) = \exp\left(i\frac{\pi}{4}\sigma_1^x\sigma_2^x\right).
# ```

# Let's set up the problem.

## Define the goal operation
U_goal = exp(im * π/4 * σx_1 * σx_2)

## Set up and solve the problem

prob = UnitarySmoothPulseProblem(
    sys,
    U_goal,
    T,
    Δt;
    a_bound=a_bound,
    da_bound=da_bound,
    dda_bound=dda_bound,
    R_da=0.01,
    R_dda=0.01,
    Δt_max=Δt_max,
    piccolo_options=PiccoloOptions(bound_state=true),
)
fid_init = unitary_rollout_fidelity(prob.trajectory, sys, drive_name=:a)
println(fid_init)
#=
```julia
solve!(prob; max_iter=300)
```

```@raw html<pre class="documenter-example-output"><code class="nohighlight hljs ansi">    initializing optimizer...
        applying constraint: timesteps all equal constraint
        applying constraint: initial value of Ũ⃗
        applying constraint: initial value of a
        applying constraint: final value of a
        applying constraint: bounds on a
        applying constraint: bounds on da
        applying constraint: bounds on dda
        applying constraint: bounds on Δt
        applying constraint: bounds on Ũ⃗
This is Ipopt version 3.14.19, running with linear solver MUMPS 5.8.1.

Number of nonzeros in equality constraint Jacobian...:   122590
Number of nonzeros in inequality constraint Jacobian.:        0
Number of nonzeros in Lagrangian Hessian.............:    22843

Total number of variables............................:     4460
                     variables with only lower bounds:        0
                variables with lower and upper bounds:     4460
                     variables with only upper bounds:        0
Total number of equality constraints.................:     4059
Total number of inequality constraints...............:        0
        inequality constraints with only lower bounds:        0
   inequality constraints with lower and upper bounds:        0
        inequality constraints with only upper bounds:        0

iter    objective    inf_pr   inf_du lg(mu)  ||d||  lg(rg) alpha_du alpha_pr  ls
   0  6.2300136e-03 3.81e-01 1.46e+01   0.0 0.00e+00    -  0.00e+00 0.00e+00   0
   1  1.5780799e-01 3.80e-01 6.29e+01  -4.0 1.03e+00    -  1.81e-03 2.36e-03h  1
   2  1.3401222e-01 3.80e-01 6.29e+01  -1.7 7.40e-01   2.0 6.20e-04 3.40e-04f  1
   3  9.8120325e-03 3.80e-01 6.29e+01  -1.7 7.56e-01   1.5 1.14e-03 9.96e-04f  1
   4  2.7649956e-01 3.80e-01 1.49e+01  -0.5 8.96e+00   1.0 2.22e-04 2.15e-04f  1

<...snip...>

 295  5.0368776e-04 5.91e-07 9.33e-04  -4.0 1.49e-03  -1.9 1.00e+00 1.00e+00h  1
 296  5.0392408e-04 3.25e-06 7.07e+01  -4.0 3.85e-03  -2.4 1.00e+00 1.00e+00h  1
 297  5.1888064e-04 3.14e-06 1.10e+00  -4.1 8.83e-02  -2.9 1.00e+00 1.56e-02h  7
 298  5.4982617e-04 2.72e-04 7.02e+01  -4.1 7.86e-02  -3.4 1.00e+00 5.00e-01h  2
 299  5.0663318e-04 2.72e-04 1.64e+00  -4.1 2.98e-01  -1.1 4.66e-01 1.84e-04h 12
iter    objective    inf_pr   inf_du lg(mu)  ||d||  lg(rg) alpha_du alpha_pr  ls
 300  5.1814196e-04 2.71e-04 7.01e+01  -4.1 8.33e-03  -1.6 1.00e+00 3.91e-03h  9

Number of Iterations....: 300

                                   (scaled)                 (unscaled)
Objective...............:   5.1814196246723770e-04    5.1814196246723770e-04
Dual infeasibility......:   7.0147652880081338e+01    7.0147652880081338e+01
Constraint violation....:   2.7079742887581304e-04    2.7079742887581304e-04
Variable bound violation:   0.0000000000000000e+00    0.0000000000000000e+00
Complementarity.........:   8.2878984114778941e-05    8.2878984114778941e-05
Overall NLP error.......:   7.0147652880081338e+01    7.0147652880081338e+01


Number of objective function evaluations             = 813
Number of objective gradient evaluations             = 301
Number of equality constraint evaluations            = 813
Number of inequality constraint evaluations          = 0
Number of equality constraint Jacobian evaluations   = 301
Number of inequality constraint Jacobian evaluations = 0
Number of Lagrangian Hessian evaluations             = 300
Total seconds in IPOPT                               = 1302.893

EXIT: Maximum Number of Iterations Exceeded.
```
=#

load_path = joinpath(dirname(Base.active_project()), "data/two_qubit_gates_molmer_049034.jld2") # hide
prob.trajectory = load_traj(load_path) # hide
nothing # hide

## Let's take a look at the final fidelity
fid_final = unitary_rollout_fidelity(prob.trajectory, sys, drive_name=:a)
println(fid_final)
@assert fid_final > 0.999

# Again, looks good!

# Now let's plot the pulse and the population trajectories for the first two columns of the unitary, i.e. initial state of $\ket{00}$ and $\ket{01}$.
plot_unitary_populations(prob.trajectory, control_name=:a)

# For fun, let's look at a minimum time pulse for this problem

min_time_prob = UnitaryMinimumTimeProblem(prob, U_goal; final_fidelity=.9995)
#=
```julia
solve!(min_time_prob; max_iter=300)
```

```@raw html<pre class="documenter-example-output"><code class="nohighlight hljs ansi">    initializing optimizer...
        applying constraint: timesteps all equal constraint
        applying constraint: initial value of Ũ⃗
        applying constraint: initial value of a
        applying constraint: final value of a
        applying constraint: bounds on a
        applying constraint: bounds on da
        applying constraint: bounds on dda
        applying constraint: bounds on Δt
        applying constraint: bounds on Ũ⃗
This is Ipopt version 3.14.19, running with linear solver MUMPS 5.8.1.

Number of nonzeros in equality constraint Jacobian...:    47302
Number of nonzeros in inequality constraint Jacobian.:       32
Number of nonzeros in Lagrangian Hessian.............:    23371

Total number of variables............................:     4460
                     variables with only lower bounds:        0
                variables with lower and upper bounds:     4460
                     variables with only upper bounds:        0
Total number of equality constraints.................:     4059
Total number of inequality constraints...............:        1
        inequality constraints with only lower bounds:        0
   inequality constraints with lower and upper bounds:        0
        inequality constraints with only upper bounds:        1

iter    objective    inf_pr   inf_du lg(mu)  ||d||  lg(rg) alpha_du alpha_pr  ls
   0  3.1295353e+04 9.80e-03 9.20e+01   0.0 0.00e+00    -  0.00e+00 0.00e+00   0
   1  3.1205625e+04 2.66e-06 9.80e+01  -1.5 9.80e-03   4.0 1.00e+00 1.00e+00f  1
   2  3.0932870e+04 1.17e-05 9.20e+01  -2.9 2.76e-02   3.5 1.00e+00 1.00e+00f  1
   3  3.0116114e+04 1.08e-04 9.17e+01  -4.0 8.25e-02   3.0 1.00e+00 1.00e+00f  1
   4  2.9426502e+04 1.60e-04 1.00e+02  -4.0 2.46e-01   2.6 1.00e+00 2.83e-01f  1

<...snip...>

 295  1.2144136e+04 3.04e-09 5.42e-04  -9.0 1.02e-04  -2.3 1.00e+00 1.00e+00h  1
 296  1.2144136e+04 1.96e-08 3.48e-03  -9.0 2.91e-04  -2.7 1.00e+00 1.00e+00h  1
 297  1.2144136e+04 2.97e-09 5.28e-04  -9.0 1.05e-04  -2.3 1.00e+00 1.00e+00h  1
 298  1.2144136e+04 2.66e-08 4.77e-03  -9.0 3.57e-04  -2.8 1.00e+00 1.00e+00h  1
 299  1.2144136e+04 5.60e-08 1.00e-02  -9.0 2.68e-04  -2.4 1.00e+00 1.00e+00h  1
iter    objective    inf_pr   inf_du lg(mu)  ||d||  lg(rg) alpha_du alpha_pr  ls
 300  1.2144136e+04 6.10e-08 1.09e-02  -9.0 2.72e-04  -1.9 1.00e+00 1.00e+00h  1

Number of Iterations....: 300

                                   (scaled)                 (unscaled)
Objective...............:   1.2144134659199719e+04    1.2144135665519536e+04
Dual infeasibility......:   1.0938603912065324e-02    1.0938604818489222e-02
Constraint violation....:   6.1030180686927338e-08    6.1030180686927338e-08
Variable bound violation:   0.0000000000000000e+00    0.0000000000000000e+00
Complementarity.........:   9.0909143731021008e-10    9.0909151264177987e-10
Overall NLP error.......:   1.0036008200383635e-04    1.0938604818489222e-02


Number of objective function evaluations             = 502
Number of objective gradient evaluations             = 301
Number of equality constraint evaluations            = 502
Number of inequality constraint evaluations          = 502
Number of equality constraint Jacobian evaluations   = 301
Number of inequality constraint Jacobian evaluations = 301
Number of Lagrangian Hessian evaluations             = 300
Total seconds in IPOPT                               = 1045.229

EXIT: Maximum Number of Iterations Exceeded.
</code><button class="copy-button fa-solid fa-copy" aria-label="Copy this code block" title="Copy"></button></pre>
```
=#

load_path = joinpath(dirname(Base.active_project()), "data/two_qubit_gates_molmer_min_time_049034.jld2") # hide
min_time_prob.trajectory = load_traj(load_path) # hide
nothing # hide

fid_final_min_time = unitary_rollout_fidelity(min_time_prob.trajectory, sys, drive_name=:a)
println(fid_final_min_time)
@assert fid_final_min_time > 0.999

# And let's plot this solution
plot_unitary_populations(min_time_prob.trajectory, control_name=:a)

# It looks like our pulse derivative bounds are holding back the solution, but regardless, the duration has decreased:

duration = get_duration(prob.trajectory)
min_time_duration = get_duration(min_time_prob.trajectory)
println(duration, " - ", min_time_duration, " = ", duration - min_time_duration)
