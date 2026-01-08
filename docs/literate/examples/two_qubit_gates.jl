# # Two Qubit Gates

# In this example we will solve for a selection of two-qubit gates using a simple two-qubit system. We will use the [`UnitarySmoothPulseProblem`](@ref) template to solve for the optimal control fields.

# ## Defining our Hamiltonian

# In quantum optimal control we work with Hamiltonians of the form

# ```math
# H(t) = H_{\text{drift}} + \sum_{j} u^j(t) H_{\text{drive}}^j,
# ```

# Specifically, for a simple two-qubit system in a rotating frame, we have

# ```math
# H = J_{12} \sigma_1^x \sigma_2^x + \sum_{i \in {1,2}} u_i^R(t) {\sigma^x_i \over 2} + u_i^I(t) {\sigma^y_i \over 2}.
# ```

# where

# ```math
# \begin{align*}
# J_{12} &= 0.001 \text{ GHz}, \\
# |u_i^R(t)| &\leq 0.1 \text{ GHz} \\
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
σx = GATES.X
σy = GATES.Y
Id = GATES.I

## Lift the operators to the two-qubit Hilbert space
σx_1 = σx ⊗ Id
σx_2 = Id ⊗ σx

σy_1 = σy ⊗ Id
σy_2 = Id ⊗ σy

## Define the parameters of the Hamiltonian
J_12 = 0.001 # GHz
u_bound = 0.100 # GHz

## Define the drift (coupling) Hamiltonian
H_drift = J_12 * (σx ⊗ σx)

## Define the control Hamiltonians
H_drives = [σx_1 / 2, σy_1 / 2, σx_2 / 2, σy_2 / 2]

## Define control (and higher derivative) bounds
u_bound = 0.1
du_bound = 0.0005
ddu_bound = 0.0025

## Scale the Hamiltonians by 2π
H_drift *= 2π
H_drives .*= 2π

## Define the time parameters
N = 100 # timesteps
T_max = 400.0 # μs (maximum duration)
drive_bounds = fill((-u_bound, u_bound), length(H_drives))

## Define the system
sys = QuantumSystem(H_drift, H_drives, drive_bounds)

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
    EmbeddedOperator(U_goal, sys),
    N;
    du_bound=du_bound,
    ddu_bound=ddu_bound,
    R_du=0.01,
    R_ddu=0.01,
    piccolo_options=PiccoloOptions(bound_state=true),
)
fid_init = unitary_rollout_fidelity(prob.trajectory, sys)
println(fid_init)

# Solve the problem
# load_path = joinpath(dirname(Base.active_project()), "data/two_qubit_gates_89ee72.jld2") # hide
# prob.trajectory = load_traj(load_path) # hide
# nothing # hide
# solve!(prob; max_iter=100, options=IpoptOptions(eval_hessian=false))

#=
```julia
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

## Let's take a look at the final fidelity
fid_final = unitary_rollout_fidelity(prob.trajectory, sys)
println(fid_final)
@assert fid_final > 0.99

# Looks good!

# Now let's plot the pulse and the population trajectories for the first two columns of the unitary, i.e. initial state of $\ket{00}$ and $\ket{01}$. For this we provide the function [`plot_unitary_populations`](@ref).
plot_unitary_populations(prob.trajectory)


# For fun, let's look at a minimum time pulse for this problem
min_time_prob = UnitaryMinimumTimeProblem(prob, U_goal; final_fidelity=.995)

# and solve the problem
# load_path = joinpath(dirname(Base.active_project()), "data/two_qubit_gates_min_time_89ee72.jld2") # hide
# min_time_prob.trajectory = load_traj(load_path) # hide
# nothing # hide
# solve!(min_time_prob; max_iter=300)

#=
```julia
```

```@raw html<pre class="documenter-example-output"><code class="nohighlight hljs ansi">    initializing optimizer...
        applying constraint: timesteps all equal constraint
        applying constraint: initial value of Ũ⃗
        applying constraint: initial value of u
        applying constraint: final value of u
        applying constraint: bounds on u
        applying constraint: bounds on du
        applying constraint: bounds on ddu
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

 295r 1.2144904e+04 2.29e-05 2.13e-04  -6.6 1.16e-02    -  1.00e+00 1.00e+00h  1
 296r 1.2173826e+04 5.43e-07 1.50e-05  -6.6 2.92e-03    -  1.00e+00 1.00e+00h  1
 297r 1.2174235e+04 2.15e-08 5.74e-07  -6.6 1.97e-04    -  1.00e+00 1.00e+00h  1
 298r 1.2153605e+04 1.35e-07 6.46e-06  -9.0 2.08e-03    -  1.00e+00 1.00e+00f  1
 299r 1.2259756e+04 2.02e-05 1.34e-02  -4.0 2.55e+00    -  1.75e-03 4.21e-03f  1
iter    objective    inf_pr   inf_du lg(mu)  ||d||  lg(rg) alpha_du alpha_pr  ls
 300r 1.2144896e+04 2.28e-05 2.08e-04  -6.6 1.16e-02    -  1.00e+00 1.00e+00h  1

Number of Iterations....: 300

                                   (scaled)                 (unscaled)
Objective...............:   1.2144895589604916e+04    1.2144895939586029e+04
Dual infeasibility......:   9.9999997700262952e+01    1.0000000058197654e+02
Constraint violation....:   2.2776062152680965e-05    2.2776062152680965e-05
Variable bound violation:   0.0000000000000000e+00    0.0000000000000000e+00
Complementarity.........:   3.9668426222216936e-07    3.9668427365347392e-07
Overall NLP error.......:   9.9999997700262952e+01    1.0000000058197654e+02


Number of objective function evaluations             = 409
Number of objective gradient evaluations             = 118
Number of equality constraint evaluations            = 409
Number of inequality constraint evaluations          = 409
Number of equality constraint Jacobian evaluations   = 302
Number of inequality constraint Jacobian evaluations = 302
Number of Lagrangian Hessian evaluations             = 300
Total seconds in IPOPT                               = 909.544

EXIT: Maximum Number of Iterations Exceeded.
</code><button class="copy-button fa-solid fa-copy" aria-label="Copy this code block" title="Copy"></button></pre>
```
=#

fid_final_min_time = unitary_rollout_fidelity(min_time_prob.trajectory, sys)
println(fid_final_min_time)
@assert fid_final_min_time > 0.99

# And let's plot this solution
plot_unitary_populations(min_time_prob.trajectory)

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

## Define the goal operation
U_goal_ms = exp(im * π/4 * σx_1 * σx_2)

## Set up and solve the problem

prob_ms = UnitarySmoothPulseProblem(
    sys,
    EmbeddedOperator(U_goal, sys),
    N;
    du_bound=du_bound,
    ddu_bound=ddu_bound,
    R_du=0.01,
    R_ddu=0.01,
    piccolo_options=PiccoloOptions(bound_state=true),
)
fid_init = unitary_rollout_fidelity(prob_ms.trajectory, sys)
println(fid_init)

# Solve the problem
# load_path = joinpath(dirname(Base.active_project()), "data/two_qubit_gates_molmer_89ee72.jld2") # hide
# prob_ms.trajectory = load_traj(load_path) # hide
# nothing # hide
# solve!(prob; max_iter=300)

#=
```julia
```

```@raw html<pre class="documenter-example-output"><code class="nohighlight hljs ansi">    initializing optimizer...
        applying constraint: timesteps all equal constraint
        applying constraint: initial value of Ũ⃗
        applying constraint: initial value of u
        applying constraint: final value of u
        applying constraint: bounds on u
        applying constraint: bounds on du
        applying constraint: bounds on ddu
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
   0  1.9968551e+00 4.24e-01 1.10e+01   0.0 0.00e+00    -  0.00e+00 0.00e+00   0
   1  1.8636183e+00 4.22e-01 1.09e+01  -2.2 1.04e+00    -  2.50e-03 6.41e-03f  1
   2  1.7504526e+00 4.20e-01 1.08e+01  -2.2 9.82e-01    -  5.03e-03 5.03e-03f  1
   3  1.5169129e+00 4.15e-01 1.07e+01  -2.0 9.83e-01    -  1.21e-02 1.17e-02f  1
   4  1.4313680e+00 4.11e-01 4.89e+01  -1.8 9.87e-01    -  4.43e-02 7.94e-03f  1

<...snip...>

 295  2.0649662e-04 8.04e-10 7.07e+01  -6.1 1.48e-04  -0.9 1.00e+00 1.00e+00h  1
 296  2.0699118e-04 9.41e-11 7.07e+01  -6.1 1.05e-04  -1.4 1.00e+00 1.00e+00H  1
 297  2.0698519e-04 2.70e-12 1.07e-02  -6.1 1.85e-06   1.8 1.00e+00 1.00e+00h  1
 298  2.0696914e-04 1.76e-12 4.14e-05  -6.1 2.20e-06   1.3 1.00e+00 1.00e+00h  1
 299  2.0692500e-04 8.75e-13 4.19e-05  -6.1 6.38e-06   0.8 1.00e+00 1.00e+00h  1
iter    objective    inf_pr   inf_du lg(mu)  ||d||  lg(rg) alpha_du alpha_pr  ls
 300  2.0681635e-04 8.07e-12 7.24e-05  -6.1 1.73e-05   0.3 1.00e+00 1.00e+00h  1

Number of Iterations....: 300

                                   (scaled)                 (unscaled)
Objective...............:   2.0681635138132748e-04    2.0681635138132748e-04
Dual infeasibility......:   7.2391829384785262e-05    7.2391829384785262e-05
Constraint violation....:   8.0724482325300042e-12    8.0724482325300042e-12
Variable bound violation:   0.0000000000000000e+00    0.0000000000000000e+00
Complementarity.........:   7.1554229772280104e-07    7.1554229772280104e-07
Overall NLP error.......:   9.9284457729729384e-05    9.9284457729729384e-05


Number of objective function evaluations             = 765
Number of objective gradient evaluations             = 301
Number of equality constraint evaluations            = 765
Number of inequality constraint evaluations          = 0
Number of equality constraint Jacobian evaluations   = 301
Number of inequality constraint Jacobian evaluations = 0
Number of Lagrangian Hessian evaluations             = 300
Total seconds in IPOPT                               = 1178.223

EXIT: Maximum Number of Iterations Exceeded.
```
=#

## Let's take a look at the final fidelity
fid_final = unitary_rollout_fidelity(prob_ms.trajectory, sys)
println(fid_final)
@assert fid_final > 0.999

# Again, looks good!

# Now let's plot the pulse and the population trajectories for the first two columns of the unitary, i.e. initial state of $\ket{00}$ and $\ket{01}$.
plot_unitary_populations(prob_ms.trajectory)

# For fun, let's look at a minimum time pulse for this problem

min_time_prob_ms = UnitaryMinimumTimeProblem(prob_ms, U_goal_ms; final_fidelity=.9995)

# and solve the problem
# load_path = joinpath(dirname(Base.active_project()), "data/two_qubit_gates_molmer_min_time_89ee72.jld2") # hide
# min_time_prob_ms.trajectory = load_traj(load_path) # hide
# nothing # hide
# solve!(min_time_prob; max_iter=300)

#=
```julia
```

```@raw html<pre class="documenter-example-output"><code class="nohighlight hljs ansi">   initializing optimizer...
        applying constraint: timesteps all equal constraint
        applying constraint: initial value of Ũ⃗
        applying constraint: initial value of u
        applying constraint: final value of u
        applying constraint: bounds on u
        applying constraint: bounds on du
        applying constraint: bounds on ddu
        applying constraint: bounds on Δt
        applying constraint: bounds on Ũ⃗
This is Ipopt version 3.14.19, running with linear solver MUMPS 5.8.1.

Number of nonzeros in equality constraint Jacobian...:    47302
Number of nonzeros in inequality constraint Jacobian.:       16
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
   0  1.2144896e+04 9.97e-03 1.00e+02   0.0 0.00e+00    -  0.00e+00 0.00e+00   0
   1  1.2126203e+04 9.97e-03 1.76e+04   0.9 2.03e+01    -  1.18e-01 9.28e-05f  2
   2  9.1682433e+03 9.31e-03 2.71e+05   0.8 4.52e+00    -  1.00e+00 6.62e-02f  1
   3  8.2020717e+03 2.88e-03 6.37e+04   0.7 1.78e-01    -  1.00e+00 6.90e-01f  1
   4  9.0133627e+03 1.38e-03 2.70e+04   0.9 1.57e-01    -  1.00e+00 5.21e-01h  1

   <...snip...>
 295r 1.3500591e+04 9.06e-04 3.82e+00  -4.0 7.41e-01    -  1.00e+00 1.80e-01f  1
 296r 1.2434231e+04 4.55e-04 1.93e-01  -4.3 1.08e-01    -  9.02e-01 1.00e+00f  1
 297r 1.2393849e+04 1.07e-04 1.45e-02  -4.3 2.66e-02    -  1.00e+00 1.00e+00h  1
 298r 1.2391589e+04 9.61e-06 1.68e-03  -4.3 9.68e-03    -  1.00e+00 1.00e+00h  1
 299r 1.2173834e+04 1.42e-05 7.36e-02  -6.5 2.20e-02    -  9.67e-01 1.00e+00f  1
iter    objective    inf_pr   inf_du lg(mu)  ||d||  lg(rg) alpha_du alpha_pr  ls
 300r 1.2196795e+04 4.35e-07 1.15e-04  -6.5 2.32e-03    -  1.00e+00 1.00e+00h  1

Number of Iterations....: 300

                                   (scaled)                 (unscaled)
Objective...............:   1.2196794676744477e+04    1.2196794796608538e+04
Dual infeasibility......:   9.9999997730845053e+01    9.9999998713595531e+01
Constraint violation....:   4.3514023147228631e-07    4.3514023147228631e-07
Variable bound violation:   0.0000000000000000e+00    0.0000000000000000e+00
Complementarity.........:   1.3299987986006171e-06    1.3299988116711870e-06
Overall NLP error.......:   9.9999997730845053e+01    9.9999998713595531e+01


Number of objective function evaluations             = 343
Number of objective gradient evaluations             = 30
Number of equality constraint evaluations            = 343
Number of inequality constraint evaluations          = 343
Number of equality constraint Jacobian evaluations   = 302
Number of inequality constraint Jacobian evaluations = 302
Number of Lagrangian Hessian evaluations             = 300
Total seconds in IPOPT                               = 936.607

EXIT: Maximum Number of Iterations Exceeded.
</code><button class="copy-button fa-solid fa-copy" aria-label="Copy this code block" title="Copy"></button></pre>
```
=#

fid_final_min_time = unitary_rollout_fidelity(min_time_prob_ms.trajectory, sys)
println(fid_final_min_time)
@assert fid_final_min_time > 0.999

# And let's plot this solution
plot_unitary_populations(min_time_prob_ms.trajectory)

# It looks like our pulse derivative bounds are holding back the solution, but regardless, the duration has decreased:

duration = get_duration(prob_ms.trajectory)
min_time_duration = get_duration(min_time_prob_ms.trajectory)
println(duration, " - ", min_time_duration, " = ", duration - min_time_duration)
