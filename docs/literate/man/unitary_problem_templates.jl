# ```@meta
# CollapsedDocStrings = true
# ```
using NamedTrajectories
using PiccoloQuantumObjects
using QuantumCollocation

# -----

#=
## Unitary Smooth Pulse Problem

```@docs; canonical = false
UnitarySmoothPulseProblem
```

The `UnitarySmoothPulseProblem` is similar to the `QuantumStateSmoothPulseProblem`, but
instead of driving the system to a target state, the goal is to drive the system to a
target unitary operator, `U_goal`.

=#

system = QuantumSystem(0.1 * PAULIS.Z, [PAULIS.X, PAULIS.Y])
U_goal = GATES.H
T = 51
Δt = 0.2

prob = UnitarySmoothPulseProblem(system, U_goal, T, Δt);

# _check the fidelity before solving_
println("Before: ", unitary_rollout_fidelity(prob.trajectory, system, drive_name=:a))

# _finding an optimal control is as simple as calling `solve!`_
#=
```julia
solve!(prob, max_iter=100, verbose=true, print_level=1);
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
</code><button class="copy-button fa-solid fa-copy" aria-label="Copy this code block" title="Copy"></button></pre>
```
=#
load_path = joinpath(dirname(Base.active_project()), "data/unitary_problem_templates_25e3be.jld2") # hide
prob.trajectory = load_traj(load_path) # hide
nothing # hide

# _check the fidelity after solving_
println("After: ", unitary_rollout_fidelity(prob.trajectory, system, drive_name=:a))

#=
The `NamedTrajectory` object stores the control pulse, state variables, and the time grid.
=#

# _extract the control pulses_
prob.trajectory.a |> size

# -----

#=
## Unitary Minimum Time Problem

```@docs; canonical = false
UnitaryMinimumTimeProblem
```

The goal of this problem is to find the shortest time it takes to drive the system to a
target unitary operator, `U_goal`. The problem is solved by minimizing the sum of all of
the time steps. It is constructed from `prob` in the previous example.
=#

min_prob = UnitaryMinimumTimeProblem(prob, U_goal);

# _check the previous duration_
println("Duration before: ", get_duration(prob.trajectory))

# _solve the minimum time problem_
#=
```julia
solve!(min_prob, max_iter=100, verbose=true, print_level=1);
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
</code><button class="copy-button fa-solid fa-copy" aria-label="Copy this code block" title="Copy"></button></pre>
```
=#
load_path = joinpath(dirname(Base.active_project()), "data/unitary_problem_templates_min_time_25e3be.jld2") # hide
min_prob.trajectory = load_traj(load_path) # hide
nothing # hide


# _check the new duration_
println("Duration after: ", get_duration(min_prob.trajectory))

# _the fidelity is preserved by a constraint_
println("Fidelity after: ", unitary_rollout_fidelity(min_prob.trajectory, system, drive_name=:a))

# -----

#=
## Unitary Sampling Problem

```@docs; canonical = false
UnitarySamplingProblem
```

A sampling problem is used to solve over multiple quantum systems with the same control.
This can be useful for exploring robustness, for example.
=#

# _create a sampling problem_
driftless_system = QuantumSystem([PAULIS.X, PAULIS.Y])
sampling_prob = UnitarySamplingProblem([system, driftless_system], U_goal, T, Δt);

# _new keys are addded to the trajectory for the new states_
println(sampling_prob.trajectory.state_names)

# _the `solve!` proceeds as in the [Quantum State Sampling Problem](#Quantum-State-Sampling-Problem)]_

# -----

#=
## Unitary Variational Problem

```@docs; canonical = false
UnitaryVariationalProblem
```

The `UnitaryVariationalProblem` uses a `VariationalQuantumSystem` to find a control that is
sensitive or robust to terms in the Hamiltonian. See the documentation for the
`VariationalQuantumSystem` in [`PiccoloQuantumObjects.jl`](https://github.com/harmoniqs/PiccoloQuantumObjects.jl)
for more details.
=#

# _create a variational system, with a variational Hamiltonian, `PAULIS.X`_
H_var = PAULIS.X
varsys = VariationalQuantumSystem([PAULIS.X, PAULIS.Y], [H_var]);

# _create a variational problem that is robust to `PAULIS.X` at the end_
robprob = UnitaryVariationalProblem(varsys, U_goal, T, Δt, robust_times=[[T]]);

# -----
