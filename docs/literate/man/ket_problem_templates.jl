# ```@meta
# CollapsedDocStrings = true
# ```
using NamedTrajectories
using PiccoloQuantumObjects
using QuantumCollocation

# -----

#=
## Quantum State Smooth Pulse Problem

```@docs; canonical = false
QuantumStateSmoothPulseProblem
```

Each problem starts with a `QuantumSystem` object, which is used to define the system's
Hamiltonian and control operators. The goal is to find a control pulse that drives the
intial state, `ψ_init`, to a target state, `ψ_goal`.
=#

# _define the quantum system_
system = QuantumSystem(0.1 * PAULIS.Z, [PAULIS.X, PAULIS.Y])
ψ_init = Vector{ComplexF64}([1.0, 0.0])
ψ_goal = Vector{ComplexF64}([0.0, 1.0])
T = 51
Δt = 0.2

# _create the smooth pulse problem_
state_prob = QuantumStateSmoothPulseProblem(system, ψ_init, ψ_goal, T, Δt);

# _check the fidelity before solving_
println("Before: ", rollout_fidelity(state_prob.trajectory, system, control_name=:a))

# _solve the problem_

#=
```julia
solve!(state_prob, max_iter=100, verbose=true, print_level=1);
```

```@raw html
<pre class="documenter-example-output"><code class="nohighlight hljs ansi">    initializing optimizer...
        applying constraint: timesteps all equal constraint
        applying constraint: initial value of ψ̃
        applying constraint: initial value of a
        applying constraint: final value of a
        applying constraint: bounds on a
        applying constraint: bounds on da
        applying constraint: bounds on dda
        applying constraint: bounds on Δt
</code><button class="copy-button fa-solid fa-copy" aria-label="Copy this code block" title="Copy"></button></pre>
```
=#
load_path = joinpath(dirname(Base.active_project()), "data/ket_problem_templates_state_prob_25e3be.jld2") # hide
state_prob.trajectory = load_traj(load_path) # hide
nothing # hide

# _check the fidelity after solving_
println("After: ", rollout_fidelity(state_prob.trajectory, system, control_name=:a))

# _extract the control pulses_
state_prob.trajectory.a |> size

# -----

#=
## Quantum State Minimum Time Problem

```@docs; canonical = false
QuantumStateMinimumTimeProblem
```
=#

# _create the minimum time problem_
min_state_prob = QuantumStateMinimumTimeProblem(state_prob, ψ_goal);

# _check the previous duration_
println("Duration before: ", get_duration(state_prob.trajectory))

# _solve the minimum time problem_

#=
```julia
solve!(min_state_prob, max_iter=100, verbose=true, print_level=1);
```

```@raw html
<pre class="documenter-example-output"><code class="nohighlight hljs ansi">    initializing optimizer...
        applying constraint: timesteps all equal constraint
        applying constraint: initial value of ψ̃
        applying constraint: initial value of a
        applying constraint: final value of a
        applying constraint: bounds on a
        applying constraint: bounds on da
        applying constraint: bounds on dda
        applying constraint: bounds on Δt
</code><button class="copy-button fa-solid fa-copy" aria-label="Copy this code block" title="Copy"></button></pre>
```
=#
load_path = joinpath(dirname(Base.active_project()), "data/ket_problem_templates_state_prob_min_time_25e3be.jld2") # hide
min_state_prob.trajectory = load_traj(load_path) # hide
nothing # hide

# _check the new duration_
println("Duration after: ", get_duration(min_state_prob.trajectory))

# _the fidelity is preserved by a constraint_
println("Fidelity after: ", rollout_fidelity(min_state_prob.trajectory, system, control_name=:a))

# -----

#=

## Quantum State Sampling Problem

```@docs; canonical = false
QuantumStateSamplingProblem
```
=#

# _create a sampling problem_
driftless_system = QuantumSystem([PAULIS.X, PAULIS.Y])
sampling_state_prob = QuantumStateSamplingProblem([system, driftless_system], ψ_init, ψ_goal, T, Δt);

# _new keys are added to the trajectory for the new states_
println(sampling_state_prob.trajectory.state_names)

# _solve the sampling problem for a few iterations_

#=
```julia
solve!(sampling_state_prob, max_iter=25, verbose=true, print_level=1);
```

```@raw html
<pre class="documenter-example-output"><code class="nohighlight hljs ansi">   initializing optimizer...
        applying constraint: timesteps all equal constraint
        applying constraint: initial value of ψ̃1_system_1
        applying constraint: initial value of a
        applying constraint: initial value of ψ̃1_system_2
        applying constraint: final value of a
        applying constraint: bounds on a
        applying constraint: bounds on da
        applying constraint: bounds on dda
        applying constraint: bounds on Δt
</code><button class="copy-button fa-solid fa-copy" aria-label="Copy this code block" title="Copy"></button></pre>
```
=#
load_path = joinpath(dirname(Base.active_project()), "data/ket_problem_templates_sampling_state_25e3be.jld2") # hide
sampling_state_prob.trajectory = load_traj(load_path) # hide
nothing # hide

# _check the fidelity of the sampling problem (use the updated key to get the initial and goal)_
println("After (original system): ", rollout_fidelity(sampling_state_prob.trajectory, system, state_name=:ψ̃1_system_1, control_name=:a))
println("After (new system): ", rollout_fidelity(sampling_state_prob.trajectory, driftless_system, state_name=:ψ̃1_system_1, control_name=:a))

# _compare this to using the original problem on the new system_
println("After (new system, original `prob`): ", rollout_fidelity(state_prob.trajectory, driftless_system, control_name=:a))

# -----
