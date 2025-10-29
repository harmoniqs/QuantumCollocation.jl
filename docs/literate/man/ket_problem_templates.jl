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
println("Before: ", rollout_fidelity(state_prob.trajectory, system))

# _solve the problem_
solve!(state_prob, max_iter=100, verbose=true, print_level=1);

# _check the fidelity after solving_
println("After: ", rollout_fidelity(state_prob.trajectory, system))

# _extract the control pulses_
state_prob.trajectory.u |> size

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
solve!(min_state_prob, max_iter=100, verbose=true, print_level=1);

# _check the new duration_
println("Duration after: ", get_duration(min_state_prob.trajectory))

# _the fidelity is preserved by a constraint_
println("Fidelity after: ", rollout_fidelity(min_state_prob.trajectory, system))

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
solve!(sampling_state_prob, max_iter=25, verbose=true, print_level=1);

# _check the fidelity of the sampling problem (use the updated key to get the initial and goal)_
println("After (original system): ", rollout_fidelity(sampling_state_prob.trajectory, system, state_name=:ψ̃1_system_1))
println("After (new system): ", rollout_fidelity(sampling_state_prob.trajectory, driftless_system, state_name=:ψ̃1_system_1))

# _compare this to using the original problem on the new system_
println("After (new system, original `prob`): ", rollout_fidelity(state_prob.trajectory, driftless_system))

# -----