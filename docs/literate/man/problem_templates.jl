# ```@meta
# CollapsedDocStrings = true
# ```
using NamedTrajectories
using PiccoloQuantumObjects
using QuantumCollocation

#=
# Problem Templates

We provide a number of problem templates for making it simple and easy to set up and solve 
certain types of quantum optimal control problems. These templates all construct a 
`DirectTrajOptProblem` object, which stores all the parts of the optimal control problem.

This page provides a brief overview of each problem template, broken down by the state of
the problem being solved.

Ket Problem Templates:
- [Quantum State Smooth Pulse Problem](#Quantum-State-Smooth-Pulse-Problem)
- [Quantum State Minimum Time Problem](#Quantum-State-Minimum-Time-Problem)
- [Quantum State Sampling Problem](#Quantum-State-Sampling-Problem)

Unitary Problem Templates:
- [Unitary Smooth Pulse Problem](#Unitary-Smooth-Pulse-Problem)
- [Unitary Minimum Time Problem](#Unitary-Minimum-Time-Problem)
- [Unitary Sampling Problem](#Unitary-Sampling-Problem)
=#

# -----
# ## Ket Problem Templates

#=
#### Quantum State Smooth Pulse Problem
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
state_prob.trajectory.a |> size

#=
#### Quantum State Minimum Time Problem 
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

#=
#### Quantum State Sampling Problem
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
# ## Unitary Problem Templates

#=
#### Unitary Smooth Pulse Problem

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
println("Before: ", unitary_rollout_fidelity(prob.trajectory, system))

# _finding an optimal control is as simple as calling `solve!`_
solve!(prob, max_iter=100, verbose=true, print_level=1);

# _check the fidelity after solving_
println("After: ", unitary_rollout_fidelity(prob.trajectory, system))

#=
The `NamedTrajectory` object stores the control pulse, state variables, and the time grid.
=#

# _extract the control pulses_
prob.trajectory.a |> size

#=
#### Unitary Minimum Time Problem

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
solve!(min_prob, max_iter=100, verbose=true, print_level=1);

# _check the new duration_
println("Duration after: ", get_duration(min_prob.trajectory))

# _the fidelity is preserved by a constraint_
println("Fidelity after: ", unitary_rollout_fidelity(min_prob.trajectory, system))

#=
#### Unitary Sampling Problem 

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


