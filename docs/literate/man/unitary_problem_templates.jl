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

system = QuantumSystem(0.1 * PAULIS.Z, [PAULIS.X, PAULIS.Y], 10.0, [1.0, 1.0])
U_goal = EmbeddedOperator(GATES.H, system)
N = 51

prob = UnitarySmoothPulseProblem(system, U_goal, N); 

# _check the fidelity before solving_
println("Before: ", unitary_rollout_fidelity(prob.trajectory, system))

# _finding an optimal control is as simple as calling `solve!`_
solve!(prob, max_iter=100);

# _check the fidelity after solving_
println("After: ", unitary_rollout_fidelity(prob.trajectory, system))

#=
The `NamedTrajectory` object stores the control pulse, state variables, and the time grid.
=#

# _extract the control pulses_
prob.trajectory.u |> size

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
solve!(min_prob, max_iter=100);

# _check the new duration_
println("Duration after: ", get_duration(min_prob.trajectory))

# _the fidelity is preserved by a constraint_
println("Fidelity after: ", unitary_rollout_fidelity(min_prob.trajectory, system))

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
driftless_system = QuantumSystem([PAULIS.X, PAULIS.Y], 10.0, [1.0, 1.0])
sampling_prob = UnitarySamplingProblem([system, driftless_system], U_goal, T);

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
varsys = VariationalQuantumSystem([PAULIS.X, PAULIS.Y], [H_var], 10.0, [1.0, 1.0]);

# _create a variational problem that is robust to `PAULIS.X` at the end_
robprob = UnitaryVariationalProblem(varsys, U_goal, T, robust_times=[[T]]);

# -----