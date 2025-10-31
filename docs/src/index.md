# QuantumCollocation.jl

**QuantumCollocation.jl** is a Julia package for solving quantum optimal control problems using direct collocation methods. It transforms continuous-time quantum control into finite-dimensional nonlinear programs (NLPs) solved efficiently with [Ipopt](https://github.com/jump-dev/Ipopt.jl).

## Quick Start

### Installation

```julia
using Pkg
Pkg.add("QuantumCollocation")
```

Or from the Julia REPL, press `]` to enter package mode:
```
pkg> add QuantumCollocation
```

### 30-Second Example

Here's a complete example optimizing a Hadamard gate on a qubit:

```julia
using QuantumCollocation
using PiccoloQuantumObjects

# Define system: drift + 2 control Hamiltonians
H_drift = 0.1 * PAULIS.Z
H_drives = [PAULIS.X, PAULIS.Y]
sys = QuantumSystem(H_drift, H_drives, 10.0, [1.0, 1.0])

# Set up problem: system, target gate, timesteps
U_goal = GATES.H
N = 51
prob = UnitarySmoothPulseProblem(sys, U_goal, N)

# Solve!
solve!(prob; max_iter=100)

# Check result
println("Fidelity: ", unitary_rollout_fidelity(prob.trajectory, sys))
```

That's it! You've optimized control pulses for a quantum gate.

## What Can QuantumCollocation Do?

- **Unitary gate optimization** - Find pulses to implement quantum gates
- **State transfer** - Drive quantum states to target states
- **Minimum time control** - Optimize gate duration
- **Robust control** - Account for system uncertainties
- **Multilevel systems** - Handle transmons, bosonic codes, etc.
- **Leakage suppression** - Constrain populations in unwanted levels
- **Custom constraints** - Add your own physics constraints

## Overview

**QuantumCollocation.jl** sets up and solves *quantum control problems* as nonlinear programs (NLPs). A generic quantum control problem looks like:

```math
\begin{aligned}
    \arg \min_{\mathbf{Z}}\quad & J(\mathbf{Z}) \\
    \text{s.t.}\qquad & \mathbf{f}(\mathbf{Z}) = 0 \\
    & \mathbf{g}(\mathbf{Z}) \le 0
\end{aligned}
```

where $\mathbf{Z}$ is a trajectory containing states and controls from [NamedTrajectories.jl](https://github.com/harmoniqs/NamedTrajectories.jl).

We provide **problem templates** for common quantum control tasks. These templates construct a `DirectTrajOptProblem` from [DirectTrajOpt.jl](https://github.com/harmoniqs/DirectTrajOpt.jl) with appropriate objectives, constraints, and dynamics.

We provide **problem templates** for common quantum control tasks. These templates construct a `DirectTrajOptProblem` from [DirectTrajOpt.jl](https://github.com/harmoniqs/DirectTrajOpt.jl) with appropriate objectives, constraints, and dynamics.

## Problem Templates

Problem templates are organized by the type of quantum system being controlled:

### Unitary (Gate) Templates
- [`UnitarySmoothPulseProblem`](@ref) - Optimize smooth pulses for unitary gates
- [`UnitaryMinimumTimeProblem`](@ref) - Minimize gate duration
- [`UnitarySamplingProblem`](@ref) - Robust control over system variations
- [`UnitaryFreePhaseProblem`](@ref) - Optimize up to global phase
- [`UnitaryVariationalProblem`](@ref) - Variational quantum optimization

### Quantum State Templates
- [`QuantumStateSmoothPulseProblem`](@ref) - Drive states with smooth pulses
- [`QuantumStateMinimumTimeProblem`](@ref) - Minimize state transfer time
- [`QuantumStateSamplingProblem`](@ref) - Robust state transfer

See the [Problem Templates Overview](@ref) for a detailed comparison and selection guide.

See the [Problem Templates Overview](@ref) for a detailed comparison and selection guide.

## How It Works

### Direct Collocation

QuantumCollocation uses *direct collocation* - discretizing continuous-time dynamics into constraints at discrete time points (knot points). For example, a smooth pulse problem for a unitary gate:

```math
\begin{aligned}
    \arg \min_{\mathbf{Z}}\quad & |1 - \mathcal{F}(U_N, U_\text{goal})|  \\
    \text{s.t.}\qquad & U_{k+1} = \exp\{- i H(u_k) \Delta t_k \} U_k, \quad \forall\, k \\
    & u_{k+1} = u_k + \dot{u}_k \Delta t_k \\
    & \dot{u}_{k+1} = \dot{u}_k + \ddot{u}_k \Delta t_k \\
    & |u_k| \le u_\text{max}, \quad |\ddot{u}_k| \le \ddot{u}_\text{max}
\end{aligned}
```

The dynamics between knot points $(U_k, u_k)$ and $(U_{k+1}, u_{k+1})$ become nonlinear equality constraints. States and controls are free variables optimized by the NLP solver.

### Key Features

- **Efficient gradients** - Sparse Jacobians and Hessians via automatic differentiation
- **Flexible constraints** - Add custom physics, leakage suppression, robustness
- **Multiple integrators** - Exponential, Pade, time-dependent dynamics
- **Extensible** - Easy to add new objectives, constraints, and problem templates

## Next Steps

- 📚 [Problem Templates Overview](@ref) - Choose the right template for your problem
- 🎯 [Working with Solutions](@ref) - Extract results, evaluate fidelity, save data
- ⚙️ [PiccoloOptions Reference](@ref) - Configure solver options and constraints
- 💡 [Examples](@ref) - See complete examples from single qubits to multilevel systems

## Related Packages

QuantumCollocation.jl is part of the [Piccolo ecosystem](https://github.com/harmoniqs/Piccolo.jl):

- [**NamedTrajectories.jl**](https://github.com/harmoniqs/NamedTrajectories.jl) - Trajectory data structures
- [**DirectTrajOpt.jl**](https://github.com/harmoniqs/DirectTrajOpt.jl) - Direct trajectory optimization framework
- [**PiccoloQuantumObjects.jl**](https://github.com/harmoniqs/PiccoloQuantumObjects.jl) - Quantum operators and systems
- [**PiccoloPlots.jl**](https://github.com/harmoniqs/PiccoloPlots.jl) - Visualization tools

Problem templates give the user the ability to add other constraints and objective functions to this problem and solve it efficiently using [Ipopt.jl](https://github.com/jump-dev/Ipopt.jl) and [MathOptInterface.jl](https://github.com/jump-dev/MathOptInterface.jl) under the hood (support for additional backends coming soon!).
