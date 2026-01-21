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
drive_bounds = [1.0, 1.0]  # symmetric bounds
sys = QuantumSystem(H_drift, H_drives, drive_bounds)

# 2. Create quantum trajectory. defines problem: system, target gate, timesteps
U_goal = GATES[:H]
T = 10.0
qtraj = UnitaryTrajectory(sys, U_goal, T) # creates zero pulse internally

# 3. Build optimization problem
N = 51  # number of timesteps
qcp = SmoothPulseProblem(qtraj, N; Q=100.0, R=1e-2)

# Solve!
solve!(qcp; options=IpoptOptions(max_iter=100))

# Check result
traj = get_trajectory(qcp)
println("Fidelity: ", fidelity(qcp))
```

That's it! You've optimized control pulses for a quantum gate.

## What Can QuantumCollocation Do?

- **Unitary gate optimization** - Find pulses to implement quantum gates
- **Open quantum systems** - Find pulses for Lindbladian dynamics 
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

## Problem Templates

Problem templates are organized by the type of quantum system being controlled:

### General Problem Templates
- [`MinimumTimeProblem`](@ref) - Minimize gate duration
- [`SamplingProblem`](@ref) - Robust control over system variations
- [`SmoothPulseProblem`](@ref) - Optimize smooth pulses for unitary gates
- [`SplinePulseProblem`](@ref) - Using higher order splines to characterize pulse shape


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
- 💡 [Two Qubit Gates](@ref), [Single Qubit Gate](@ref) - See complete examples from single qubits to multilevel systems (**MOVING TO PICCOLO DOCS**)

## Related Packages

QuantumCollocation.jl is part of the [Piccolo ecosystem](https://github.com/harmoniqs/Piccolo.jl):

- [**NamedTrajectories.jl**](https://github.com/harmoniqs/NamedTrajectories.jl) - Trajectory data structures
- [**DirectTrajOpt.jl**](https://github.com/harmoniqs/DirectTrajOpt.jl) - Direct trajectory optimization framework
- [**PiccoloQuantumObjects.jl**](https://github.com/harmoniqs/PiccoloQuantumObjects.jl) - Quantum operators and systems
- [**PiccoloPlots.jl**](https://github.com/harmoniqs/PiccoloPlots.jl) - Visualization tools

Problem templates give the user the ability to add other constraints and objective functions to this problem and solve it efficiently using [Ipopt.jl](https://github.com/jump-dev/Ipopt.jl) and [MathOptInterface.jl](https://github.com/jump-dev/MathOptInterface.jl) under the hood (support for additional backends coming soon!).
