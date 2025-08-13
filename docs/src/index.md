# QuantumCollocation.jl

**QuantumCollocation.jl** sets up and solves *quantum control problems* as nonlinear programs (NLPs). In this context, a generic quantum control problem looks like
```math
\begin{aligned}
    \arg \min_{\mathbf{Z}}\quad & J(\mathbf{Z}) \\
    \nonumber \text{s.t.}\qquad & \mathbf{f}(\mathbf{Z}) = 0 \\
    \nonumber & \mathbf{g}(\mathbf{Z}) \le 0  
\end{aligned}
```
where $\mathbf{Z}$ is a trajectory  containing states and controls, from [NamedTrajectories.jl](https://github.com/harmoniqs/NamedTrajectories.jl).

-----

We provide a number of **problem templates** for making it simple and easy to set up and solve 
certain types of quantum optimal control problems. These templates all construct a 
`DirectTrajOptProblem` object from [DirectTrajOpt.jl](https://github.com/harmoniqs/DirectTrajOpt.jl), which stores all the parts of the optimal control problem.

-----

### Get started

The problem templates are broken down by the state variable of the problem being solved.

Ket Problem Templates:
- [Quantum State Smooth Pulse Problem](@ref)
- [Quantum State Minimum Time Problem](@ref)
- [Quantum State Sampling Problem](@ref)

Unitary Problem Templates:
- [Unitary Smooth Pulse Problem](@ref)
- [Unitary Minimum Time Problem](@ref)
- [Unitary Sampling Problem](@ref)

### Background

*Problem Templates* are reusable design patterns for setting up and solving common quantum control problems. 

For example, a *UnitarySmoothPulseProblem* is tasked with generating a *pulse* sequence $a_{1:T-1}$ in orderd to minimize infidelity, subject to constraints from the Schroedinger equation,
```math
    \begin{aligned}
        \arg \min_{\mathbf{Z}}\quad & |1 - \mathcal{F}(U_T, U_\text{goal})|  \\
        \nonumber \text{s.t.}
        \qquad & U_{t+1} = \exp\{- i H(a_t) \Delta t_t \} U_t, \quad \forall\, t \\
    \end{aligned}
```
while a *UnitaryMinimumTimeProblem* minimizes time and constrains fidelity,
```math
    \begin{aligned}
        \arg \min_{\mathbf{Z}}\quad & \sum_{t=1}^T \Delta t_t \\
        \qquad & U_{t+1} = \exp\{- i H(a_t) \Delta t_t \} U_t, \quad \forall\, t \\
        \nonumber & \mathcal{F}(U_T, U_\text{goal}) \ge 0.9999
    \end{aligned}
```

-----

In each case, the dynamics between *knot points* $(U_t, a_t)$ and $(U_{t+1}, a_{t+1})$ are enforced as constraints on the states, which are free variables in the solver; this optimization framework is called *direct trajectory optimization*. 

-----

Problem templates give the user the ability to add other constraints and objective functions to this problem and solve it efficiently using [Ipopt.jl](https://github.com/jump-dev/Ipopt.jl) and [MathOptInterface.jl](https://github.com/jump-dev/MathOptInterface.jl) under the hood (support for additional backends coming soon!).
