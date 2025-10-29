# # Problem Templates Overview

# QuantumCollocation.jl provides **8 problem templates** that cover common quantum optimal control scenarios. These templates make it easy to set up and solve problems without manually constructing objectives, constraints, and integrators.

# ## Template Comparison

# | Template | State Type | Objective | Time | Use Case |
# |:---------|:-----------|:----------|:-----|:---------|
# | [`UnitarySmoothPulseProblem`](@ref) | Unitary | Minimize control effort + infidelity | Fixed | Standard gate synthesis with smooth pulses |
# | [`UnitaryMinimumTimeProblem`](@ref) | Unitary | Minimize duration | Variable | Fastest gate given fidelity constraint |
# | [`UnitarySamplingProblem`](@ref) | Unitary | Minimize control effort + infidelity | Fixed | Robust control over multiple systems |
# | [`UnitaryFreePhase Problem`](@ref) | Unitary | Minimize control effort + infidelity | Fixed | Gate synthesis with free global phase |
# | [`UnitaryVariationalProblem`](@ref) | Unitary | Minimize control effort + infidelity ± sensitivity | Fixed | Sensitivity/robustness to Hamiltonian terms |
# | [`QuantumStateSmoothPulseProblem`](@ref) | Ket | Minimize control effort + infidelity | Fixed | State transfer with smooth pulses |
# | [`QuantumStateMinimumTimeProblem`](@ref) | Ket | Minimize duration | Variable | Fastest state transfer |
# | [`QuantumStateSamplingProblem`](@ref) | Ket | Minimize control effort + infidelity | Fixed | Robust state transfer over multiple systems |

# ## Key Differences

# ### Unitary vs Ket (Quantum State)
# - **Unitary problems**: Optimize gate operations (full unitary matrices), commonly used for universal quantum control
# - **Ket problems**: Optimize state-to-state transfers, useful for initialization and specific state preparation

# ### Smooth Pulse vs Minimum Time
# - **Smooth Pulse**: Fixed total time `T × Δt`, minimizes control effort with regularization on `u`, `u̇`, `ü`
# - **Minimum Time**: Variable timesteps `Δt[k]`, minimizes total duration subject to fidelity constraint

# ### Sampling Problems
# - Solve for a **single control pulse** that works well across **multiple quantum systems**
# - Useful for robustness against parameter uncertainties or manufacturing variations
# - Examples: different coupling strengths, detunings, or environmental conditions

# ### Free Phase & Variational
# - **Free Phase**: Optimizes global phase of target unitary (sometimes easier to reach)
# - **Variational**: Uses sensitivity analysis to find controls that are robust or sensitive to specific Hamiltonian terms

# ## Quick Selection Guide

# **I want to implement a quantum gate:**
# - Start simple? → `UnitarySmoothPulseProblem`
# - Need speed? → `UnitaryMinimumTimeProblem`
# - Need robustness? → `UnitarySamplingProblem`

# **I want to prepare a quantum state:**
# - Standard case? → `QuantumStateSmoothPulseProblem`
# - Speed critical? → `QuantumStateMinimumTimeProblem`
# - Robust preparation? → `QuantumStateSamplingProblem`

# **I'm tuning my solution:**
# - Struggling with convergence? → Try `UnitaryFreePhase Problem`
# - Need parameter sensitivity? → Use `UnitaryVariationalProblem`

# ## Common Parameters

# All templates share these key parameters:

# ```julia
# prob = UnitarySmoothPulseProblem(
#     system,           # QuantumSystem defining H(u)
#     U_goal,           # Target unitary or state
#     T,                # Number of timesteps
#     Δt;               # Timestep duration (or initial guess for min-time)
#     
#     # Control bounds
#     u_bound = 1.0,              # |u| ≤ u_bound
#     u_bounds = [...],           # Per-drive bounds
#     
#     # Derivative bounds (smoothness)
#     du_bound = 0.01,            # |u̇| ≤ du_bound
#     ddu_bound = 0.001,          # |ü| ≤ ddu_bound
#     
#     # Regularization weights
#     R_u = 0.01,                 # Penalize u²
#     R_du = 0.01,                # Penalize u̇²
#     R_ddu = 0.01,               # Penalize ü²
#     
#     # Initial guess
#     u_guess = nothing,          # Optional initial controls
#     
#     # Advanced options
#     piccolo_options = PiccoloOptions(...)
# )
# ```

# See the individual template pages for parameter details and examples.
