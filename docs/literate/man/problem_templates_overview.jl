# # Problem Templates Overview

# QuantumCollocation.jl provides **4 problem templates** that cover common quantum optimal control scenarios. These templates make it easy to set up and solve problems without manually constructing objectives, constraints, and integrators.
# ## Template Comparison


# | Template | Objective | Time | Use Case |
# |:---------|:-----------|:-----|:---------|
# | [`SmoothPulseProblem`](@ref) | Minimize control effort + infidelity | Fixed | Standard gate/state synthesis with smooth pulses |
# | [`MinimumTimeProblem`](@ref) | Minimize duration | Variable | Fastest gate/state synthesis given fidelity constraint |
# | [`SplinePulseProblem`](@ref) | Minimize control effort + infidelity | Fixed | Gate/state synthesis with spline-based pulses (linear or cubic Hermite) |
# | [`SamplingProblem`](@ref) | Minimize control effort + weighted sum of infidelity objectives | Fixed | Robust gate/state synthesis where the controls are shared across all systems, with differing dynamics. |

# ### Smooth Pulse vs Minimum Time
# - **Smooth Pulse**: Fixed total time `T × Δt`, minimizes control effort with regularization on `u`, `u̇`, `ü`
# - **Minimum Time**: Variable timesteps `Δt[k]`, minimizes total duration subject to fidelity constraint

# ### Sampling Problems
# - Solve for a **single control pulse** that works well across **multiple quantum systems**
# - Useful for robustness against parameter uncertainties or manufacturing variations
# - Examples: different coupling strengths, detunings, or environmental conditions

# ## Quick Selection Guide

# **I want to implement a quantum gate:**
# - Start simple? → [`SmoothPulseProblem`](@ref) + `UnitaryTrajectory`
# - Need speed? → [`MinimumTimeProblem`](@ref) + `UnitaryTrajectory`
# - Need robustness? → [`SamplingProblem`](@ref) + `UnitaryTrajectory`

# **I want to prepare a quantum state:**
# - Standard case? → [`SmoothPulseProblem`](@ref) + `KetTrajectory`
# - Speed critical? → [`MinimumTimeProblem`](@ref) + `KetTrajectory`
# - Robust preparation? → [`SamplingProblem`](@ref) + `KetTrajectory`

# ## Common Parameters

# All templates share these key parameters:

using QuantumCollocation # hide
using PiccoloQuantumObjects # hide
H_drift = 0.1 * PAULIS.Z # hide
H_drives = [PAULIS.X, PAULIS.Y] # hide
drive_bounds = [1.0, 1.0]  # hide
sys = QuantumSystem(H_drift, H_drives, drive_bounds) # hide
U_goal = GATES[:H] # hide
T = 10.0  # hide
qtraj = UnitaryTrajectory(sys, U_goal, T) # hide
N = 51 # hide

prob = SmoothPulseProblem(
    qtraj,              # QuantumTrajectory wrapping system information, Unitary/Ket/MultiKet problem type
    N;                  # Number of timesteps 

    Q=100.0,            # Objective weighting coefficient for the infidelity
    R=1e-2,             # Objective weighting coefficient for the controls regularization
    
    piccolo_options = PiccoloOptions(verbose = true),  # PiccoloOptions for solver configuration
)

# See the individual template pages for parameter details and examples.
