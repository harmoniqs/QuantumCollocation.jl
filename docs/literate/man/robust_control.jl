# # Robust Quantum Control with SamplingProblem

# This guide explains how to optimize control pulses that are **robust to parameter uncertainty**
# using `SamplingProblem` and `SamplingTrajectory`.

# ## The Problem: Parameter Uncertainty

# In real quantum hardware, you don't know exact system parameters:
# - Qubit frequencies drift over time
# - Coupling strengths vary between devices
# - Calibration has uncertainty

# A pulse optimized for the "nominal" system may fail on the actual hardware.

# ## The Solution: Robust Optimization

# `SamplingProblem` optimizes a **single control pulse** that works well across 
# **multiple systems** with parameter variations:

# ```
#                     ┌──────────────┐
#                     │   System 1   │──→ Fidelity₁
#                     │  (nominal)   │
#                     └──────────────┘
#                            ↑
#     Shared Controls ───────┼──────────────────
#                            ↓
#                     ┌──────────────┐
#                     │   System 2   │──→ Fidelity₂
#                     │  (+5% drift) │
#                     └──────────────┘
#                            ↓
#                     ┌──────────────┐
#                     │   System 3   │──→ Fidelity₃
#                     │  (-5% drift) │
#                     └──────────────┘
#
#     Objective = w₁·Infidelity₁ + w₂·Infidelity₂ + w₃·Infidelity₃
# ```

# ## Key Features

# 1. **Shared controls** - One pulse `u(t)` applied to all systems
# 2. **Separate dynamics** - Each system evolves independently with its own Hamiltonian
# 3. **Weighted objective** - Minimize weighted average infidelity across all systems
# 4. **Same goal** - All systems target the same gate/state transfer

# ## Two Workflow Options

# There are two ways to set up robust optimization, depending on your starting point.

# ### Option 1: Transform an Existing Problem (Recommended)

# Use `SamplingProblem(qcp, systems)` when you already have a solved base problem:

using QuantumCollocation
using PiccoloQuantumObjects
using LinearAlgebra

# Define the nominal system
H_drift = 0.01 * GATES[:Z]
H_drives = [GATES[:X], GATES[:Y]]
T = 10.0
N = 51

sys_nominal = QuantumSystem(H_drift, H_drives)

# Create and solve base problem first
pulse = ZeroOrderPulse(0.1 * randn(2, N), collect(range(0.0, T, length=N)))
qtraj = UnitaryTrajectory(sys_nominal, pulse, GATES[:H])
qcp = SmoothPulseProblem(qtraj, N; Q=100.0, R=1e-2)

solve!(qcp; max_iter=100, verbose=false)

# Now create system variations (±5% frequency drift)
sys_plus5 = QuantumSystem(1.05 * H_drift, H_drives)
sys_minus5 = QuantumSystem(0.95 * H_drift, H_drives)
systems = [sys_nominal, sys_plus5, sys_minus5]

# Transform to robust problem - uses previous solution as warm start
sampling_qcp = SamplingProblem(qcp, systems; 
    weights=[0.5, 0.25, 0.25]  # More weight on nominal
)

solve!(sampling_qcp; max_iter=100, verbose=false)

# This workflow has two advantages:
# 1. **Warm start**: The robust problem starts from a good solution
# 2. **Faster convergence**: The base problem is easier to solve

# ### Option 2: Build from Scratch with SamplingTrajectory

# Use `SamplingTrajectory` directly when you want to build a robust problem from the start.
# This is useful when you don't need a warm start:

## Create SamplingTrajectory wrapper
sampling_qtraj = SamplingTrajectory(qtraj, systems; 
    weights=[0.5, 0.25, 0.25]
)

## The trajectory is then converted to NamedTrajectory for optimization
traj = NamedTrajectory(sampling_qtraj, N)
println("State variables: ", [n for n in traj.names if startswith(string(n), "Ũ")])

# ## Comparison of Workflows

# | Aspect | `SamplingProblem(qcp, systems)` | Direct `SamplingTrajectory` |
# |--------|--------------------------------|------------------------------|
# | **Initialization** | Uses solved trajectory | Starts from pulse initial guess |
# | **Workflow** | Two-stage (solve then robustify) | Single-stage |
# | **Convergence** | Usually faster (warm start) | Can be slower |
# | **Use case** | Refining existing solutions | Fresh robust optimization |

# ## Under the Hood

# The trajectory created for sampling has **separate state variables** for each system,
# but only **one set of controls**:

println("Trajectory components:")
for (name, comp) in pairs(traj.components)
    println("  :$name → indices $(comp)")
end

# You'll see:
# - `:Ũ⃗1`, `:Ũ⃗2`, `:Ũ⃗3` - separate unitary evolution for each system
# - `:u` - shared control pulse
# - `:du`, `:ddu` - control derivatives (if using SmoothPulseProblem)

# ## Choosing Weights

# The `weights` parameter controls how much each system contributes to the objective:

# ```julia
# weights = [0.5, 0.25, 0.25]  # 50% nominal, 25% each variation
# ```

# **Guidelines:**
# - **Equal weights** (`nothing` or uniform): Treat all systems equally important
# - **Higher nominal weight**: Prioritize performance on expected parameters
# - **Higher variation weights**: More conservative, sacrifices nominal for robustness

# ## Advanced: Custom System Variations

# You can create any parameter variations relevant to your hardware:

## Frequency uncertainty
freq_variations = [0.95, 1.0, 1.05]
freq_systems = [QuantumSystem(f * H_drift, H_drives) for f in freq_variations]

## Coupling strength uncertainty  
coupling_variations = [0.9, 1.0, 1.1]
coupling_systems = [QuantumSystem(H_drift, c .* H_drives) for c in coupling_variations]

## Combined (grid sampling)
combined_systems = [
    QuantumSystem(f * H_drift, c .* H_drives)
    for f in freq_variations
    for c in coupling_variations
]
println("Combined grid: $(length(combined_systems)) systems")

# ## Tips for Robust Optimization

# 1. **Start with fewer systems** - Begin with 3-5 samples, add more if needed
# 2. **Use meaningful variations** - Base variations on actual hardware uncertainty
# 3. **Balance weights** - Don't over-weight rare parameter values
# 4. **Check individual fidelities** - After optimization, verify performance on each system
# 5. **Consider compute cost** - More systems = larger problem = slower optimization

# ## See Also

# - [`SmoothPulseProblem`](@ref) - Base smooth pulse optimization
# - [`MinimumTimeProblem`](@ref) - Time-optimal control
# - [Problem Templates Overview](@ref) - All available templates
