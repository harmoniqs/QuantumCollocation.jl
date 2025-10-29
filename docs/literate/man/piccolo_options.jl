# # PiccoloOptions Reference

# `PiccoloOptions` provides advanced configuration for problem templates. This page documents all available options and their effects.

using QuantumCollocation
using PiccoloQuantumObjects

# ## Creating PiccoloOptions

# Default options:
opts = PiccoloOptions()

# Custom options:
opts_custom = PiccoloOptions(
    verbose = true,
    leakage_constraint = true,
    bound_state = true
)

# Pass to any problem template:
system = QuantumSystem(0.1 * PAULIS.Z, [PAULIS.X, PAULIS.Y], 10.0, [1.0, 1.0])
U_goal = EmbeddedOperator(GATES.H, system)
T = 51

prob = UnitarySmoothPulseProblem(
    system, U_goal, T;
    piccolo_options = opts_custom
)

# ## General Options

# ### `verbose::Bool = false`
# Print detailed information during problem setup.

opts_verbose = PiccoloOptions(verbose = true)
# Shows: constraint counts, objective terms, trajectory initialization details

# ### `free_time::Bool = false`
# Allow timesteps to vary (used internally by minimum time problems).
# Typically not set directly by users.

# ## State and Unitary Options

# ### `bound_state::Bool = false`
# Constrain state/unitary to lie on unit sphere (for numerical stability).
# **Recommended for difficult convergence.**

opts_bounded = PiccoloOptions(bound_state = true)
# Adds constraint: ||ψ||² = 1 (ket) or ||U||² = N (unitary)

# ### `geodesic::Bool = true`
# Use geodesic interpolation for initial trajectory.
# - `true`: States evolve along shortest path on manifold (better initial guess)
# - `false`: Linear interpolation (simpler but often worse)

opts_linear = PiccoloOptions(geodesic = false)

# ## Control Initialization

# ### `init_trajectory::Union{NamedTrajectory, Nothing} = nothing`
# Provide custom initial trajectory instead of automatic initialization.

# traj_custom = initialize_trajectory(...)
# opts_init = PiccoloOptions(init_trajectory = traj_custom)

# ### `build_trajectory_constraints::Bool = true`
# Automatically extract constraints from trajectory bounds.
# Set to `false` if manually managing constraints.

# ## Leakage Suppression (Multilevel Systems)

# ### `leakage_constraint::Bool = false`
# Add constraint to limit leakage population.

# ### `leakage_constraint_value::Float64 = 1e-3`
# Maximum allowed leakage: ∑ᵢ |⟨i|ψ⟩|² ≤ leakage_constraint_value

# ### `leakage_cost::Float64 = 1.0`
# Penalty weight for leakage in objective (soft constraint).

opts_leakage = PiccoloOptions(
    leakage_constraint = true,
    leakage_constraint_value = 1e-2,  # 1% max leakage
    leakage_cost = 1e-1
)

# Example with embedded operator:
# sys_transmon = TransmonSystem(levels=5, δ=0.2)
# op = EmbeddedOperator(:X, sys_transmon)
# prob_leakage = UnitarySmoothPulseProblem(
#     sys_transmon, op, T, Δt;
#     piccolo_options = opts_leakage
# )

# ## Control Constraints

# ### `complex_control_norm_constraint_name::Union{Symbol, Nothing} = nothing`
# Apply norm constraint to complex control amplitudes.

# For systems with complex drives (e.g., rotating frame):
# opts_norm = PiccoloOptions(
#     complex_control_norm_constraint_name = :u,
#     complex_control_norm_constraint_radius = 0.2
# )
# Enforces: |u_real + i*u_imag| ≤ radius

# ### `complex_control_norm_constraint_radius::Float64 = 1.0`
# Radius for complex control norm constraint.

# ## Timestep Options

# ### `timesteps_all_equal::Bool = false`
# Force all timesteps to be equal: Δt[k] = Δt[1] ∀k.
# Useful for hardware with fixed sampling rates.

opts_equal_dt = PiccoloOptions(timesteps_all_equal = true)

# ## Advanced Dynamics

# ### `rollout_integrator::Symbol = :pade`
# Integration method for evaluating fidelity.
# - `:pade`: Padé approximation (default, fast)
# - `:exp`: Matrix exponential (more accurate)

opts_exp = PiccoloOptions(rollout_integrator = :exp)

# ## Derivative Constraints

# ### `zero_initial_and_final_derivative::Bool = false`
# Force derivatives to zero at boundaries: u̇[1] = u̇[T] = 0, ü[1] = ü[T] = 0.
# Creates "smooth ramp" pulses that start and end at zero derivative.

opts_smooth_edges = PiccoloOptions(zero_initial_and_final_derivative = true)

# ## Common Configuration Patterns

# ### Quick and dirty optimization
opts_quick = PiccoloOptions(
    verbose = false,
    geodesic = true
)

# ### High-fidelity gate
opts_hifi = PiccoloOptions(
    verbose = true,
    bound_state = true,
    geodesic = true,
    rollout_integrator = :exp
)

# ### Multilevel system with leakage suppression
opts_multilevel = PiccoloOptions(
    bound_state = true,
    leakage_constraint = true,
    leakage_constraint_value = 1e-2,
    leakage_cost = 1e-1,
    verbose = true
)

# ### Smooth pulses for hardware
opts_hardware = PiccoloOptions(
    zero_initial_and_final_derivative = true,
    timesteps_all_equal = true,
    bound_state = true
)

# ### Robust optimization
opts_robust = PiccoloOptions(
    bound_state = true,
    geodesic = true,
    verbose = true
)

# ## Tips and Tricks

# **When to use `bound_state=true`:**
# - Optimization struggling to converge
# - Fidelity stuck below 0.99
# - Numerical instabilities in state evolution
# - Working with large Hilbert spaces

# **Leakage vs bounds:**
# - Leakage *constraint* enforces hard limit (may fail to converge)
# - Leakage *cost* adds soft penalty (more forgiving)
# - Use both for best results

# **Geodesic initialization:**
# - Almost always better than linear
# - Only disable for debugging or special cases
# - Particularly important for large T

# **Rollout integrator:**
# - `:pade` is fast and usually sufficient
# - `:exp` more accurate for sensitive systems
# - Both give same result for well-conditioned problems

println("PiccoloOptions configured!")
