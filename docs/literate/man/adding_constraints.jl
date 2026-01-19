# # Adding Custom Constraints

# QuantumCollocation.jl provides problem templates with built-in constraints, but you often need to add custom constraints for specific physics or hardware requirements.

# This guide shows you how to add:
# - Path constraints (applied at every timestep)
# - Boundary constraints (initial/final conditions)
# - Nonlinear constraints (complex physics)
# - Leakage suppression
# - Control envelope constraints

using QuantumCollocation
using PiccoloQuantumObjects
using NamedTrajectories
using DirectTrajOpt

# ## Basic Setup

# Start with a simple system:

H_drift = 0.1 * PAULIS.Z
H_drives = [PAULIS.X, PAULIS.Y]
sys = QuantumSystem(H_drift, H_drives, [1.0, 1.0])
U_goal = GATES.H
N = 51

# ## 1. Path Constraints (Box Constraints)

# The simplest constraints are bounds on variables at all timesteps.

# ### Built-in Bounds

# Problem templates already handle basic bounds via the trajectory:

prob = UnitarySmoothPulseProblem(sys, U_goal, N)

# Control bounds come from `sys.drive_bounds`:
println("Control bounds from system: ", sys.drive_bounds)

# Derivative bounds specified in problem:
prob_with_bounds = UnitarySmoothPulseProblem(sys, U_goal, N;
    du_bound=0.5,    # First derivative bound
    ddu_bound=0.1    # Second derivative bound
)

# ### Custom Variable Bounds

# To add bounds on other trajectory variables:

# Create trajectory with custom bounds
T = 10.0  # Total duration
traj = initialize_unitary_trajectory(
    U_goal,
    N,
    T / N,
    sys.n_drives,
    (sys.drive_bounds, fill((-0.5, 0.5), sys.n_drives), fill((-0.1, 0.1), sys.n_drives));
    state_name=:Ũ⃗,
    control_name=:u,
)

# Add a custom bounded variable
traj_extended = NamedTrajectory(
    merge(traj.data, Dict(:custom_var => randn(1, N))),
    controls=traj.controls,
    timestep=traj.timestep,
    bounds=merge(traj.bounds, Dict(:custom_var => [(-1.0, 1.0)])),
)

# ## 2. Linear Equality Constraints

# Use `EqualityConstraint` for linear constraints of form `A*z = b`:

using LinearAlgebra

# Example: Force controls to sum to zero at each timestep
# (useful for differential drive schemes)

prob = UnitarySmoothPulseProblem(sys, U_goal, N)

# Get trajectory dimensions
n_drives = sys.n_drives
n_timesteps = N

# Create constraint: u1(t) + u2(t) = 0 for all t
constraints = AbstractConstraint[]

for k in 1:n_timesteps
    # Get indices for u1 and u2 at timestep k
    u1_idx = (k-1) * n_drives + 1
    u2_idx = (k-1) * n_drives + 2
    
    # A * [u1; u2] = 0, where A = [1, 1]
    A = zeros(1, prob.trajectory.dim * N)
    A[1, u1_idx] = 1.0
    A[1, u2_idx] = 1.0
    
    constraint = EqualityConstraint(A, [0.0], [k])
    push!(constraints, constraint)
end

# Create problem with constraints
prob_with_constraints = DirectTrajOptProblem(
    prob.trajectory,
    prob.objective,
    prob.dynamics;
    constraints=vcat(prob.constraints, constraints)
)

# ## 3. Nonlinear Path Constraints

# For nonlinear constraints applied at each timestep, use `NonlinearKnotPointConstraint`:

# Example: Limit total control power: √(u₁² + u₂²) ≤ P_max

P_max = 0.8

# Define constraint function
function power_constraint(Z⃗, k, traj)
    # Extract controls at timestep k
    u_slice = slice(k, :u, traj.dims)
    u = Z⃗[u_slice]
    
    # Return constraint value (should be ≤ 0)
    return sqrt(sum(u.^2)) - P_max
end

# Create the constraint
power_constr = NonlinearKnotPointConstraint(
    power_constraint,
    prob.trajectory,
    1:N;  # Apply at all timesteps
    name=:power_limit
)

# Add to problem
prob_power = DirectTrajOptProblem(
    prob.trajectory,
    prob.objective,
    prob.dynamics;
    constraints=vcat(prob.constraints, [power_constr])
)

# ## 4. Nonlinear Global Constraints

# For constraints over the entire trajectory, use `NonlinearGlobalConstraint`:

# Example: Limit total energy: ∫ u²(t) dt ≤ E_budget

E_budget = 10.0

function energy_constraint(Z⃗, traj)
    total_energy = 0.0
    
    for k in 1:traj.T
        u_slice = slice(k, :u, traj.dims)
        Δt_slice = slice(k, :Δt, traj.dims)
        
        u = Z⃗[u_slice]
        Δt = Z⃗[Δt_slice][1]
        
        total_energy += sum(u.^2) * Δt
    end
    
    return total_energy - E_budget
end

energy_constr = NonlinearGlobalConstraint(
    energy_constraint,
    prob.trajectory;
    name=:energy_budget
)

# ## 5. Leakage Suppression (Multilevel Systems)

# For multilevel systems, suppress population in leakage levels using `PiccoloOptions`:

# Define a 3-level system (qubit + 1 leakage level)
a = annihilate(3)
H_drives_3level = [(a + a')/2, (a - a')/(2im)]
sys_3level = QuantumSystem(H_drives_3level, [1.0, 1.0])

# Target: X gate in computational subspace
U_goal_embedded = EmbeddedOperator(GATES.X, sys_3level)

# Enable leakage suppression
opts = PiccoloOptions(
    leakage_constraint=true,
    leakage_constraint_value=1e-3,  # Maximum leakage population
    leakage_cost=1.0                 # Penalty weight
)

prob_leakage = UnitarySmoothPulseProblem(
    sys_3level,
    U_goal_embedded,
    N;
    piccolo_options=opts
)

solve!(prob_leakage; max_iter=100)

# Check leakage:
# (Leakage evaluation code would go here)

# ## 6. Soft Constraints vs Hard Constraints

# **Hard constraints** (equality/inequality):
# - Must be satisfied exactly
# - Can make problem infeasible
# - Use for physics requirements

# **Soft constraints** (penalties in objective):
# - Violated at a cost
# - Always feasible
# - Use for preferences

# Example: Add soft constraint via objective penalty

# Soft control smoothness (already built-in via R_u, R_du, R_ddu)
prob_smooth = UnitarySmoothPulseProblem(sys, U_goal, N;
    R_u=0.01,    # Control amplitude penalty
    R_du=0.1,    # First derivative penalty (smoothness)
    R_ddu=1.0    # Second derivative penalty (extra smooth)
)

# ## 7. State Constraints (Avoid Forbidden Regions)

# Example: Keep qubit in upper hemisphere of Bloch sphere

function upper_hemisphere_constraint(Z⃗, k, traj)
    # Extract state at timestep k
    state_slice = slice(k, :Ũ⃗, traj.dims)
    Ũ⃗ = Z⃗[state_slice]
    
    # Convert to unitary (implementation depends on representation)
    # For isomorphic vectors, extract Z expectation value
    # Here's conceptual code - actual implementation depends on system
    
    # Return constraint: should keep ⟨Z⟩ > 0
    # return -Z_expectation  # Example
    
    return 0.0  # Placeholder
end

# ## 8. Time-Dependent Constraints

# Example: Ramp control amplitude over time

function amplitude_ramp_constraint(Z⃗, k, traj)
    # Maximum amplitude increases linearly with time
    max_amp = 0.5 * (k / traj.T)  # Ramp from 0 to 0.5
    
    u_slice = slice(k, :u, traj.dims)
    u = Z⃗[u_slice]
    
    # All controls should be below max_amp
    return maximum(abs.(u)) - max_amp
end

ramp_constr = NonlinearKnotPointConstraint(
    amplitude_ramp_constraint,
    prob.trajectory,
    1:N;
    name=:amplitude_ramp
)

# ## 9. Coupled State-Control Constraints

# Example: Control amplitude proportional to state fidelity
# (more aggressive control when far from target)

function adaptive_control_constraint(Z⃗, k, traj)
    state_slice = slice(k, traj.state, traj.dims)
    u_slice = slice(k, :u, traj.dims)
    
    # Get distance from target (simplified)
    # state = Z⃗[state_slice]
    # distance = ... (compute distance metric)
    
    u = Z⃗[u_slice]
    
    # Constraint: |u| ≤ f(distance)
    # return maximum(abs.(u)) - constraint_function(distance)
    
    return 0.0  # Placeholder
end

# ## Key Takeaways

# 1. **Start with problem templates** - they include sensible defaults
# 2. **Use PiccoloOptions** for common constraints (leakage, robustness)
# 3. **EqualityConstraint** for linear constraints
# 4. **NonlinearKnotPointConstraint** for per-timestep nonlinear constraints
# 5. **NonlinearGlobalConstraint** for trajectory-wide constraints
# 6. **Soft constraints** (objectives) are more robust than hard constraints
# 7. **Test feasibility** - start with loose constraints, then tighten
#
# ## Debugging Constraints
#
# If your problem becomes infeasible:
# 1. Check constraint values before optimization
# 2. Relax bounds temporarily
# 3. Use soft constraints instead of hard
# 4. Verify constraint gradients (use finite differences)
# 5. Start from a feasible initial trajectory
#
# ## Next Steps
#
# - [Custom Objectives](@ref) - Add custom cost functions
# - [Initial Trajectories](@ref) - Better initialization strategies
# - [PiccoloOptions Reference](@ref) - Built-in constraint options
