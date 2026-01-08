# # Custom Objectives and Regularization

# Beyond the default fidelity objective, you often want to customize the cost function to achieve specific control properties. This guide shows how to tune regularization weights and add custom objectives.

using QuantumCollocation
using PiccoloQuantumObjects
using NamedTrajectories
using DirectTrajOpt

# ## Basic Setup

H_drift = 0.1 * PAULIS.Z
H_drives = [PAULIS.X, PAULIS.Y]
sys = QuantumSystem(H_drift, H_drives, [1.0, 1.0])
U_goal = GATES.H
N = 51

# ## Understanding Default Objectives

# A standard `UnitarySmoothPulseProblem` minimizes:
# ```math
# J = Q \cdot \text{infidelity} + \frac{1}{2}\sum_k (R_u u_k^2 + R_{du} \dot{u}_k^2 + R_{ddu} \ddot{u}_k^2)
# ```

# Let's see the defaults:

prob_default = UnitarySmoothPulseProblem(sys, U_goal, N)

# Default weights (from problem template):
# - Q = 100.0 (infidelity weight)
# - R_u = 0.01 (control amplitude)
# - R_du = 0.01 (first derivative)  
# - R_ddu = 0.01 (second derivative)

# ## 1. Tuning Regularization Weights

# ### Infidelity Weight (Q)

# **Higher Q** → prioritize fidelity over smoothness
# **Lower Q** → prioritize smooth pulses over fidelity

prob_high_fidelity = UnitarySmoothPulseProblem(sys, U_goal, N;
    Q=1000.0,    # 10× default - demand high fidelity
    R_u=0.01,
    R_du=0.01,
    R_ddu=0.01
)

solve!(prob_high_fidelity; max_iter=100)
println("High Q fidelity: ", unitary_rollout_fidelity(prob_high_fidelity.trajectory, sys))

prob_smooth_priority = UnitarySmoothPulseProblem(sys, U_goal, N;
    Q=10.0,      # Lower Q - accept lower fidelity for smoother pulses
    R_u=0.1,
    R_du=1.0,
    R_ddu=10.0
)

solve!(prob_smooth_priority; max_iter=100)
println("Smooth priority fidelity: ", unitary_rollout_fidelity(prob_smooth_priority.trajectory, sys))

# ### Control Amplitude (R_u)

# Penalizes large control values: minimizes average power

# Low amplitude preference:
prob_low_power = UnitarySmoothPulseProblem(sys, U_goal, N;
    R_u=1.0      # 100× default - strong amplitude penalty
)

solve!(prob_low_power; max_iter=100)

# Check power:
avg_power_default = mean(prob_default.trajectory.u.^2)
avg_power_low = mean(prob_low_power.trajectory.u.^2)
println("Average power - default: ", avg_power_default)
println("Average power - low: ", avg_power_low)

# ### First Derivative (R_du)

# Penalizes rapid changes: enforces smoothness

# Very smooth pulses:
prob_very_smooth = UnitarySmoothPulseProblem(sys, U_goal, N;
    R_du=10.0    # Strong smoothness penalty
)

solve!(prob_very_smooth; max_iter=100)

# ### Second Derivative (R_ddu)

# Penalizes acceleration: prevents sharp corners

# Ultra-smooth (C² continuous) pulses:
prob_ultra_smooth = UnitarySmoothPulseProblem(sys, U_goal, N;
    R_ddu=100.0  # Penalize curvature heavily
)

solve!(prob_ultra_smooth; max_iter=100)

# ## 2. Per-Drive Regularization

# Different drives may need different regularization:

# Example: Strong Y drive, gentle X drive
R_u_per_drive = [0.01, 0.1]    # Penalize Y more
R_du_per_drive = [0.01, 1.0]   # Y must be smoother
R_ddu_per_drive = [0.01, 10.0]

prob_asymmetric = UnitarySmoothPulseProblem(sys, U_goal, N;
    R_u=R_u_per_drive,
    R_du=R_du_per_drive,
    R_ddu=R_ddu_per_drive
)

solve!(prob_asymmetric; max_iter=100)

# ## 3. Adding Custom Objectives

# To add completely custom cost functions, work with the `Objective` type.

# ### Example: Minimize Peak Power

# Add a penalty for maximum control amplitude:

using ForwardDiff

# Define a custom objective function
function peak_power_objective(Z⃗, traj)
    cost = 0.0
    for k in 1:traj.T
        u_slice = slice(k, :u, traj.dims)
        u = Z⃗[u_slice]
        # Smooth approximation of max: log-sum-exp
        cost += log(sum(exp.(10 .* u.^2))) / 10
    end
    return cost
end

# Compute gradient (for optimization)
function peak_power_gradient(Z⃗, traj)
    return ForwardDiff.gradient(z -> peak_power_objective(z, traj), Z⃗)
end

# Compute Hessian structure (sparsity pattern)
function peak_power_hessian_structure(traj)
    # For simple objectives, dense structure is fine
    n = traj.dim * traj.T
    return [(i, j) for i in 1:n for j in 1:i]
end

# Compute Hessian values
function peak_power_hessian(Z⃗, traj)
    H = ForwardDiff.hessian(z -> peak_power_objective(z, traj), Z⃗)
    # Extract lower triangle
    vals = Float64[]
    for i in 1:size(H, 1)
        for j in 1:i
            push!(vals, H[i, j])
        end
    end
    return vals
end

# Create Objective
peak_power_obj = Objective(
    peak_power_objective,
    peak_power_gradient,
    peak_power_hessian,
    peak_power_hessian_structure
)

# Combine with standard objective
prob_with_peak = UnitarySmoothPulseProblem(sys, U_goal, N)
total_objective = prob_with_peak.objective + 0.1 * peak_power_obj

# Create new problem with combined objective
prob_custom = DirectTrajOptProblem(
    prob_with_peak.trajectory,
    total_objective,
    prob_with_peak.dynamics;
    constraints=prob_with_peak.constraints
)

solve!(prob_custom; max_iter=100)

# ## 4. Time-Dependent Weights

# Example: Penalize control more at the end (for clean turn-off)

function time_weighted_control_objective(Z⃗, traj)
    cost = 0.0
    for k in 1:traj.T
        # Weight increases linearly with time
        weight = k / traj.T
        
        u_slice = slice(k, :u, traj.dims)
        u = Z⃗[u_slice]
        
        cost += weight * sum(u.^2)
    end
    return cost
end

# (Similar gradient/hessian computation as above)

# ## 5. Energy Budget Objective

# Instead of just penalizing amplitude, minimize total energy:

function energy_objective(Z⃗, traj)
    energy = 0.0
    for k in 1:traj.T
        u_slice = slice(k, :u, traj.dims)
        Δt_slice = slice(k, :Δt, traj.dims)
        
        u = Z⃗[u_slice]
        Δt = Z⃗[Δt_slice][1]
        
        energy += sum(u.^2) * Δt
    end
    return energy
end

# ## 6. Spectral Regularization

# Penalize high-frequency components (alternative to derivative penalties):

using FFTW

function spectral_regularization(Z⃗, traj)
    cost = 0.0
    for drive_idx in 1:traj.dims[:u]
        # Extract time series for this drive
        u_timeseries = [Z⃗[slice(k, :u, traj.dims)[drive_idx]] for k in 1:traj.T]
        
        # FFT
        spectrum = fft(u_timeseries)
        
        # Penalize high frequencies (e.g., > 50% Nyquist)
        cutoff = div(length(spectrum), 2)
        high_freq = spectrum[cutoff:end]
        
        cost += sum(abs2.(high_freq))
    end
    return cost
end

# ## 7. Robustness Objectives

# Penalize sensitivity to parameters:

# Example: Minimize gradient w.r.t. system parameters
function robustness_objective(Z⃗, traj, sys)
    # Compute fidelity gradient w.r.t. drift Hamiltonian
    # (Requires variational quantum system)
    # This is conceptual - see UnitaryVariationalProblem for implementation
    return 0.0  # Placeholder
end

# ## 8. Physical Realism Objectives

# Example: Match target pulse shape (e.g., Gaussian envelope)

function gaussian_envelope_objective(Z⃗, traj)
    cost = 0.0
    for k in 1:traj.T
        # Desired Gaussian envelope
        t_norm = (k - traj.T/2) / (traj.T/4)
        target_amplitude = exp(-t_norm^2)
        
        u_slice = slice(k, :u, traj.dims)
        u = Z⃗[u_slice]
        
        # Penalize deviation from Gaussian envelope
        cost += (norm(u) - target_amplitude)^2
    end
    return cost
end

# ## 9. Combining Multiple Objectives

# Objectives are additive - you can combine many:

prob = UnitarySmoothPulseProblem(sys, U_goal, N;
    Q=100.0,      # High fidelity
    R_u=0.01,     # Moderate amplitude penalty
    R_du=0.1,     # Strong smoothness
    R_ddu=1.0     # Very strong curvature penalty
)

# Then add custom objectives:
# total_obj = prob.objective + 0.1 * energy_obj + 0.01 * spectral_obj + ...

# ## Common Regularization Recipes

# **Smooth, implementable pulses** (most common):
# ```julia
# R_u=0.01, R_du=0.1, R_ddu=1.0
# ```

# **Low power, accept longer time**:
# ```julia
# R_u=1.0, R_du=0.01, R_ddu=0.01
# ```

# **Ultra-smooth for noisy hardware**:
# ```julia
# R_u=0.01, R_du=1.0, R_ddu=10.0
# ```

# **High fidelity at any cost**:
# ```julia
# Q=1000.0, R_u=1e-6, R_du=1e-6, R_ddu=1e-6
# ```

# **Aggressive pulses for fast gates**:
# ```julia
# Q=100.0, R_u=1e-4, R_du=1e-4, R_ddu=1e-3
# ```

# ## Tuning Strategy

# 1. **Start with defaults** - they work well for most cases
# 2. **Adjust one weight at a time** - see the effect
# 3. **Check convergence** - some combinations are hard to optimize
# 4. **Validate physically** - can your hardware actually do this?
# 5. **Iterate** - optimal weights are problem-specific

# ## Key Takeaways

# - `Q` controls fidelity vs smoothness tradeoff
# - `R_u` penalizes control amplitude (power)
# - `R_du` enforces smoothness (rate limits)
# - `R_ddu` prevents sharp corners (acceleration limits)
# - Per-drive weights allow asymmetric regularization
# - Custom objectives enable arbitrary cost functions
# - Start with defaults, tune empirically
# - Higher regularization → smoother but slower convergence
#
# ## Next Steps
#
# - [Adding Constraints](@ref) - Enforce hard limits
# - [Initial Trajectories](@ref) - Better starting points
# - [Problem Templates Overview](@ref) - Choose the right template
