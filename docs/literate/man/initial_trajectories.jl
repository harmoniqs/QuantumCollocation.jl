# # Initial Trajectory Strategies

# Good initialization is crucial for nonlinear optimization. A well-chosen initial trajectory can:
# - Reduce solve time by 10-100×
# - Avoid local minima
# - Improve convergence reliability
# - Enable harder problems to solve

# This guide shows various initialization strategies for quantum optimal control problems.

using QuantumCollocation
using PiccoloQuantumObjects
using NamedTrajectories

# ## Basic Setup

H_drift = 0.1 * PAULIS.Z
H_drives = [PAULIS.X, PAULIS.Y]
sys = QuantumSystem(H_drift, H_drives, 10.0, [1.0, 1.0])
U_goal = GATES.H
N = 51

# ## 1. Default Initialization (Random + Rollout)

# Problem templates use default initialization:

prob_default = UnitarySmoothPulseProblem(sys, U_goal, N)

# Check initial fidelity:
fid_init_default = unitary_rollout_fidelity(prob_default.trajectory, sys)
println("Default initialization fidelity: ", fid_init_default)

# Default process:
# 1. Random controls (small amplitude)
# 2. Forward rollout to get states
# 3. Satisfy derivative relationships

# ## 2. Warm Start from Previous Solution

# **Best strategy:** Use solution from similar problem

# Solve a simpler problem first:
prob_easy = UnitarySmoothPulseProblem(sys, U_goal, N;
    Q=10.0,      # Lower fidelity requirement
    R_u=0.001    # Less regularization
)
solve!(prob_easy; max_iter=50)

# Use as initialization for harder problem:
prob_hard = UnitarySmoothPulseProblem(sys, U_goal, N;
    Q=1000.0,    # Higher fidelity
    R_u=0.1,     # More regularization
    init_trajectory=prob_easy.trajectory
)
solve!(prob_hard; max_iter=100)

println("Warm start fidelity: ", unitary_rollout_fidelity(prob_hard.trajectory, sys))

# ## 3. Custom Control Guess (u_guess)

# Provide initial control pulses based on physical intuition:

# Example: Smooth ramp-up/ramp-down pulse

function create_smooth_pulse(N, n_drives, max_amplitude=0.5)
    u = zeros(n_drives, N)
    
    for k in 1:N
        # Smooth envelope: sin² ramp
        t_norm = (k - 1) / (N - 1)
        envelope = sin(π * t_norm)^2
        
        # Drive 1: cosine modulation
        u[1, k] = max_amplitude * envelope * cos(2π * t_norm)
        
        # Drive 2: sine modulation (π/2 out of phase)
        u[2, k] = max_amplitude * envelope * sin(2π * t_norm)
    end
    
    return u
end

u_guess = create_smooth_pulse(N, sys.n_drives)

prob_with_guess = UnitarySmoothPulseProblem(sys, U_goal, N;
    u_guess=u_guess
)

fid_init_guess = unitary_rollout_fidelity(prob_with_guess.trajectory, sys)
println("Custom guess fidelity: ", fid_init_guess)

solve!(prob_with_guess; max_iter=100)

# ## 4. Geodesic Initialization

# For unitary gates, initialize state trajectory along geodesic path:

prob_geodesic = UnitarySmoothPulseProblem(sys, U_goal, N;
    piccolo_options=PiccoloOptions(geodesic=true)
)

fid_init_geodesic = unitary_rollout_fidelity(prob_geodesic.trajectory, sys)
println("Geodesic initialization fidelity: ", fid_init_geodesic)

solve!(prob_geodesic; max_iter=100)

# Geodesic initialization:
# - Interpolates between identity and target unitary
# - Follows shortest path in SU(N)
# - Often gives better starting fidelity
# - Particularly useful for large gates

# ## 5. GRAPE-style Initialization

# Use constant pulses at maximum amplitude (aggressive):

u_grape = ones(sys.n_drives, N) .* 0.8  # 80% of maximum
u_grape[1, :] .*= cos.(range(0, 2π, length=N))  # Modulate drive 1
u_grape[2, :] .*= sin.(range(0, 2π, length=N))  # Modulate drive 2

prob_grape = UnitarySmoothPulseProblem(sys, U_goal, N;
    u_guess=u_grape
)

solve!(prob_grape; max_iter=100)

# GRAPE-style works well when:
# - Strong controls available
# - Short time horizons
# - Seeking maximum speed

# ## 6. Time Scaling / Trajectory Resampling

# Solve at coarse resolution, then refine:

# Coarse problem (fewer timesteps)
N_coarse = 25
prob_coarse = UnitarySmoothPulseProblem(sys, U_goal, N_coarse)
solve!(prob_coarse; max_iter=100)

# Resample to finer resolution
function resample_trajectory(traj_coarse, N_fine)
    # Linear interpolation of controls
    N_coarse = traj_coarse.T
    u_coarse = traj_coarse.u
    
    u_fine = zeros(size(u_coarse, 1), N_fine)
    for drive_idx in 1:size(u_coarse, 1)
        # Simple linear interpolation
        t_coarse = range(0, 1, length=N_coarse)
        t_fine = range(0, 1, length=N_fine)
        
        for (i, t) in enumerate(t_fine)
            # Find surrounding coarse points
            k = min(N_coarse - 1, max(1, floor(Int, t * (N_coarse - 1)) + 1))
            α = (t * (N_coarse - 1)) - (k - 1)
            
            u_fine[drive_idx, i] = (1 - α) * u_coarse[drive_idx, k] + 
                                    α * u_coarse[drive_idx, min(k + 1, N_coarse)]
        end
    end
    
    return u_fine
end

u_fine = resample_trajectory(prob_coarse.trajectory, N)

# Solve fine problem with resampled initialization
prob_fine = UnitarySmoothPulseProblem(sys, U_goal, N;
    u_guess=u_fine
)
solve!(prob_fine; max_iter=100)

println("Coarse-to-fine final fidelity: ", unitary_rollout_fidelity(prob_fine.trajectory, sys))

# ## 7. Adiabatic / Homotopy Continuation

# Gradually increase difficulty:

# Strategy: Solve sequence with increasing Q (fidelity weight)
Q_sequence = [1.0, 10.0, 100.0, 1000.0]
prob_current = nothing

for Q in Q_sequence
    if isnothing(prob_current)
        prob_current = UnitarySmoothPulseProblem(sys, U_goal, N; Q=Q)
    else
        prob_current = UnitarySmoothPulseProblem(sys, U_goal, N;
            Q=Q,
            init_trajectory=prob_current.trajectory
        )
    end
    
    solve!(prob_current; max_iter=50)
    fid = unitary_rollout_fidelity(prob_current.trajectory, sys)
    println("Q=$Q → fidelity: $fid")
end

# ## 8. Random Restart Strategy

# Try multiple random initializations, keep best:

function solve_with_random_restarts(sys, U_goal, N, n_restarts=5)
    best_fidelity = 0.0
    best_trajectory = nothing
    
    for restart in 1:n_restarts
        prob = UnitarySmoothPulseProblem(sys, U_goal, N)
        solve!(prob; max_iter=50, verbose=false)
        
        fid = unitary_rollout_fidelity(prob.trajectory, sys)
        println("Restart $restart: fidelity = $fid")
        
        if fid > best_fidelity
            best_fidelity = fid
            best_trajectory = prob.trajectory
        end
    end
    
    return best_trajectory
end

# Run random restarts
best_traj = solve_with_random_restarts(sys, U_goal, N, 3)

# Refine best solution
prob_refined = UnitarySmoothPulseProblem(sys, U_goal, N;
    init_trajectory=best_traj
)
solve!(prob_refined; max_iter=100)

# ## 9. Physical Insights: Bang-Bang to Smooth

# Start with bang-bang (on/off) controls, smooth them out:

function create_bangbang_pulse(N, n_drives, switch_times)
    u = zeros(n_drives, N)
    
    for (drive_idx, switches) in enumerate(switch_times)
        state = 1.0
        switch_idx = 1
        
        for k in 1:N
            if switch_idx <= length(switches) && k >= switches[switch_idx]
                state *= -1
                switch_idx += 1
            end
            u[drive_idx, k] = state * 0.8
        end
    end
    
    return u
end

# Define switching pattern
switch_times = [[10, 25, 40], [5, 30, 45]]
u_bangbang = create_bangbang_pulse(N, sys.n_drives, switch_times)

prob_bangbang = UnitarySmoothPulseProblem(sys, U_goal, N;
    u_guess=u_bangbang,
    R_du=10.0,    # Strong derivative penalty will smooth it
    R_ddu=100.0
)

solve!(prob_bangbang; max_iter=200)

# ## 10. Transfer Learning from Similar Gates

# Use solution from different but related gate:

# Solve for X gate
prob_X = UnitarySmoothPulseProblem(sys, GATES.X, N)
solve!(prob_X; max_iter=100)

# Use as initialization for Y gate (related by π/2 rotation)
prob_Y = UnitarySmoothPulseProblem(sys, GATES.Y, N;
    init_trajectory=prob_X.trajectory
)
solve!(prob_Y; max_iter=100)

println("Transfer learning Y-gate fidelity: ", unitary_rollout_fidelity(prob_Y.trajectory, sys))

# ## Initialization Strategy Decision Tree

# **Easy problem** (single qubit, strong controls):
# → Default initialization works fine

# **Medium problem** (two qubits, moderate controls):
# → Geodesic initialization OR coarse-to-fine

# **Hard problem** (many levels, weak controls, tight constraints):
# → Homotopy continuation (gradually increase difficulty)
# → OR warm start from related solved problem
# → OR coarse-to-fine with multiple refinements

# **Unknown difficulty**:
# → Try default first
# → If convergence slow: switch to geodesic
# → If still slow: use coarse-to-fine
# → If repeatedly failing: random restarts + homotopy

# ## Best Practices

# 1. **Always check initial fidelity** - should be > 0.01 for most problems
# 2. **Warm start when possible** - reuse previous solutions
# 3. **Scale up gradually** - coarse to fine, easy to hard
# 4. **Save successful trajectories** - build a library
# 5. **Use physical intuition** - u_guess can encode domain knowledge
# 6. **Be patient with hard problems** - good initialization takes time
# 7. **Monitor convergence** - poor init shows in first few iterations

# ## Common Issues and Fixes

# **Issue**: Optimizer gives up immediately (max_iter reached, low fidelity)
# **Fix**: Better initialization - try geodesic or warm start

# **Issue**: Solution oscillates, doesn't converge
# **Fix**: Increase regularization (R_du, R_ddu) OR use smoother u_guess

# **Issue**: Infeasible problem from start
# **Fix**: Relax constraints temporarily, find feasible point, then tighten

# **Issue**: Good fidelity but violates constraints
# **Fix**: Start from feasible trajectory respecting all constraints

# ## Key Takeaways

# - Default initialization works for ~80% of problems
# - Warm starting from related solutions is the best strategy
# - Geodesic initialization helps for large unitary gates
# - Coarse-to-fine scaling is robust for hard problems
# - Physical intuition (u_guess) can dramatically help
# - Random restarts catch lucky initializations
# - Homotopy continuation solves hard problems reliably
# - Save and reuse successful trajectories
#
# ## Next Steps
#
# - [Custom Objectives](@ref) - Tune cost function
# - [Adding Constraints](@ref) - Enforce requirements
# - [Problem Templates Overview](@ref) - Choose right template
