# # Working with Solutions

# This guide covers how to solve problems, extract data from solutions, evaluate performance, and save/load results.

using QuantumCollocation
using PiccoloQuantumObjects
using NamedTrajectories

# ## Solving Problems

# Once you've created a problem template, solving is straightforward with the `solve!` function:

system = QuantumSystem(0.1 * PAULIS.Z, [PAULIS.X, PAULIS.Y], 10.0, [1.0, 1.0])
U_goal = EmbeddedOperator(GATES.H, system)
N = 51

prob = UnitarySmoothPulseProblem(system, U_goal, N)

# The `solve!` function accepts several key options:

solve!(prob;
    max_iter = 100,        # Maximum optimization iterations
    verbose = true,         # Print convergence information
    print_level = 1         # Ipopt output level (0-5)
)

# ### Understanding Convergence

# Ipopt reports several key metrics:
# - **Objective**: Current cost function value
# - **inf_pr**: Constraint violation (primal infidelity) - should go to ~0
# - **inf_du**: Dual infidelity - measure of optimality
# - **lg(mu)**: Log of barrier parameter
# - **alpha_du/alpha_pr**: Step sizes

# Successful convergence typically shows `inf_pr` < 1e-6 and status `Optimal Solution Found`.

# ## Extracting Data from Solutions

# ### Accessing Controls

# Control pulses are stored in the trajectory with automatic naming:

u = prob.trajectory.u       # Control amplitudes [n_drives × T]
du = prob.trajectory.du     # First derivatives
ddu = prob.trajectory.ddu   # Second derivatives

println("Control shape: ", size(u))
println("Number of drives: ", size(u, 1))
println("Number of timesteps: ", size(u, 2))

# Access individual drive controls:
u_drive_1 = u[1, :]   # First drive over time

# ### Accessing States

# For unitary problems:
Ũ⃗ = prob.trajectory.Ũ⃗  # Vectorized unitary [N² × T]

# The unitary is stored in "isovec" format (vectorized). To get the actual unitary matrix at timestep k:
using LinearAlgebra
k = N  # Final timestep
U_k = iso_vec_to_operator(Ũ⃗[:, k])
println("Final unitary dimensions: ", size(U_k))

# For ket (state) problems, use:
# ψ̃ = prob.trajectory.ψ̃  # Vectorized state [2N × T]

# ### Time Grid

# Access timestep information:
Δt_vec = prob.trajectory.Δt  # Timestep durations

# Calculate total duration:
duration = get_duration(prob.trajectory)
println("Total gate time: ", duration, " (arbitrary units)")

# For minimum time problems, timesteps vary:
# min_prob = UnitaryMinimumTimeProblem(prob, U_goal)
# solve!(min_prob, max_iter=100)
# Δt_optimized = min_prob.trajectory.Δt  # Variable timesteps

# ## Evaluating Solutions

# ### Fidelity Calculations

# **Direct fidelity** - Compare final state to goal:
U_final = iso_vec_to_operator(prob.trajectory.Ũ⃗[:, end])
fid_direct = unitary_fidelity(U_final, U_goal)
println("Direct fidelity: ", fid_direct)

# **Rollout fidelity** - Simulate dynamics forward:
fid_rollout = unitary_rollout_fidelity(prob.trajectory, system)
println("Rollout fidelity: ", fid_rollout)

# The rollout fidelity is more accurate as it accounts for actual dynamics, while direct fidelity only checks the final point.

# ### For Embedded Operators (Multilevel Systems)

# When working with subspaces (e.g., qubit in transmon):
# op = EmbeddedOperator(:X, system)
# prob_embedded = UnitarySmoothPulseProblem(system, op, T, Δt)
# solve!(prob_embedded, max_iter=100)
# 
# # Evaluate fidelity only in computational subspace
# fid_subspace = unitary_rollout_fidelity(
#     prob_embedded.trajectory, 
#     system; 
#     subspace = op.subspace
# )

# ### Leakage Evaluation

# For multilevel systems, check population in leakage levels:
# This requires analyzing state populations during evolution
# See the Multilevel Transmon example for details

# ### Constraint Violations

# Check if solution satisfies all constraints:
# - Dynamics constraints: Compare rollout vs direct fidelity
# - Bound constraints: Verify |u| ≤ u_bound
# - Derivative constraints: Check |du|, |ddu| within bounds

println("Max control amplitude: ", maximum(abs.(u)))
println("Max control derivative: ", maximum(abs.(du)))

# ## Saving and Loading

# ### Save a Trajectory

# using JLD2
# save_object("my_solution.jld2", prob.trajectory)

# ### Load and Reuse

# Load trajectory for warm-starting:
# traj_loaded = load_object("my_solution.jld2")
# 
# # Use as initial guess for new problem
# prob_refined = UnitarySmoothPulseProblem(
#     system, U_goal, T, Δt;
#     u_guess = traj_loaded.u,
#     piccolo_options = PiccoloOptions(verbose=false)
# )

# ### Save Entire Problem

# To save all problem information including constraints and objectives:
# save_object("my_problem.jld2", prob)

# ## Post-Processing

# ### Resampling Trajectories

# Change time discretization while preserving control shape:
# T_new = 101  # More timesteps
# traj_resampled = resample(prob.trajectory, T_new)

# ### Extracting for Experiments

# Prepare control pulses for hardware:

using PiccoloPlots  # For visualization
using CairoMakie

# Plot controls
fig = plot_controls(prob.trajectory)
# save("controls.png", fig)

# Extract control data for export
control_data = Dict(
    "time" => cumsum([0; prob.trajectory.Δt[:]]),
    "drive_1_real" => u[1, :],
    "drive_2_real" => u[2, :],
    "duration" => duration
)

# # Save to CSV or other format for AWG
# using CSV, DataFrames
# df = DataFrame(control_data)
# CSV.write("pulse_sequence.csv", df)

# ### Pulse Filtering

# Apply smoothing to reduce high-frequency noise:
# using DSP
# for i in 1:size(u, 1)
#     u_filtered[i, :] = filtfilt(responsetype, u[i, :])
# end

# ## Best Practices

# **Starting a new optimization:**
# 1. Begin with coarse discretization (small T)
# 2. Use relaxed bounds and convergence criteria
# 3. Refine solution incrementally
# 4. Use previous solution as warm start

# **Debugging poor convergence:**
# 1. Check `inf_pr` - high values indicate constraint violations
# 2. Verify system Hamiltonian is correct
# 3. Try looser bounds or larger `u_bound`
# 4. Increase regularization weights (`R_u`, `R_du`, `R_ddu`)
# 5. Use `piccolo_options.bound_state=true` for better numerics

# **Improving solutions:**
# 1. Increase T (more timesteps = finer control)
# 2. Add derivative constraints for smoother pulses
# 3. Use minimum time optimization for fastest gates
# 4. Apply leakage constraints for multilevel systems
# 5. Use sampling problems for robust control

println("Solution evaluation complete!")
