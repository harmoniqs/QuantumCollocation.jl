# # Minimum Time Optimization

# In this example, we'll show how to optimize for the **fastest possible gate** while maintaining high fidelity. This is useful when you want to minimize gate duration to reduce decoherence effects.

# ## Problem Setup

# We'll start with a smooth pulse solution for an X gate, then optimize for minimum time.

using QuantumCollocation
using PiccoloQuantumObjects
using NamedTrajectories

using PiccoloPlots
using CairoMakie

# Define a simple qubit system:

H_drift = 0.05 * PAULIS.Z
H_drives = [PAULIS.X, PAULIS.Y]
T_max = 50.0  # Start with generous time budget
drive_bounds = [(-0.5, 0.5), (-0.5, 0.5)]

sys = QuantumSystem(H_drift, H_drives, T_max, drive_bounds)

# Target gate: X gate (π rotation about X axis)

U_goal = GATES.X
N = 101  # Use more timesteps for better time resolution

# ## Step 1: Find a Smooth Pulse Solution

# First, we solve for a smooth pulse with a fixed time to get a good starting point:

prob_smooth = UnitarySmoothPulseProblem(sys, U_goal, N)
solve!(prob_smooth; max_iter=100)

fid_smooth = unitary_rollout_fidelity(prob_smooth.trajectory, sys)
duration_smooth = get_duration(prob_smooth.trajectory)

println("Smooth pulse fidelity: ", fid_smooth)
println("Smooth pulse duration: ", duration_smooth)

# Let's visualize the smooth pulse solution:

plot_unitary_populations(prob_smooth.trajectory)

# ## Step 2: Optimize for Minimum Time

# Now we'll use the smooth pulse as a starting point and optimize for minimum time while constraining the fidelity:

prob_mintime = UnitaryMinimumTimeProblem(
    prob_smooth,  # Use smooth pulse as initial guess
    U_goal;
    final_fidelity=0.9999  # Require high fidelity
)

# Solve the minimum time problem:

solve!(prob_mintime; max_iter=300)

# ## Results

# Let's compare the results:

fid_mintime = unitary_rollout_fidelity(prob_mintime.trajectory, sys)
duration_mintime = get_duration(prob_mintime.trajectory)

println("\n=== Comparison ===")
println("Smooth pulse:")
println("  Duration: ", duration_smooth)
println("  Fidelity: ", fid_smooth)
println("\nMinimum time:")
println("  Duration: ", duration_mintime)
println("  Fidelity: ", fid_mintime)
println("\nSpeedup: ", duration_smooth / duration_mintime, "×")

@assert fid_mintime > 0.9999
@assert duration_mintime < duration_smooth

# Visualize the minimum time solution:

plot_unitary_populations(prob_mintime.trajectory)

# ## Understanding the Tradeoff

# Let's look at how the control pulses changed:

fig = Figure(size=(1000, 600))

## Smooth pulse controls
ax1 = Axis(fig[1, 1], xlabel="Time", ylabel="Control amplitude", title="Smooth Pulse")
for i in 1:size(prob_smooth.trajectory.u, 1)
    lines!(ax1, prob_smooth.trajectory.u[i, :], label="u$i")
end
axislegend(ax1)

## Minimum time controls  
ax2 = Axis(fig[1, 2], xlabel="Time", ylabel="Control amplitude", title="Minimum Time")
for i in 1:size(prob_mintime.trajectory.u, 1)
    lines!(ax2, prob_mintime.trajectory.u[i, :], label="u$i")
end
axislegend(ax2)

fig

# Notice how the minimum time solution:
# - Uses fewer timesteps (compressed duration)
# - Has more aggressive control amplitudes
# - May have sharper transitions

# ## Varying the Fidelity Constraint

# Let's see how gate duration depends on the required fidelity:

fidelities = [0.99, 0.995, 0.999, 0.9999]
durations = Float64[]

for target_fid in fidelities
    prob = UnitaryMinimumTimeProblem(
        prob_smooth,
        U_goal;
        final_fidelity=target_fid
    )
    solve!(prob; max_iter=200, verbose=false)
    push!(durations, get_duration(prob.trajectory))
    println("Fidelity ", target_fid, " → Duration ", durations[end])
end

# Plot the tradeoff:

fig_tradeoff = Figure()
ax = Axis(
    fig_tradeoff[1, 1],
    xlabel="Required Fidelity",
    ylabel="Gate Duration",
    title="Fidelity vs Duration Tradeoff"
)
scatter!(ax, fidelities, durations, markersize=15)
lines!(ax, fidelities, durations)
fig_tradeoff

# ## Key Takeaways

# 1. **Start with smooth pulse** - Good initialization is crucial
# 2. **Higher fidelity = longer duration** - There's always a tradeoff
# 3. **More timesteps = better resolution** - Use enough N for time optimization
# 4. **Control bounds matter** - Tighter bounds → longer minimum time
#
# ## When to Use Minimum Time Problems
#
# Use `UnitaryMinimumTimeProblem` when:
# - Decoherence is limiting your gate fidelity
# - You want to maximize throughput in quantum circuits
# - You need to compare against theoretical speed limits
# - You're exploring the fundamental limits of your control system
#
# For most applications, `UnitarySmoothPulseProblem` with reasonable duration is sufficient and more robust.
