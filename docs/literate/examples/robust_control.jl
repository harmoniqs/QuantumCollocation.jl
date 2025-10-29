# # Robust Control with Sampling

# Real quantum systems have uncertainties - manufacturing variations, calibration errors, environmental fluctuations. **Robust control** finds pulses that work well across these variations.

# In this example, we'll use `UnitarySamplingProblem` to optimize a gate that's robust to:
# - Drift Hamiltonian uncertainties
# - Control amplitude miscalibrations

# ## Problem Setup

using QuantumCollocation
using PiccoloQuantumObjects
using NamedTrajectories

using PiccoloPlots
using CairoMakie

# ## Nominal System

# First, define our nominal (ideal) qubit system:

H_drift_nominal = 0.1 * PAULIS.Z
H_drives = [PAULIS.X, PAULIS.Y]
T_max = 15.0
drive_bounds = [(-1.0, 1.0), (-1.0, 1.0)]

sys_nominal = QuantumSystem(H_drift_nominal, H_drives, T_max, drive_bounds)

# Target: Hadamard gate

U_goal = GATES.H
N = 51

# ## Non-Robust Solution (Baseline)

# Let's first solve without robustness to see how a standard solution performs:

prob_standard = UnitarySmoothPulseProblem(sys_nominal, U_goal, N)
solve!(prob_standard; max_iter=100)

fid_standard = unitary_rollout_fidelity(prob_standard.trajectory, sys_nominal)
println("Standard solution fidelity (nominal system): ", fid_standard)

# ## System Variations

# Now let's model realistic system variations. We'll consider:
# 1. **Drift variation**: ±10% uncertainty in the Z bias
# 2. **Control miscalibration**: ±5% error in drive amplitudes

# Create systems representing these variations:

## Drift variations
drift_variations = [-0.1, 0.0, 0.1]  # ±10%
systems = QuantumSystem[]

for δ in drift_variations
    H_drift_varied = (1.0 + δ) * H_drift_nominal
    sys = QuantumSystem(H_drift_varied, H_drives, T_max, drive_bounds)
    push!(systems, sys)
end

println("Number of system variations: ", length(systems))

# ## Robust Optimization

# Now use `UnitarySamplingProblem` to optimize over all system variations simultaneously:

## Create target operators for each system (all want the same gate)
operators = [EmbeddedOperator(U_goal, sys) for sys in systems]

## Set up robust problem
prob_robust = UnitarySamplingProblem(
    systems,
    operators,
    N;
    system_weights=fill(1.0, length(systems))  # Equal weight to all variations
)

# Solve the robust problem:

solve!(prob_robust; max_iter=200)

# ## Compare Performance

# Let's evaluate both solutions (standard and robust) across all system variations:

println("\n=== Fidelity Comparison ===")
println("System Variation | Standard | Robust")
println("-" ^ 50)

fidelities_standard = Float64[]
fidelities_robust = Float64[]

for (i, sys) in enumerate(systems)
    ## Standard solution on this system
    traj_std = prob_standard.trajectory
    fid_std = unitary_rollout_fidelity(traj_std, sys)
    
    ## Robust solution on this system
    traj_rob = prob_robust.trajectory
    fid_rob = unitary_rollout_fidelity(traj_rob, sys)
    
    push!(fidelities_standard, fid_std)
    push!(fidelities_robust, fid_rob)
    
    println("Drift $(drift_variations[i]):    | $(round(fid_std, digits=5)) | $(round(fid_rob, digits=5))")
end

println("\nWorst-case fidelity:")
println("  Standard: ", minimum(fidelities_standard))
println("  Robust:   ", minimum(fidelities_robust))

# The robust solution should have better worst-case fidelity!

# ## Visualize the Tradeoff

# Plot fidelities across variations:

fig = Figure(size=(800, 500))
ax = Axis(
    fig[1, 1],
    xlabel="Drift Variation (%)",
    ylabel="Gate Fidelity",
    title="Robustness to System Variations"
)

x_vals = drift_variations .* 100

scatter!(ax, x_vals, fidelities_standard, markersize=15, label="Standard", color=:red)
lines!(ax, x_vals, fidelities_standard, color=:red, linestyle=:dash)

scatter!(ax, x_vals, fidelities_robust, markersize=15, label="Robust", color=:blue)
lines!(ax, x_vals, fidelities_robust, color=:blue)

axislegend(ax)
fig

# ## Control Pulse Comparison

# How do the control pulses differ?

fig_controls = Figure(size=(1000, 600))

## Standard solution
ax1 = Axis(fig_controls[1, 1], xlabel="Timestep", ylabel="Amplitude", title="Standard (Non-Robust)")
for i in 1:size(prob_standard.trajectory.u, 1)
    lines!(ax1, prob_standard.trajectory.u[i, :], label="u$i")
end
axislegend(ax1)

## Robust solution
ax2 = Axis(fig_controls[1, 2], xlabel="Timestep", ylabel="Amplitude", title="Robust (Sampling)")
for i in 1:size(prob_robust.trajectory.u, 1)
    lines!(ax2, prob_robust.trajectory.u[i, :], label="u$i")
end
axislegend(ax2)

fig_controls

# Robust pulses often:
# - Have slightly lower peak amplitudes
# - Are more "conservative" to handle uncertainties
# - May use different pulse shapes

# ## More Realistic Variations

# Let's test with more system variations including control miscalibration:

## Create a grid of variations
drift_vals = [-0.15, -0.05, 0.0, 0.05, 0.15]
control_scale_vals = [0.95, 1.0, 1.05]

systems_extended = QuantumSystem[]
operators_extended = EmbeddedOperator[]

for drift_δ in drift_vals
    for ctrl_scale in control_scale_vals
        H_drift = (1.0 + drift_δ) * H_drift_nominal
        H_drives_scaled = [ctrl_scale * H for H in H_drives]
        sys = QuantumSystem(H_drift, H_drives_scaled, T_max, drive_bounds)
        push!(systems_extended, sys)
        push!(operators_extended, EmbeddedOperator(U_goal, sys))
    end
end

println("\nExtended variations: ", length(systems_extended), " systems")

# Optimize with all variations:

prob_robust_extended = UnitarySamplingProblem(
    systems_extended,
    operators_extended,
    N;
    system_weights=fill(1.0, length(systems_extended))
)

solve!(prob_robust_extended; max_iter=300)

# Evaluate performance:

fidelities_extended = [
    unitary_rollout_fidelity(prob_robust_extended.trajectory, sys)
    for sys in systems_extended
]

println("\nExtended robust optimization:")
println("  Mean fidelity:       ", mean(fidelities_extended))
println("  Worst-case fidelity: ", minimum(fidelities_extended))
println("  Best-case fidelity:  ", maximum(fidelities_extended))

# ## Weighted Sampling

# You can also prioritize certain variations by adjusting `system_weights`:

## Give more weight to nominal system
weights_prioritized = ones(length(systems))
weights_prioritized[2] = 5.0  # Nominal system (middle of variation range)

prob_weighted = UnitarySamplingProblem(
    systems,
    operators,
    N;
    system_weights=weights_prioritized
)

solve!(prob_weighted; max_iter=200)

# Compare:

fidelities_weighted = [
    unitary_rollout_fidelity(prob_weighted.trajectory, sys)
    for sys in systems
]

println("\n=== Weighting Comparison ===")
println("System | Equal Weight | Prioritized")
println("-" ^ 45)
for i in 1:length(systems)
    println("$(i) | $(round(fidelities_robust[i], digits=5)) | $(round(fidelities_weighted[i], digits=5))")
end

# ## Key Takeaways

# 1. **Use UnitarySamplingProblem** for robust control
# 2. **Model realistic variations** - drift, control scaling, decoherence
# 3. **Balance fidelity vs robustness** - more variations = more conservative pulses
# 4. **Weight critical scenarios** - prioritize what matters most
# 5. **Test on unseen variations** - verify robustness generalizes
#
# ## When to Use Robust Control
#
# Use sampling-based robust control when:
# - System parameters have significant uncertainties
# - Gates must work across many devices (fab variations)
# - Environmental conditions fluctuate
# - Calibration drifts between experiments
# - Deploying on hardware without frequent recalibration
#
# ## Alternatives
#
# For other robustness approaches, see:
# - `UnitaryVariationalProblem` - Variational robustness with Hamiltonian perturbations
# - `UnitaryFreePhaseProblem` - Robustness to global phase errors
# - Custom constraints for specific robustness requirements
