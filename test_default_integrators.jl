using QuantumCollocation
using PiccoloQuantumObjects

println("=" ^ 70)
println("Testing Default Integrators")
println("=" ^ 70)
println()

# Create a simple quantum system
sys = QuantumSystem(
    GATES[:Z],              # H_drift
    [GATES[:X], GATES[:Y]], # H_drives
    1.0,                    # T_max
    [1.0, 1.0]             # drive_bounds
)

# ----------------------------------------------------------------------------- #
# Example 1: Using default integrator
# ----------------------------------------------------------------------------- #

println("Example 1: Default integrator (simplest approach)")
println("-" ^ 70)

qtraj = unitary_trajectory(sys, GATES[:H], 10)
println("✓ Created UnitaryTrajectory")

# Get default integrator - no need to specify anything!
integrator = default_integrator(qtraj)
println("✓ Created default integrator")
println("  Type: ", typeof(integrator))
println()

# ----------------------------------------------------------------------------- #
# Example 2: Creating custom integrators when needed
# ----------------------------------------------------------------------------- #

println("Example 2: Custom integrator (when user wants control)")
println("-" ^ 70)

qtraj2 = unitary_trajectory(sys, GATES[:X], 10)
println("✓ Created UnitaryTrajectory")

# User can create custom integrator using trajectory info
custom_integrator = UnitaryIntegrator(
    system(qtraj2),
    trajectory(qtraj2),
    state_name(qtraj2),
    control_name(qtraj2)
)
println("✓ Created custom UnitaryIntegrator")
println("  Type: ", typeof(custom_integrator))
println()

# ----------------------------------------------------------------------------- #
# Example 3: Adding derivative constraints
# ----------------------------------------------------------------------------- #

println("Example 3: Combining with derivative integrators")
println("-" ^ 70)

qtraj3 = unitary_trajectory(sys, GATES[:Y], 10)
println("✓ Created UnitaryTrajectory")

# Get default dynamics integrator
dynamics = default_integrator(qtraj3)
println("✓ Created dynamics integrator")

# Add derivative constraints directly using underlying trajectory
using DirectTrajOpt
du_integrator = DerivativeIntegrator(trajectory(qtraj3), :u, :du)
ddu_integrator = DerivativeIntegrator(trajectory(qtraj3), :du, :ddu)
println("✓ Created derivative integrators")

# Combine all integrators
all_integrators = [dynamics, du_integrator, ddu_integrator]
println("✓ Combined ", length(all_integrators), " integrators")
println()

# ----------------------------------------------------------------------------- #
# Example 4: Works for all trajectory types
# ----------------------------------------------------------------------------- #

println("Example 4: Default integrators for all trajectory types")
println("-" ^ 70)

# Ket trajectory
ψ_init = ComplexF64[1.0, 0.0]
ψ_goal = ComplexF64[0.0, 1.0]
ket_traj = ket_trajectory(sys, ψ_init, ψ_goal, 10)
ket_int = default_integrator(ket_traj)
println("✓ KetTrajectory → ", typeof(ket_int))

# Density trajectory
open_sys = OpenQuantumSystem(GATES[:Z], [GATES[:X], GATES[:Y]], 1.0, [1.0, 1.0])
ρ_init = ComplexF64[1.0 0.0; 0.0 0.0]
ρ_goal = ComplexF64[0.0 0.0; 0.0 1.0]
density_traj = density_trajectory(open_sys, ρ_init, ρ_goal, 10)
density_int = default_integrator(density_traj)
println("✓ DensityTrajectory → ", typeof(density_int))
println()

println("=" ^ 70)
println("All examples completed successfully! ✓")
println("=" ^ 70)
println()
println("Key Takeaway:")
println("  • Use default_integrator(qtraj) for simple cases")
println("  • Create custom integrators when you need control")
println("  • Access trajectory info via system(), trajectory(), state_name(), etc.")
