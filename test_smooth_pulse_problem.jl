using QuantumCollocation
using PiccoloQuantumObjects
using Test

println("=" ^ 70)
println("Testing Unified SmoothPulseProblem")
println("=" ^ 70)
println()

# =============================================================================
# Test 1: UnitaryTrajectory → SmoothPulseProblem
# =============================================================================

println("Test 1: SmoothPulseProblem with UnitaryTrajectory")
println("-" ^ 70)

sys = QuantumSystem(
    GATES[:Z],
    [GATES[:X], GATES[:Y]],
    1.0,
    [1.0, 1.0]
)

# Step 1: Create quantum trajectory
qtraj = unitary_trajectory(sys, GATES[:H], 10)
println("✓ Created UnitaryTrajectory")

# Step 2: Create smooth pulse problem (automatically adds derivatives!)
prob = SmoothPulseProblem(qtraj; Q=100.0, R=1e-2)
println("✓ Created SmoothPulseProblem")

# Verify structure
@test prob isa DirectTrajOptProblem
println("✓ Problem is DirectTrajOptProblem")

@test length(prob.integrators) == 3  # dynamics + du + ddu
println("✓ Has 3 integrators (dynamics, du, ddu)")

@test haskey(prob.trajectory.components, :u)
@test haskey(prob.trajectory.components, :du)
@test haskey(prob.trajectory.components, :ddu)
println("✓ Trajectory has u, du, ddu components")
println()

# =============================================================================
# Test 2: KetTrajectory → SmoothPulseProblem
# =============================================================================

println("Test 2: SmoothPulseProblem with KetTrajectory")
println("-" ^ 70)

ψ_init = ComplexF64[1.0, 0.0]
ψ_goal = ComplexF64[0.0, 1.0]

qtraj_ket = ket_trajectory(sys, ψ_init, ψ_goal, 10)
println("✓ Created KetTrajectory")

prob_ket = SmoothPulseProblem(qtraj_ket; Q=50.0, R=1e-3)
println("✓ Created SmoothPulseProblem")

@test prob_ket isa DirectTrajOptProblem
@test length(prob_ket.integrators) == 3
println("✓ Structure correct")
println()

# =============================================================================
# Test 3: DensityTrajectory → SmoothPulseProblem
# =============================================================================

println("Test 3: SmoothPulseProblem with DensityTrajectory")
println("-" ^ 70)

open_sys = OpenQuantumSystem(
    GATES[:Z],
    [GATES[:X], GATES[:Y]],
    1.0,
    [1.0, 1.0]
)

ρ_init = ComplexF64[1.0 0.0; 0.0 0.0]
ρ_goal = ComplexF64[0.0 0.0; 0.0 1.0]

qtraj_density = density_trajectory(open_sys, ρ_init, ρ_goal, 10)
println("✓ Created DensityTrajectory")

prob_density = SmoothPulseProblem(qtraj_density; Q=100.0)
println("✓ Created SmoothPulseProblem")

@test prob_density isa DirectTrajOptProblem
@test length(prob_density.integrators) == 3
println("✓ Structure correct")
println()

# =============================================================================
# Test 4: Custom n_derivatives
# =============================================================================

println("Test 4: Custom number of derivatives")
println("-" ^ 70)

qtraj1 = unitary_trajectory(sys, GATES[:X], 10)

# Only first derivative
prob_1deriv = SmoothPulseProblem(qtraj1; n_derivatives=1)
@test length(prob_1deriv.integrators) == 2  # dynamics + du
@test haskey(prob_1deriv.trajectory.components, :du)
@test !haskey(prob_1deriv.trajectory.components, :ddu)
println("✓ n_derivatives=1 works (2 integrators)")

# No derivatives
qtraj2 = unitary_trajectory(sys, GATES[:Y], 10)
prob_0deriv = SmoothPulseProblem(qtraj2; n_derivatives=0)
@test length(prob_0deriv.integrators) == 1  # only dynamics
@test !haskey(prob_0deriv.trajectory.components, :du)
println("✓ n_derivatives=0 works (1 integrator)")
println()

# =============================================================================
# Summary
# =============================================================================

println("=" ^ 70)
println("All tests passed! ✓")
println("=" ^ 70)
println()
println("Key Features Demonstrated:")
println("  1. Unified interface: SmoothPulseProblem(qtraj)")
println("  2. Type dispatch: works for Unitary/Ket/Density trajectories")
println("  3. Automatic derivative addition via add_control_derivatives()")
println("  4. Default integrators via default_integrator()")
println("  5. Flexible n_derivatives parameter")
println()
println("Usage Pattern:")
println("  qtraj = unitary_trajectory(sys, U_goal, N)")
println("  prob = SmoothPulseProblem(qtraj; Q=100.0, R=1e-2)")
println("  solve!(prob; max_iter=100)")
