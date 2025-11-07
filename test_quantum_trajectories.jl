using QuantumCollocation
using PiccoloQuantumObjects

# Test UnitaryTrajectory
sys = QuantumSystem(
    GATES[:Z],              # H_drift
    [GATES[:X], GATES[:Y]], # H_drives
    1.0,                    # T_max
    [1.0, 1.0]             # drive_bounds
)

U_goal = GATES[:H]  # Hadamard gate
N = 10

println("Testing unitary_trajectory...")
qtraj = unitary_trajectory(sys, U_goal, N)
println("  Type: ", typeof(qtraj))
println("  Is UnitaryTrajectory: ", qtraj isa UnitaryTrajectory)
println("  System: ", get_system(qtraj) === sys)
println("  Goal: ", get_goal(qtraj) == U_goal)
println("  State name: ", get_state_name(qtraj))
println("  Control name: ", get_control_name(qtraj))
println("  Can access bounds: ", haskey(qtraj.bounds, :u))
println("  Can access components: ", haskey(qtraj.components, :Ũ⃗))
println("  Can index: ", size(qtraj[:u]))
println()

# Test KetTrajectory
ψ_init = ComplexF64[1.0, 0.0]
ψ_goal = ComplexF64[0.0, 1.0]

println("Testing ket_trajectory...")
qtraj2 = ket_trajectory(sys, ψ_init, ψ_goal, N)
println("  Type: ", typeof(qtraj2))
println("  Is KetTrajectory: ", qtraj2 isa KetTrajectory)
println("  System: ", get_system(qtraj2) === sys)
println("  Goal: ", get_goal(qtraj2) == ψ_goal)
println("  State name: ", get_state_name(qtraj2))
println("  Control name: ", get_control_name(qtraj2))
println()

# Test DensityTrajectory
open_sys = OpenQuantumSystem(
    GATES[:Z],              # H_drift
    [GATES[:X], GATES[:Y]], # H_drives
    1.0,                    # T_max
    [1.0, 1.0]             # drive_bounds
)

ρ_init = ComplexF64[1.0 0.0; 0.0 0.0]
ρ_goal = ComplexF64[0.0 0.0; 0.0 1.0]

println("Testing density_trajectory...")
qtraj3 = density_trajectory(open_sys, ρ_init, ρ_goal, N)
println("  Type: ", typeof(qtraj3))
println("  Is DensityTrajectory: ", qtraj3 isa DensityTrajectory)
println("  System: ", get_system(qtraj3) === open_sys)
println("  Goal: ", get_goal(qtraj3) == ρ_goal)
println("  State name: ", get_state_name(qtraj3))
println("  Control name: ", get_control_name(qtraj3))
println()

println("All basic tests passed! ✓")
