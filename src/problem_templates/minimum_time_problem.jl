export MinimumTimeProblem

@doc raw"""
    MinimumTimeProblem(qcp::QuantumControlProblem; kwargs...)

Convert an existing quantum control problem to minimum-time optimization.

**IMPORTANT**: This function requires an existing `QuantumControlProblem` (e.g., from `SmoothPulseProblem`).
It cannot be created directly from a quantum trajectory. The workflow is:
1. Create base problem with `SmoothPulseProblem` (or similar)
2. Solve base problem to get feasible solution
3. Convert to minimum-time with `MinimumTimeProblem`

This ensures the problem starts from a good initialization and maintains solution quality
through the final fidelity constraint.

# Type Dispatch
Automatically handles different quantum trajectory types through the type parameter:
- `QuantumControlProblem{UnitaryTrajectory}` → Uses `FinalUnitaryFidelityConstraint`
- `QuantumControlProblem{KetTrajectory}` → Uses `FinalKetFidelityConstraint`
- `QuantumControlProblem{DensityTrajectory}` → Not yet implemented

The optimization problem is:

```math
\begin{aligned}
\underset{\vec{\tilde{q}}, u, \Delta t}{\text{minimize}} & \quad
J_{\text{original}}(\vec{\tilde{q}}, u) + D \sum_t \Delta t_t \\
\text{ subject to } & \quad \text{original dynamics \& constraints} \\
& F_{\text{final}} \geq F_{\text{threshold}} \\
& \quad \Delta t_{\text{min}} \leq \Delta t_t \leq \Delta t_{\text{max}} \\
\end{aligned}
```

where q represents the quantum state (unitary, ket, or density matrix).

# Arguments
- `qcp::QuantumControlProblem`: Existing quantum control problem to convert

# Keyword Arguments
- `final_fidelity::Float64=0.99`: Minimum fidelity constraint at final time
- `D::Float64=100.0`: Weight on minimum-time objective ∑Δt
- `piccolo_options::PiccoloOptions=PiccoloOptions()`: Piccolo solver options

# Returns
- `QuantumControlProblem`: New problem with minimum-time objective and fidelity constraint

# Examples
```julia
# Standard workflow
sys = QuantumSystem(H_drift, H_drives, T, drive_bounds)
qtraj = UnitaryTrajectory(sys, U_goal, N; Δt_bounds=(0.01, 0.5))

# Step 1: Create and solve base smooth pulse problem
qcp_smooth = SmoothPulseProblem(qtraj; Q=100.0, R=1e-2)
solve!(qcp_smooth; max_iter=100)

# Step 2: Convert to minimum-time
qcp_mintime = MinimumTimeProblem(qcp_smooth; final_fidelity=0.99, D=100.0)
solve!(qcp_mintime; max_iter=100)

# Compare durations
duration_before = sum(get_timesteps(get_trajectory(qcp_smooth)))
duration_after = sum(get_timesteps(get_trajectory(qcp_mintime)))
@assert duration_after <= duration_before

# Nested transformations also work
qcp_final = MinimumTimeProblem(
    RobustnessProblem(qcp_smooth);  # Future feature
    final_fidelity=0.95
)
```

# Convenience Constructors

You can also update the goal when creating minimum-time problem:

```julia
# Different goal for minimum-time optimization
qcp_mintime = MinimumTimeProblem(qcp_smooth; goal=U_goal_new, final_fidelity=0.98)
```
"""
function MinimumTimeProblem(
    qcp::QuantumControlProblem{QT};
    goal::Union{Nothing,AbstractPiccoloOperator,AbstractVector}=nothing,
    final_fidelity::Float64=0.99,
    D::Float64=100.0,
    piccolo_options::PiccoloOptions=PiccoloOptions()
) where {QT<:AbstractQuantumTrajectory}

    if piccolo_options.verbose
        println("    constructing MinimumTimeProblem from QuantumControlProblem{$QT}...")
        println("\tfinal fidelity constraint: $(final_fidelity)")
        println("\tminimum-time weight D: $(D)")
    end

    # Copy trajectory and constraints from original problem
    traj = deepcopy(qcp.prob.trajectory)
    constraints = deepcopy(qcp.prob.constraints)

    # Add minimum-time objective to existing objective
    J = qcp.prob.objective + MinimumTimeObjective(traj, D=D)

    # Use updated goal if provided, otherwise use original
    qtraj_for_constraint = if isnothing(goal)
        qcp.qtraj
    else
        # Create new quantum trajectory with updated goal
        _update_goal(qcp.qtraj, goal)
    end

    # Add final fidelity constraint - dispatches on QT type parameter!
    fidelity_constraint = _final_fidelity_constraint(
        qtraj_for_constraint,
        final_fidelity,
        traj
    )

    # Handle single constraint or multiple constraints
    if fidelity_constraint isa AbstractVector
        append!(constraints, fidelity_constraint)
    else
        push!(constraints, fidelity_constraint)
    end

    # Create new optimization problem with same integrators
    new_prob = DirectTrajOptProblem(
        traj,
        J,
        qcp.prob.integrators,
        constraints
    )

    # Return new QuantumControlProblem with potentially updated qtraj
    return QuantumControlProblem(qtraj_for_constraint, new_prob)
end

# ============================================================================= #
# Type-specific helper functions
# ============================================================================= #

# Helper to update goal in quantum trajectory (convenience constructor support)
function _update_goal(qtraj::UnitaryTrajectory, new_goal::AbstractPiccoloOperator)
    # Create new trajectory with updated goal, preserving Δt_bounds
    traj = PiccoloQuantumObjects.get_trajectory(qtraj)
    Δt_bounds = if haskey(traj.bounds, :Δt)
        # Extract scalar bounds from vector bounds (Δt is 1D)
        lb, ub = traj.bounds.Δt
        (lb[1], ub[1])
    else
        nothing
    end

    return UnitaryTrajectory(
        PiccoloQuantumObjects.get_system(qtraj),
        new_goal,
        traj.N;
        Δt_bounds=Δt_bounds,
    )
end

function _update_goal(qtraj::KetTrajectory, new_goal::AbstractVector{<:Number})
    # Keep initial state, update goal
    traj = PiccoloQuantumObjects.get_trajectory(qtraj)
    Δt_bounds = if haskey(traj.bounds, :Δt)
        # Extract scalar bounds from vector bounds (Δt is 1D)
        lb, ub = traj.bounds.Δt
        (lb[1], ub[1])
    else
        nothing
    end

    return KetTrajectory(
        PiccoloQuantumObjects.get_system(qtraj),
        get_state(qtraj),  # Keep initial state
        new_goal,          # New goal state
        traj.N;
        Δt_bounds=Δt_bounds,
    )
end

function _update_goal(qtraj::DensityTrajectory, new_goal::AbstractMatrix)
    traj = PiccoloQuantumObjects.get_trajectory(qtraj)
    Δt_bounds = if haskey(traj.bounds, :Δt)
        # Extract scalar bounds from vector bounds (Δt is 1D)
        lb, ub = traj.bounds.Δt
        (lb[1], ub[1])
    else
        nothing
    end

    return DensityTrajectory(
        PiccoloQuantumObjects.get_system(qtraj),
        get_state(qtraj),  # Keep initial state
        new_goal,      # New goal state
        traj.N;
        Δt_bounds=Δt_bounds,
    )
end

# Fidelity constraint functions - dispatch on quantum trajectory type

function _final_fidelity_constraint(
    qtraj::UnitaryTrajectory,
    final_fidelity::Float64,
    traj::NamedTrajectory
)
    U_goal = PiccoloQuantumObjects.get_goal(qtraj)
    state_sym = PiccoloQuantumObjects.get_state_name(qtraj)
    return FinalUnitaryFidelityConstraint(U_goal, state_sym, final_fidelity, traj)
end

function _final_fidelity_constraint(
    qtraj::KetTrajectory,
    final_fidelity::Float64,
    traj::NamedTrajectory
)
    ψ_goal = PiccoloQuantumObjects.get_goal(qtraj)
    state_sym = PiccoloQuantumObjects.get_state_name(qtraj)
    return FinalKetFidelityConstraint(ψ_goal, state_sym, final_fidelity, traj)
end

function _final_fidelity_constraint(
    qtraj::DensityTrajectory,
    final_fidelity::Float64,
    traj::NamedTrajectory
)
    # TODO: Implement density matrix fidelity constraint when available
    throw(ArgumentError("Final fidelity constraint for DensityTrajectory not yet implemented"))
end

# ============================================================================= #
# EnsembleTrajectory Support
# ============================================================================= #

"""
    _final_fidelity_constraint(qtraj::EnsembleTrajectory, final_fidelity, traj)

Create fidelity constraints for each trajectory in an EnsembleTrajectory.
Returns a vector of constraints, one per ensemble member.
"""
function _final_fidelity_constraint(
    qtraj::EnsembleTrajectory,
    final_fidelity::Float64,
    traj::NamedTrajectory
)
    state_names = get_ensemble_state_names(qtraj)
    goals = PiccoloQuantumObjects.get_goal(qtraj)
    
    # Create one fidelity constraint per ensemble member
    constraints = [
        _ensemble_fidelity_constraint(qtraj.trajectories[1], goal, name, final_fidelity, traj)
        for (goal, name) in zip(goals, state_names)
    ]
    
    return constraints
end

# Dispatch on base trajectory type for ensemble fidelity constraint
function _ensemble_fidelity_constraint(
    base_qtraj::UnitaryTrajectory,
    goal::AbstractPiccoloOperator,
    state_sym::Symbol,
    final_fidelity::Float64,
    traj::NamedTrajectory
)
    return FinalUnitaryFidelityConstraint(goal, state_sym, final_fidelity, traj)
end

function _ensemble_fidelity_constraint(
    base_qtraj::KetTrajectory,
    goal::AbstractVector,
    state_sym::Symbol,
    final_fidelity::Float64,
    traj::NamedTrajectory
)
    return FinalKetFidelityConstraint(goal, state_sym, final_fidelity, traj)
end

function _ensemble_fidelity_constraint(
    base_qtraj::DensityTrajectory,
    goal::AbstractMatrix,
    state_sym::Symbol,
    final_fidelity::Float64,
    traj::NamedTrajectory
)
    throw(ArgumentError("Final fidelity constraint for DensityTrajectory ensemble not yet implemented"))
end

# Update goal for EnsembleTrajectory is not typically needed since goals are embedded
# in the individual trajectories. But we provide a fallback that errors clearly.
function _update_goal(qtraj::EnsembleTrajectory, new_goal)
    throw(ArgumentError(
        "Updating goals for EnsembleTrajectory is not directly supported. " *
        "Create a new EnsembleTrajectory with the desired goals instead."
    ))
end

# ============================================================================= #
# Tests
# ============================================================================= #

@testitem "MinimumTimeProblem from SmoothPulseProblem (Unitary)" begin
    using QuantumCollocation
    using PiccoloQuantumObjects
    using DirectTrajOpt
    using NamedTrajectories

    # Create and solve smooth pulse problem
    sys = QuantumSystem(0.1 * GATES[:Z], [GATES[:X], GATES[:Y]], 1.0, [1.0, 1.0])
    qtraj = UnitaryTrajectory(sys, GATES[:H], 50; Δt_bounds=(0.01, 0.5))
    qcp_smooth = SmoothPulseProblem(qtraj; Q=100.0, R=1e-2)

    solve!(qcp_smooth; max_iter=50, verbose=false, print_level=1)
    duration_before = sum(get_timesteps(get_trajectory(qcp_smooth)))

    # Convert to minimum-time problem
    qcp_mintime = MinimumTimeProblem(qcp_smooth; final_fidelity=0.95, D=100.0)

    @test qcp_mintime isa QuantumControlProblem
    @test qcp_mintime isa QuantumControlProblem{UnitaryTrajectory}
    @test haskey(get_trajectory(qcp_mintime).components, :du)
    @test haskey(get_trajectory(qcp_mintime).components, :ddu)

    # Test accessors
    @test get_system(qcp_mintime) === sys
    @test get_goal(qcp_mintime) === GATES[:H]

    # Solve minimum-time problem
    solve!(qcp_mintime; max_iter=50, verbose=false, print_level=1)
    duration_after = sum(get_timesteps(get_trajectory(qcp_mintime)))

    # Duration should decrease (or stay same if already optimal)
    @test duration_after <= duration_before
end

@testitem "MinimumTimeProblem from SmoothPulseProblem (Ket)" begin
    using QuantumCollocation
    using PiccoloQuantumObjects
    using DirectTrajOpt
    using NamedTrajectories

    sys = QuantumSystem(GATES[:Z], [GATES[:X], GATES[:Y]], 1.0, [1.0, 1.0])
    ψ_init = ComplexF64[1.0, 0.0]
    ψ_goal = ComplexF64[0.0, 1.0]
    qtraj = KetTrajectory(sys, ψ_init, ψ_goal, 30; Δt_bounds=(0.01, 0.5))

    # Create smooth pulse problem
    qcp_smooth = SmoothPulseProblem(qtraj; Q=50.0, R=1e-3)
    solve!(qcp_smooth; max_iter=10, verbose=false, print_level=1)

    # Convert to minimum-time
    qcp_mintime = MinimumTimeProblem(qcp_smooth; final_fidelity=0.90, D=50.0)

    @test qcp_mintime isa QuantumControlProblem{KetTrajectory}
    @test haskey(get_trajectory(qcp_mintime).components, :du)

    # Test problem solve
    solve!(qcp_mintime; max_iter=10, print_level=1, verbose=false)
end

@testitem "MinimumTimeProblem with updated goal" begin
    using QuantumCollocation
    using PiccoloQuantumObjects
    using DirectTrajOpt

    sys = QuantumSystem(GATES[:Z], [GATES[:X], GATES[:Y]], 1.0, [1.0, 1.0])
    qtraj = UnitaryTrajectory(sys, GATES[:H], 20; Δt_bounds=(0.01, 0.5))

    qcp_smooth = SmoothPulseProblem(qtraj; Q=100.0, R=1e-2)
    solve!(qcp_smooth; max_iter=5, verbose=false, print_level=1)

    # Create minimum-time with different goal
    qcp_mintime = MinimumTimeProblem(
        qcp_smooth;
        goal=GATES[:X],  # Different goal!
        final_fidelity=0.95,
        D=100.0
    )

    @test qcp_mintime isa QuantumControlProblem
    @test get_goal(qcp_mintime) === GATES[:X]  # Goal should be updated
    @test get_goal(qcp_mintime) !== get_goal(qcp_smooth)  # Different from original
end

@testitem "MinimumTimeProblem type dispatch" begin
    using QuantumCollocation
    using PiccoloQuantumObjects
    using DirectTrajOpt

    # Test that type parameter is correct for different trajectory types
    sys = QuantumSystem(GATES[:Z], [GATES[:X], GATES[:Y]], 1.0, [1.0, 1.0])

    # Unitary
    qtraj_u = UnitaryTrajectory(sys, GATES[:H], 10)
    qcp_u = SmoothPulseProblem(qtraj_u)
    qcp_mintime_u = MinimumTimeProblem(qcp_u)
    @test qcp_mintime_u isa QuantumControlProblem{UnitaryTrajectory}

    # Ket
    ψ_init = ComplexF64[1.0, 0.0]
    ψ_goal = ComplexF64[0.0, 1.0]
    qtraj_k = KetTrajectory(sys, ψ_init, ψ_goal, 10)
    qcp_k = SmoothPulseProblem(qtraj_k)
    qcp_mintime_k = MinimumTimeProblem(qcp_k)
    @test qcp_mintime_k isa QuantumControlProblem{KetTrajectory}
end

@testitem "MinimumTimeProblem with SamplingTrajectory" begin
    using QuantumCollocation
    using PiccoloQuantumObjects
    using DirectTrajOpt
    using NamedTrajectories

    # Robust minimum-time gate optimization
    sys_nominal = QuantumSystem(0.1 * GATES[:Z], [GATES[:X]], 1.0, [1.0])
    sys_perturbed = QuantumSystem(0.11 * GATES[:Z], [GATES[:X]], 1.0, [1.0])

    qtraj = UnitaryTrajectory(sys_nominal, GATES[:X], 30; Δt_bounds=(0.01, 0.5))
    qcp = SmoothPulseProblem(qtraj; Q=100.0, R=1e-2)

    # Create sampling problem
    sampling_prob = SamplingProblem(qcp, [sys_nominal, sys_perturbed]; Q=100.0)
    solve!(sampling_prob; max_iter=20, verbose=false, print_level=1)

    duration_before = sum(get_timesteps(get_trajectory(sampling_prob)))

    # Convert to minimum-time
    mintime_prob = MinimumTimeProblem(sampling_prob; final_fidelity=0.90, D=50.0)

    @test mintime_prob isa QuantumControlProblem{<:SamplingTrajectory}
    @test mintime_prob.qtraj isa SamplingTrajectory

    # Should have fidelity constraints for each sample
    # (one per system in the sampling ensemble)

    # Solve minimum-time
    solve!(mintime_prob; max_iter=50, verbose=false, print_level=1)

    duration_after = sum(get_timesteps(get_trajectory(mintime_prob)))
    @test duration_after <= duration_before * 1.1  # Allow small tolerance
end

@testitem "MinimumTimeProblem with SamplingTrajectory (Ket)" begin
    using QuantumCollocation
    using PiccoloQuantumObjects
    using DirectTrajOpt

    # Robust minimum-time state transfer
    sys_nominal = QuantumSystem(GATES[:Z], [GATES[:X], GATES[:Y]], 1.0, [1.0, 1.0])
    sys_perturbed = QuantumSystem(1.1 * GATES[:Z], [GATES[:X], GATES[:Y]], 1.0, [1.0, 1.0])

    ψ_init = ComplexF64[1.0, 0.0]
    ψ_goal = ComplexF64[0.0, 1.0]
    qtraj = KetTrajectory(sys_nominal, ψ_init, ψ_goal, 25; Δt_bounds=(0.01, 0.5))

    qcp = SmoothPulseProblem(qtraj; Q=50.0, R=1e-3)

    # Create sampling problem
    sampling_prob = SamplingProblem(qcp, [sys_nominal, sys_perturbed]; Q=50.0)
    solve!(sampling_prob; max_iter=15, verbose=false, print_level=1)

    # Convert to minimum-time
    mintime_prob = MinimumTimeProblem(sampling_prob; final_fidelity=0.85, D=30.0)

    @test mintime_prob isa QuantumControlProblem{<:SamplingTrajectory}

    # Solve
    solve!(mintime_prob; max_iter=15, verbose=false, print_level=1)
end
@testitem "MinimumTimeProblem with EnsembleTrajectory (Ket)" begin
    using QuantumCollocation
    using PiccoloQuantumObjects
    using DirectTrajOpt
    using NamedTrajectories
    using LinearAlgebra

    # Multi-state minimum-time optimization (X gate via state transfer)
    sys = QuantumSystem(0.1 * GATES[:Z], [GATES[:X], GATES[:Y]], 1.0, [1.0, 1.0])
    
    # Create ensemble of ket trajectories for X gate
    ψ0 = ComplexF64[1.0, 0.0]
    ψ1 = ComplexF64[0.0, 1.0]
    
    qtraj1 = KetTrajectory(sys, ψ0, ψ1, 30; Δt_bounds=(0.01, 0.5))  # |0⟩ → |1⟩
    qtraj2 = KetTrajectory(sys, ψ1, ψ0, 30; Δt_bounds=(0.01, 0.5))  # |1⟩ → |0⟩
    
    ensemble_qtraj = EnsembleTrajectory([qtraj1, qtraj2])
    
    # Create and solve smooth pulse problem
    qcp_smooth = SmoothPulseProblem(ensemble_qtraj; Q=100.0, R=1e-2)
    solve!(qcp_smooth; max_iter=30, verbose=false, print_level=1)
    
    duration_before = sum(get_timesteps(get_trajectory(qcp_smooth)))
    
    # Convert to minimum-time problem
    qcp_mintime = MinimumTimeProblem(qcp_smooth; final_fidelity=0.90, D=50.0)
    
    @test qcp_mintime isa QuantumControlProblem{<:EnsembleTrajectory}
    @test qcp_mintime.qtraj isa EnsembleTrajectory{KetTrajectory}
    
    # Should have fidelity constraints for each ensemble member
    # (one per trajectory in the ensemble)
    
    # Solve minimum-time problem
    solve!(qcp_mintime; max_iter=30, verbose=false, print_level=1)
    
    duration_after = sum(get_timesteps(get_trajectory(qcp_mintime)))
    
    # Duration should not increase significantly
    @test duration_after <= duration_before * 1.1

    # Verify fidelity constraints are met for both states
    traj = get_trajectory(qcp_mintime)
    state_names = get_ensemble_state_names(qcp_mintime.qtraj)
    goals = get_goal(qcp_mintime.qtraj)
    
    for (name, goal) in zip(state_names, goals)
        ψ̃_final = traj[end][name]
        ψ_final = iso_to_ket(ψ̃_final)
        fid = fidelity(ψ_final, goal)
        @test fid >= 0.89  # Just under constraint to account for numerical tolerance
    end
end

@testitem "MinimumTimeProblem with EnsembleTrajectory (Unitary)" begin
    using QuantumCollocation
    using PiccoloQuantumObjects
    using DirectTrajOpt
    using NamedTrajectories
    using LinearAlgebra

    # Multi-goal minimum-time optimization
    sys = QuantumSystem(0.1 * GATES[:Z], [GATES[:X], GATES[:Y]], 1.0, [1.0, 1.0])
    
    # Create ensemble of unitary trajectories
    # Note: Using same goal for both since different goals are very hard to optimize together
    qtraj1 = UnitaryTrajectory(sys, GATES[:X], 30; Δt_bounds=(0.01, 0.5))
    qtraj2 = UnitaryTrajectory(sys, GATES[:X], 30; Δt_bounds=(0.01, 0.5))
    
    ensemble_qtraj = EnsembleTrajectory([qtraj1, qtraj2])
    
    # Create and solve smooth pulse problem
    qcp_smooth = SmoothPulseProblem(ensemble_qtraj; Q=100.0, R=1e-2)
    solve!(qcp_smooth; max_iter=30, verbose=false, print_level=1)
    
    duration_before = sum(get_timesteps(get_trajectory(qcp_smooth)))
    
    # Convert to minimum-time problem
    qcp_mintime = MinimumTimeProblem(qcp_smooth; final_fidelity=0.90, D=50.0)
    
    @test qcp_mintime isa QuantumControlProblem{<:EnsembleTrajectory}
    @test qcp_mintime.qtraj isa EnsembleTrajectory{UnitaryTrajectory}
    
    # Solve minimum-time problem
    solve!(qcp_mintime; max_iter=30, verbose=false, print_level=1)
    
    duration_after = sum(get_timesteps(get_trajectory(qcp_mintime)))
    
    # Duration should not increase significantly
    @test duration_after <= duration_before * 1.1

    # Verify fidelity constraints are met for all unitaries
    traj = get_trajectory(qcp_mintime)
    state_names = get_ensemble_state_names(qcp_mintime.qtraj)
    goals = get_goal(qcp_mintime.qtraj)
    
    for (name, goal) in zip(state_names, goals)
        Ũ⃗_final = traj[end][name]
        U_final = iso_vec_to_operator(Ũ⃗_final)
        fid = unitary_fidelity(U_final, goal)
        @test fid >= 0.89  # Just under constraint to account for numerical tolerance
    end
end

@testitem "MinimumTimeProblem with time-dependent UnitaryTrajectory" begin
    using QuantumCollocation
    using PiccoloQuantumObjects
    using DirectTrajOpt
    using NamedTrajectories
    using LinearAlgebra

    # Time-dependent Hamiltonian
    ω = 2π * 5.0
    H(u, t) = GATES[:Z] + u[1] * cos(ω * t) * GATES[:X] + u[2] * sin(ω * t) * GATES[:Y]
    
    sys = QuantumSystem(H, 1.0, [1.0, 1.0]; time_dependent=true)
    
    qtraj = UnitaryTrajectory(sys, GATES[:H], 30; Δt_bounds=(0.01, 0.5))
    
    # Verify time is in trajectory
    @test haskey(qtraj.components, :t)
    
    # Create and solve smooth pulse problem
    qcp_smooth = SmoothPulseProblem(qtraj; Q=100.0, R=1e-2)
    
    # Should have TimeIntegrator
    has_time_integrator = any(int -> int isa TimeIntegrator, qcp_smooth.prob.integrators)
    @test has_time_integrator
    
    solve!(qcp_smooth; max_iter=30, verbose=false, print_level=1)
    
    duration_before = sum(get_timesteps(get_trajectory(qcp_smooth)))
    
    # Convert to minimum-time
    qcp_mintime = MinimumTimeProblem(qcp_smooth; final_fidelity=0.85, D=50.0)
    
    @test qcp_mintime isa QuantumControlProblem{UnitaryTrajectory}
    
    # Solve minimum-time problem
    solve!(qcp_mintime; max_iter=30, verbose=false, print_level=1)
    
    duration_after = sum(get_timesteps(get_trajectory(qcp_mintime)))
    @test duration_after <= duration_before * 1.1
end

@testitem "MinimumTimeProblem with time-dependent KetTrajectory" begin
    using QuantumCollocation
    using PiccoloQuantumObjects
    using DirectTrajOpt
    using NamedTrajectories
    using LinearAlgebra

    # Time-dependent Hamiltonian
    ω = 2π * 5.0
    H(u, t) = GATES[:Z] + u[1] * cos(ω * t) * GATES[:X]
    
    sys = QuantumSystem(H, 1.0, [1.0]; time_dependent=true)
    
    ψ_init = ComplexF64[1.0, 0.0]
    ψ_goal = ComplexF64[0.0, 1.0]
    
    qtraj = KetTrajectory(sys, ψ_init, ψ_goal, 30; Δt_bounds=(0.01, 0.5))
    
    # Verify time is in trajectory
    @test haskey(qtraj.components, :t)
    
    # Create and solve smooth pulse problem
    qcp_smooth = SmoothPulseProblem(qtraj; Q=50.0, R=1e-3)
    
    # Should have TimeIntegrator
    has_time_integrator = any(int -> int isa TimeIntegrator, qcp_smooth.prob.integrators)
    @test has_time_integrator
    
    solve!(qcp_smooth; max_iter=30, verbose=false, print_level=1)
    
    duration_before = sum(get_timesteps(get_trajectory(qcp_smooth)))
    
    # Convert to minimum-time
    qcp_mintime = MinimumTimeProblem(qcp_smooth; final_fidelity=0.85, D=50.0)
    
    @test qcp_mintime isa QuantumControlProblem{KetTrajectory}
    
    # Solve minimum-time problem
    solve!(qcp_mintime; max_iter=30, verbose=false, print_level=1)
    
    duration_after = sum(get_timesteps(get_trajectory(qcp_mintime)))
    @test duration_after <= duration_before * 1.1
end