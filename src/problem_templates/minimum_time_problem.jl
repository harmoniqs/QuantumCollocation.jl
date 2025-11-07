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
    goal::Union{Nothing, AbstractPiccoloOperator, AbstractVector}=nothing,
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
    traj = get_trajectory(qtraj)
    Δt_bounds = if haskey(traj.bounds, :Δt)
        # Extract scalar bounds from vector bounds (Δt is 1D)
        lb, ub = traj.bounds.Δt
        (lb[1], ub[1])
    else
        nothing
    end
    
    return UnitaryTrajectory(
        get_system(qtraj),
        new_goal,
        traj.N;
        Δt_bounds=Δt_bounds,
    )
end

function _update_goal(qtraj::KetTrajectory, new_goal::Union{AbstractVector, Vector{<:AbstractVector}})
    # Keep initial states, update goal
    traj = get_trajectory(qtraj)
    Δt_bounds = if haskey(traj.bounds, :Δt)
        # Extract scalar bounds from vector bounds (Δt is 1D)
        lb, ub = traj.bounds.Δt
        (lb[1], ub[1])
    else
        nothing
    end
    
    return KetTrajectory(
        get_system(qtraj),
        get_state(qtraj),  # Keep initial state(s)
        new_goal,      # New goal state(s)
        traj.N;
        Δt_bounds=Δt_bounds,
    )
end

function _update_goal(qtraj::DensityTrajectory, new_goal::AbstractMatrix)
    traj = get_trajectory(qtraj)
    Δt_bounds = if haskey(traj.bounds, :Δt)
        # Extract scalar bounds from vector bounds (Δt is 1D)
        lb, ub = traj.bounds.Δt
        (lb[1], ub[1])
    else
        nothing
    end
    
    return DensityTrajectory(
        get_system(qtraj),
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
    U_goal = get_goal(qtraj)
    state_sym = get_state_name(qtraj)
    return FinalUnitaryFidelityConstraint(U_goal, state_sym, final_fidelity, traj)
end

function _final_fidelity_constraint(
    qtraj::KetTrajectory,
    final_fidelity::Float64,
    traj::NamedTrajectory
)
    # Get state names directly from KetTrajectory
    state_names = qtraj.state_names
    
    ψ_goals = get_goal(qtraj)
    
    # If single goal, wrap in array
    if ψ_goals isa AbstractVector{<:Number}
        ψ_goals = [ψ_goals]
    end
    
    @assert length(state_names) == length(ψ_goals) "Number of state names must match number of goals"
    
    # Create constraint for each state
    constraints = [
        FinalKetFidelityConstraint(ψ_goal, state_name, final_fidelity, traj)
        for (ψ_goal, state_name) in zip(ψ_goals, state_names)
    ]
    
    # Return single constraint or vector of constraints
    return length(constraints) == 1 ? constraints[1] : constraints
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
# Tests
# ============================================================================= #

@testitem "MinimumTimeProblem from SmoothPulseProblem (Unitary)" begin
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

@testitem "MinimumTimeProblem with multiple ket states" begin
    using PiccoloQuantumObjects
    using DirectTrajOpt
    using NamedTrajectories
    
    sys = QuantumSystem(GATES[:Z], [GATES[:X], GATES[:Y]], 1.0, [1.0, 1.0])
    
    # Multiple initial and goal states
    ψ_inits = [ComplexF64[1.0, 0.0], ComplexF64[0.0, 1.0]]
    ψ_goals = [ComplexF64[0.0, 1.0], ComplexF64[1.0, 0.0]]
    
    qtraj = KetTrajectory(sys, ψ_inits, ψ_goals, 20; Δt_bounds=(0.01, 0.5))
    
    # Create smooth pulse problem
    qcp_smooth = SmoothPulseProblem(qtraj; Q=50.0, R=1e-3)
    
    # Convert to minimum-time
    qcp_mintime = MinimumTimeProblem(qcp_smooth; final_fidelity=0.85, D=50.0)
    
    @test qcp_mintime isa QuantumControlProblem{KetTrajectory}
    
    # Check that we have multiple fidelity constraints (one per state)
    traj = get_trajectory(qcp_mintime)
    state_names = [name for name in traj.names if startswith(string(name), "ψ̃")]
    @test length(state_names) == 2
    
    # Test that problem can be solved
    solve!(qcp_mintime; max_iter=5, print_level=1, verbose=false)
end
