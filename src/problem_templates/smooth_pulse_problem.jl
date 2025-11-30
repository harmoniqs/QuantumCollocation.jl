export SmoothPulseProblem

@doc raw"""
    SmoothPulseProblem(qtraj::AbstractQuantumTrajectory; kwargs...)

Construct a `QuantumControlProblem` for smooth pulse optimization that dispatches on the quantum trajectory type.

This unified interface replaces `UnitarySmoothPulseProblem`, `QuantumStateSmoothPulseProblem`, etc.
The problem automatically:
- Adds control derivatives (u̇, ü) to the trajectory
- Creates appropriate dynamics integrator via `default_integrator`
- Adds derivative integrators for smoothness constraints
- Constructs objective with infidelity and regularization terms

# Arguments
- `qtraj::AbstractQuantumTrajectory`: Quantum trajectory (UnitaryTrajectory, KetTrajectory, or DensityTrajectory)

# Keyword Arguments
- `integrator::Union{Nothing, AbstractIntegrator, Vector{<:AbstractIntegrator}}=nothing`: Optional custom integrator(s). If `nothing`, uses default type-appropriate integrator. For KetTrajectory with multiple states, this will automatically return a vector of integrators (one per state). Can also be explicitly provided as a vector.
- `du_bound::Float64=Inf`: Bound on first derivative
- `ddu_bound::Float64=1.0`: Bound on second derivative
- `Q::Float64=100.0`: Weight on infidelity/objective
- `R::Float64=1e-2`: Weight on regularization terms (u, u̇, ü)
- `R_u::Union{Float64, Vector{Float64}}=R`: Weight on control regularization
- `R_du::Union{Float64, Vector{Float64}}=R`: Weight on first derivative regularization
- `R_ddu::Union{Float64, Vector{Float64}}=R`: Weight on second derivative regularization
- `constraints::Vector{<:AbstractConstraint}=AbstractConstraint[]`: Additional constraints
- `piccolo_options::PiccoloOptions=PiccoloOptions()`: Piccolo solver options

# Returns
- `QuantumControlProblem`: Wrapper containing quantum trajectory and optimization problem

# Examples
```julia
# Unitary gate synthesis
sys = QuantumSystem(H_drift, H_drives, T, drive_bounds)
qtraj = UnitaryTrajectory(sys, U_goal, N)
qcp = SmoothPulseProblem(qtraj; Q=100.0, R=1e-2)
solve!(qcp; max_iter=100)

# Quantum state transfer
qtraj = KetTrajectory(sys, ψ_init, ψ_goal, N)
qcp = SmoothPulseProblem(qtraj; Q=50.0, R=1e-3)
solve!(qcp)

# Open system
open_sys = OpenQuantumSystem(H_drift, H_drives, T, drive_bounds)
qtraj = DensityTrajectory(open_sys, ρ_init, ρ_goal, N)
qcp = SmoothPulseProblem(qtraj; Q=100.0)
solve!(qcp)
```
"""
function SmoothPulseProblem(
    qtraj::AbstractQuantumTrajectory;
    integrator::Union{Nothing,AbstractIntegrator,Vector{<:AbstractIntegrator}}=nothing,
    du_bound::Float64=Inf,
    ddu_bound::Float64=1.0,
    Q::Float64=100.0,
    R::Float64=1e-2,
    R_u::Union{Float64,Vector{Float64}}=R,
    R_du::Union{Float64,Vector{Float64}}=R,
    R_ddu::Union{Float64,Vector{Float64}}=R,
    constraints::Vector{<:AbstractConstraint}=AbstractConstraint[],
    piccolo_options::PiccoloOptions=PiccoloOptions(),
)
    if piccolo_options.verbose
        println("    constructing SmoothPulseProblem for $(typeof(qtraj))...")
    end

    # Extract info from quantum trajectory
    sys = PiccoloQuantumObjects.get_system(qtraj)
    state_sym = PiccoloQuantumObjects.get_state_name(qtraj)
    control_sym = PiccoloQuantumObjects.get_control_name(qtraj)

    # Add control derivatives to trajectory (always 2 derivatives for smooth pulses)
    du_bounds = fill(du_bound, sys.n_drives)
    ddu_bounds = fill(ddu_bound, sys.n_drives)

    traj_smooth = add_control_derivatives(
        PiccoloQuantumObjects.get_trajectory(qtraj),
        2;  # Always use 2 derivatives
        control_name=control_sym,
        derivative_bounds=(du_bounds, ddu_bounds)
    )

    # Get control derivative names
    control_names = [
        name for name ∈ traj_smooth.names
        if endswith(string(name), string(control_sym))
    ]

    # Build objective: type-specific infidelity + regularization
    J = _state_objective(qtraj, traj_smooth, state_sym, Q)

    # Add regularization for control and derivatives
    J += QuadraticRegularizer(control_names[1], traj_smooth, R_u)
    J += QuadraticRegularizer(control_names[2], traj_smooth, R_du)
    J += QuadraticRegularizer(control_names[3], traj_smooth, R_ddu)

    # Add optional Piccolo constraints and objectives
    J += _apply_piccolo_options(qtraj, piccolo_options, constraints, traj_smooth, state_sym)

    # Initialize dynamics integrators - handle both single integrator and vector of integrators
    if isnothing(integrator)
        # Use default BilinearIntegrator for the trajectory type
        default_int = BilinearIntegrator(qtraj)
        if default_int isa AbstractVector
            dynamics_integrators = AbstractIntegrator[default_int...]
        else
            dynamics_integrators = AbstractIntegrator[default_int]
        end
    elseif integrator isa AbstractIntegrator
        # Single custom integrator provided
        dynamics_integrators = AbstractIntegrator[integrator]
    else
        # Vector of custom integrators provided
        dynamics_integrators = AbstractIntegrator[integrator...]
    end

    # Start with dynamics integrators
    integrators = copy(dynamics_integrators)

    # Add derivative integrators (always 2 for smooth pulses)
    push!(integrators, DerivativeIntegrator(control_names[1], control_names[2], traj_smooth))

    push!(integrators, DerivativeIntegrator(control_names[2], control_names[3], traj_smooth))

    if qtraj.system.time_dependent
        push!(integrators, TimeIntegrator(:t, traj_smooth))
    end

    prob = DirectTrajOptProblem(
        traj_smooth,
        J,
        integrators;
        constraints=constraints
    )

    return QuantumControlProblem(qtraj, prob)
end

# ============================================================================= #
# Type-specific helper functions
# ============================================================================= #

# Unitary trajectory: single infidelity objective
function _state_objective(qtraj::UnitaryTrajectory, traj::NamedTrajectory, state_sym::Symbol, Q::Float64)
    U_goal = PiccoloQuantumObjects.get_goal(qtraj)
    return UnitaryInfidelityObjective(U_goal, state_sym, traj; Q=Q)
end

# Ket trajectory: single infidelity objective
function _state_objective(qtraj::KetTrajectory, traj::NamedTrajectory, state_sym::Symbol, Q::Float64)
    return KetInfidelityObjective(state_sym, traj; Q=Q)
end

# Density trajectory: no fidelity objective yet (TODO)
function _state_objective(qtraj::DensityTrajectory, traj::NamedTrajectory, state_sym::Symbol, Q::Float64)
    # TODO: Add fidelity objective when we support general mixed state fidelity
    return NullObjective(traj)
end

# Apply Piccolo options with trajectory-type-specific logic
function _apply_piccolo_options(
    qtraj::UnitaryTrajectory,
    piccolo_options::PiccoloOptions,
    constraints::Vector{<:AbstractConstraint},
    traj::NamedTrajectory,
    state_sym::Symbol
)
    U_goal = PiccoloQuantumObjects.get_goal(qtraj)
    return apply_piccolo_options!(
        piccolo_options, constraints, traj;
        state_names=state_sym,
        state_leakage_indices=U_goal isa EmbeddedOperator ?
                              get_iso_vec_leakage_indices(U_goal) :
                              nothing
    )
end

function _apply_piccolo_options(
    qtraj::KetTrajectory,
    piccolo_options::PiccoloOptions,
    constraints::Vector{<:AbstractConstraint},
    traj::NamedTrajectory,
    state_sym::Symbol
)
    return apply_piccolo_options!(
        piccolo_options, constraints, traj;
        state_names=state_sym
    )
end

function _apply_piccolo_options(
    qtraj::DensityTrajectory,
    piccolo_options::PiccoloOptions,
    constraints::Vector{<:AbstractConstraint},
    traj::NamedTrajectory,
    state_sym::Symbol
)
    return apply_piccolo_options!(
        piccolo_options, constraints, traj;
        state_names=state_sym
    )
end

# ============================================================================= #
# Tests
# ============================================================================= #

@testitem "SmoothPulseProblem with UnitaryTrajectory" begin
    using QuantumCollocation
    using PiccoloQuantumObjects
    using DirectTrajOpt

    sys = QuantumSystem(GATES[:Z], [GATES[:X], GATES[:Y]], 1.0, [1.0, 1.0])
    qtraj = UnitaryTrajectory(sys, GATES[:H], 10)
    qcp = SmoothPulseProblem(qtraj; Q=100.0, R=1e-2)

    @test qcp isa QuantumControlProblem
    @test length(qcp.prob.integrators) == 3  # dynamics + du + ddu
    @test haskey(qcp.prob.trajectory.components, :u)
    @test haskey(qcp.prob.trajectory.components, :du)
    @test haskey(qcp.prob.trajectory.components, :ddu)

    # Test accessors
    @test get_system(qcp) === sys
    @test get_goal(qcp) === GATES[:H]
    @test get_trajectory(qcp) === qcp.prob.trajectory
end

@testitem "SmoothPulseProblem with KetTrajectory" begin
    using QuantumCollocation
    using PiccoloQuantumObjects
    using DirectTrajOpt

    sys = QuantumSystem(GATES[:Z], [GATES[:X], GATES[:Y]], 1.0, [1.0, 1.0])
    ψ_init = ComplexF64[1.0, 0.0]
    ψ_goal = ComplexF64[0.0, 1.0]
    qtraj = KetTrajectory(sys, ψ_init, ψ_goal, 10)
    qcp = SmoothPulseProblem(qtraj; Q=50.0, R=1e-3)

    @test qcp isa QuantumControlProblem
    @test length(qcp.prob.integrators) == 3
    @test haskey(qcp.prob.trajectory.components, :du)
    @test haskey(qcp.prob.trajectory.components, :ddu)

    # Test problem solve
    solve!(qcp, max_iter=5, print_level=1, verbose=false)
end

@testitem "SmoothPulseProblem with DensityTrajectory" begin
    using QuantumCollocation
    using PiccoloQuantumObjects
    using DirectTrajOpt

    sys = OpenQuantumSystem(GATES[:Z], [GATES[:X], GATES[:Y]], 1.0, [1.0, 1.0])
    ρ_init = ComplexF64[1.0 0.0; 0.0 0.0]
    ρ_goal = ComplexF64[0.0 0.0; 0.0 1.0]
    qtraj = DensityTrajectory(sys, ρ_init, ρ_goal, 10)
    qcp = SmoothPulseProblem(qtraj; Q=100.0)

    @test qcp isa QuantumControlProblem
    @test length(qcp.prob.integrators) == 3

    # Test problem solve
    solve!(qcp, max_iter=5, print_level=1, verbose=false)
end

# ============================================================================= #
# EnsembleTrajectory Tests (multi-state optimization with shared system)
# ============================================================================= #

@testitem "EnsembleTrajectory multi-state optimization (manual setup)" begin
    using QuantumCollocation
    using PiccoloQuantumObjects
    using DirectTrajOpt
    using NamedTrajectories

    # Use case: Optimize a single pulse that transfers multiple initial states 
    # to their respective goal states (gate-like behavior for kets)
    
    sys = QuantumSystem(GATES[:Z], [GATES[:X], GATES[:Y]], 1.0, [1.0, 1.0])
    
    # Create individual trajectories for different state transfers
    ψ0 = ComplexF64[1.0, 0.0]
    ψ1 = ComplexF64[0.0, 1.0]
    
    qtraj1 = KetTrajectory(sys, ψ0, ψ1, 20)  # |0⟩ → |1⟩
    qtraj2 = KetTrajectory(sys, ψ1, ψ0, 20)  # |1⟩ → |0⟩
    
    # Create ensemble trajectory
    ensemble_qtraj = EnsembleTrajectory([qtraj1, qtraj2])
    
    @test ensemble_qtraj isa EnsembleTrajectory{KetTrajectory}
    @test get_ensemble_state_names(ensemble_qtraj) == [:ψ̃_init_1, :ψ̃_init_2]
    
    # Build combined trajectory with multiple states
    new_traj, state_names = build_ensemble_trajectory_from_trajectories([qtraj1, qtraj2])
    
    @test haskey(new_traj.components, :ψ̃_init_1)
    @test haskey(new_traj.components, :ψ̃_init_2)
    @test haskey(new_traj.components, :u)
    
    # Verify both initial conditions are set correctly
    @test new_traj.initial[:ψ̃_init_1] ≈ ket_to_iso(ψ0)
    @test new_traj.initial[:ψ̃_init_2] ≈ ket_to_iso(ψ1)
    
    # Build objective: sum of infidelities for each state
    goals = get_goal(ensemble_qtraj)
    J = KetInfidelityObjective(goals[1], state_names[1], new_traj; Q=50.0)
    J += KetInfidelityObjective(goals[2], state_names[2], new_traj; Q=50.0)
    
    # Add regularization
    J += QuadraticRegularizer(:u, new_traj, 1e-3)
    
    # Build integrators: one BilinearIntegrator per state
    Ĝ = u_ -> sys.G(u_, 0.0)
    integrators = [
        BilinearIntegrator(Ĝ, state_names[1], :u, new_traj),
        BilinearIntegrator(Ĝ, state_names[2], :u, new_traj),
    ]
    
    # Create and solve problem
    prob = DirectTrajOptProblem(new_traj, J, integrators)
    qcp = QuantumControlProblem(ensemble_qtraj, prob)
    
    solve!(qcp; max_iter=50, verbose=false, print_level=1)
    
    # Both state transfers should have reasonable fidelity
    # (this is essentially implementing an X gate via state transfer)
end

@testitem "EnsembleTrajectory vs SamplingTrajectory distinction" begin
    using QuantumCollocation
    using PiccoloQuantumObjects
    using DirectTrajOpt
    using NamedTrajectories

    # This test documents the key difference:
    # - EnsembleTrajectory: SAME system, DIFFERENT initial/goal states
    # - SamplingTrajectory: DIFFERENT systems, SAME goal
    
    sys = QuantumSystem(GATES[:Z], [GATES[:X]], 1.0, [1.0])
    
    # ===== EnsembleTrajectory setup =====
    # Multiple state transfers on the SAME system
    ψ0 = ComplexF64[1.0, 0.0]
    ψ1 = ComplexF64[0.0, 1.0]
    
    qtraj_ket1 = KetTrajectory(sys, ψ0, ψ1, 10)
    qtraj_ket2 = KetTrajectory(sys, ψ1, ψ0, 10)
    
    ensemble_qtraj = EnsembleTrajectory([qtraj_ket1, qtraj_ket2])
    
    @test get_system(ensemble_qtraj) === sys  # Single system
    @test length(ensemble_qtraj.trajectories) == 2  # Multiple trajectories
    @test get_ensemble_state_names(ensemble_qtraj) == [:ψ̃_init_1, :ψ̃_init_2]
    
    # ===== SamplingTrajectory setup =====
    # Same goal, different systems (robust optimization)
    sys_perturbed = QuantumSystem(1.1 * GATES[:Z], [GATES[:X]], 1.0, [1.0])
    
    qtraj_unitary = UnitaryTrajectory(sys, GATES[:X], 10)
    
    sampling_qtraj = SamplingTrajectory(qtraj_unitary, [sys, sys_perturbed])
    
    @test get_system(sampling_qtraj) === sys  # Nominal system
    @test length(sampling_qtraj.systems) == 2  # Multiple systems
    @test get_ensemble_state_names(sampling_qtraj) == [:Ũ⃗_sample_1, :Ũ⃗_sample_2]
    
    # Key differences:
    # 1. EnsembleTrajectory has `trajectories` field (multiple init/goals)
    # 2. SamplingTrajectory has `systems` field (multiple system params)
    # 3. State naming: _init_ suffix vs _sample_ suffix
end
