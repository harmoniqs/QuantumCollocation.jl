export SmoothPulseProblem

@doc raw"""
    SmoothPulseProblem(qtraj::AbstractQuantumTrajectory; kwargs...)

Construct a `DirectTrajOptProblem` for smooth pulse optimization that dispatches on the quantum trajectory type.

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

# Examples
```julia
# Unitary gate synthesis
sys = QuantumSystem(H_drift, H_drives, T, drive_bounds)
qtraj = UnitaryTrajectory(sys, U_goal, N)
prob = SmoothPulseProblem(qtraj; Q=100.0, R=1e-2)

# Quantum state transfer
qtraj = KetTrajectory(sys, ψ_init, ψ_goal, N)
prob = SmoothPulseProblem(qtraj; Q=50.0, R=1e-3)

# Open system
open_sys = OpenQuantumSystem(H_drift, H_drives, T, drive_bounds)
qtraj = DensityTrajectory(open_sys, ρ_init, ρ_goal, N)
prob = SmoothPulseProblem(qtraj; Q=100.0)
```
"""
function SmoothPulseProblem(
    qtraj::AbstractQuantumTrajectory;
    integrator::Union{Nothing, AbstractIntegrator, Vector{<:AbstractIntegrator}}=nothing,
    du_bound::Float64=Inf,
    ddu_bound::Float64=1.0,
    Q::Float64=100.0,
    R::Float64=1e-2,
    R_u::Union{Float64, Vector{Float64}}=R,
    R_du::Union{Float64, Vector{Float64}}=R,
    R_ddu::Union{Float64, Vector{Float64}}=R,
    constraints::Vector{<:AbstractConstraint}=AbstractConstraint[],
    piccolo_options::PiccoloOptions=PiccoloOptions(),
)
    if piccolo_options.verbose
        println("    constructing SmoothPulseProblem for $(typeof(qtraj))...")
    end
    
    # Extract info from quantum trajectory
    sys = system(qtraj)
    state_sym = state_name(qtraj)
    control_sym = control_name(qtraj)
    
    # Add control derivatives to trajectory (always 2 derivatives for smooth pulses)
    du_bounds = fill(du_bound, sys.n_drives)
    ddu_bounds = fill(ddu_bound, sys.n_drives)
    
    traj_smooth = add_control_derivatives(
        trajectory(qtraj),
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
    
    # Initialize integrators - handle both single integrator and vector of integrators
    if isnothing(integrator)
        default_integrator = BilinearIntegrator(qtraj)
    else
        default_integrator = integrator
    end
    
    # Convert to vector if needed
    if default_integrator isa AbstractIntegrator
        integrators = AbstractIntegrator[default_integrator]
    else
        # Already a vector
        integrators = AbstractIntegrator[default_integrator...]
    end
   
    # Derivative integrators (always 2 for smooth pulses)
    push!(integrators, DerivativeIntegrator(control_names[1], control_names[2]))
    push!(integrators, DerivativeIntegrator(control_names[2], control_names[3]))
    
    return DirectTrajOptProblem(
        traj_smooth,
        J,
        integrators;
        constraints=constraints
    )
end

# ============================================================================= #
# Type-specific helper functions
# ============================================================================= #

# Unitary trajectory: single infidelity objective
function _state_objective(qtraj::UnitaryTrajectory, traj::NamedTrajectory, state_sym::Symbol, Q::Float64)
    U_goal = goal(qtraj)
    return UnitaryInfidelityObjective(U_goal, state_sym, traj; Q=Q)
end

# Ket trajectory: infidelity objective for each state
function _state_objective(qtraj::KetTrajectory, traj::NamedTrajectory, state_sym::Symbol, Q::Float64)
    # Use state_names from qtraj instead of searching
    state_names = [
        name for name ∈ traj.names
            if startswith(string(name), string(state_sym))
    ]
    
    # Start with first state objective
    J = KetInfidelityObjective(state_names[1], traj; Q=Q)
    
    # Add remaining states
    for name in state_names[2:end]
        J += KetInfidelityObjective(name, traj; Q=Q)
    end
    
    return J
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
    U_goal = goal(qtraj)
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
    state_names = [
        name for name ∈ traj.names
            if startswith(string(name), string(state_sym))
    ]
    
    return apply_piccolo_options!(
        piccolo_options, constraints, traj;
        state_names=state_names
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
    using PiccoloQuantumObjects
    using DirectTrajOpt
    
    sys = QuantumSystem(GATES[:Z], [GATES[:X], GATES[:Y]], 1.0, [1.0, 1.0])
    qtraj = UnitaryTrajectory(sys, GATES[:H], 10)
    prob = SmoothPulseProblem(qtraj; Q=100.0, R=1e-2)
    
    @test prob isa DirectTrajOptProblem
    @test length(prob.integrators) == 3  # dynamics + du + ddu
    @test haskey(prob.trajectory.components, :u)
    @test haskey(prob.trajectory.components, :du)
    @test haskey(prob.trajectory.components, :ddu)
end

@testitem "SmoothPulseProblem with KetTrajectory" begin
    using PiccoloQuantumObjects
    using DirectTrajOpt
    
    sys = QuantumSystem(GATES[:Z], [GATES[:X], GATES[:Y]], 1.0, [1.0, 1.0])
    ψ_init = ComplexF64[1.0, 0.0]
    ψ_goal = ComplexF64[0.0, 1.0]
    qtraj = KetTrajectory(sys, ψ_init, ψ_goal, 10)
    prob = SmoothPulseProblem(qtraj; Q=50.0, R=1e-3)
    
    @test prob isa DirectTrajOptProblem
    @test length(prob.integrators) == 3
    @test haskey(prob.trajectory.components, :du)
    @test haskey(prob.trajectory.components, :ddu)

    # Test problem solve
    solve!(prob, max_iter=5, print_level=1, verbose=false)
end

@testitem "SmoothPulseProblem with KetTrajectory (multiple states)" begin
    using PiccoloQuantumObjects
    using DirectTrajOpt
    
    sys = QuantumSystem(GATES[:Z], [GATES[:X], GATES[:Y]], 1.0, [1.0, 1.0])
    
    # Multiple initial and goal states
    ψ_inits = [
        ComplexF64[1.0, 0.0],
        ComplexF64[0.0, 1.0]
    ]
    ψ_goals = [
        ComplexF64[0.0, 1.0],
        ComplexF64[1.0, 0.0]
    ]
    
    qtraj = KetTrajectory(sys, ψ_inits, ψ_goals, 10)
    prob = SmoothPulseProblem(qtraj; Q=50.0, R=1e-3)
    
    @test prob isa DirectTrajOptProblem
    @test haskey(prob.trajectory.components, :du)
    @test haskey(prob.trajectory.components, :ddu)
    
    # Should have multiple state variables
    @test haskey(prob.trajectory.components, :ψ̃1)
    @test haskey(prob.trajectory.components, :ψ̃2)

    num_states = length(ψ_inits)

    # Total integrators: num_states dynamics + 2 derivative integrators
    @test length(prob.integrators) == num_states + 2
    
    # Check that the objective includes contributions from both states
    @test prob.objective isa Objective

    # Test problem solve
    solve!(prob, max_iter=5, print_level=1, verbose=false)
end

@testitem "SmoothPulseProblem with DensityTrajectory" begin
    using PiccoloQuantumObjects
    using DirectTrajOpt
    
    sys = OpenQuantumSystem(GATES[:Z], [GATES[:X], GATES[:Y]], 1.0, [1.0, 1.0])
    ρ_init = ComplexF64[1.0 0.0; 0.0 0.0]
    ρ_goal = ComplexF64[0.0 0.0; 0.0 1.0]
    qtraj = DensityTrajectory(sys, ρ_init, ρ_goal, 10)
    prob = SmoothPulseProblem(qtraj; Q=100.0)
    
    @test prob isa DirectTrajOptProblem
    @test length(prob.integrators) == 3

    # Test problem solve
    solve!(prob, max_iter=5, print_level=1, verbose=false)
end
