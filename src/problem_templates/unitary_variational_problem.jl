export UnitaryVariationalProblem


"""
    UnitaryVariationalProblem(
        system::VariationalQuantumSystem,
        goal::AbstractPiccoloOperator,
        T::Int,
        Δt::Union{Float64, <:AbstractVector{Float64}};
        robust_times::AbstractVector{<:Union{AbstractVector{Int}, Nothing}}=[nothing for s ∈ system.G_vars],
        sensitive_times::AbstractVector{<:Union{AbstractVector{Int}, Nothing}}=[nothing for s ∈ system.G_vars],
        kwargs...
    )

Constructs a unitary variational problem for optimizing quantum control trajectories.

# Arguments

- `system::VariationalQuantumSystem`: The quantum system to be controlled, containing variational parameters.
- `goal::AbstractPiccoloOperator`: The target operator or state to achieve at the end of the trajectory.
- `T::Int`: The total number of timesteps in the trajectory.
- `Δt::Union{Float64, <:AbstractVector{Float64}}`: The timestep duration or a vector of timestep durations.

# Keyword Arguments

- `robust_times::AbstractVector{<:AbstractVector{Int}}=[Int[] for s ∈ system.G_vars]`: Times at which robustness to variations in the trajectory is enforced.
- `sensitive_times::AbstractVector{<:AbstractVector{Int}}=[Int[] for s ∈ system.G_vars]`: Times at which sensitivity to variations in the trajectory is enhanced.
- `variational_integrator=VariationalUnitaryIntegrator`: The integrator used for unitary evolution.
- `variational_scales::AbstractVector{<:Float64}=fill(1.0, length(system.G_vars))`: Scaling factors for the variational state variables.
- `state_name::Symbol = :Ũ⃗`: The name of the state variable in the trajectory.
- `variational_state_name::Symbol = :Ũ⃗ᵥ`: The name of the variational state variable.
- `control_name::Symbol = :a`: The name of the control variable.
- `timestep_name::Symbol = :Δt`: The name of the timestep variable.
- `init_trajectory::Union{NamedTrajectory, Nothing}=nothing`: An optional initial trajectory to start optimization.
- `a_bound::Float64=1.0`: The bound for the control variable `a`.
- `a_bounds=fill(a_bound, system.n_drives)`: Bounds for each control variable.
- `da_bound::Float64=Inf`: The bound for the derivative of the control variable.
- `da_bounds=fill(da_bound, system.n_drives)`: Bounds for each derivative of the control variable.
- `dda_bound::Float64=1.0`: The bound for the second derivative of the control variable.
- `dda_bounds=fill(dda_bound, system.n_drives)`: Bounds for each second derivative of the control variable.
- `Δt_min::Float64=Δt isa Float64 ? 0.5 * Δt : 0.5 * minimum(Δt)`: Minimum allowed timestep duration.
- `Δt_max::Float64=Δt isa Float64 ? 2.0 * Δt : 2.0 * maximum(Δt)`: Maximum allowed timestep duration.
- `Q::Float64=100.0`: Weight for the unitary infidelity objective.
- `Q_s::Float64=1e-2`: Weight for sensitivity objectives.
- `Q_r::Float64=100.0`: Weight for robustness objectives.
- `R=1e-2`: Regularization weight for control variables.
- `R_a::Union{Float64, Vector{Float64}}=R`: Regularization weights for control.
- `R_da::Union{Float64, Vector{Float64}}=R`: Regularization weights for control derivative.
- `R_dda::Union{Float64, Vector{Float64}}=R`: Regularization weights for control second derivative.
- `constraints::Vector{<:AbstractConstraint}=AbstractConstraint[]`: Additional constraints for the optimization problem.
- `piccolo_options::PiccoloOptions=PiccoloOptions()`: Options for configuring the Piccolo optimization framework.

# Returns

A `DirectTrajOptProblem` object representing the optimization problem, including the 
trajectory, objective, integrators, and constraints.

# Notes

This function constructs a trajectory optimization problem for quantum control using 
variational principles. It supports robust and sensitive trajectory design, regularization, 
and optional constraints. The problem is solved using the Piccolo optimization framework.

"""
function UnitaryVariationalProblem end

function UnitaryVariationalProblem(
    system::VariationalQuantumSystem,
    goal::AbstractPiccoloOperator,
    T::Int,
    Δt::Union{Float64, <:AbstractVector{Float64}};
    robust_times::AbstractVector{<:AbstractVector{Int}}=[Int[] for s ∈ system.G_vars],
    sensitive_times::AbstractVector{<:AbstractVector{Int}}=[Int[] for s ∈ system.G_vars],
    variational_integrator=VariationalUnitaryIntegrator,
    variational_scales::AbstractVector{<:Float64}=fill(1.0, length(system.G_vars)),
    state_name::Symbol = :Ũ⃗,
    variational_state_name::Symbol = :Ũ⃗ᵥ,
    control_name::Symbol = :a,
    timestep_name::Symbol = :Δt,
    init_trajectory::Union{NamedTrajectory, Nothing}=nothing,
    a_bound::Float64=1.0,
    a_bounds=fill(a_bound, system.n_drives),
    da_bound::Float64=Inf,
    da_bounds=fill(da_bound, system.n_drives),
    dda_bound::Float64=1.0,
    dda_bounds=fill(dda_bound, system.n_drives),
    Δt_min::Float64=Δt isa Float64 ? 0.5 * Δt : 0.5 * minimum(Δt),
    Δt_max::Float64=Δt isa Float64 ? 2.0 * Δt : 2.0 * maximum(Δt),
    Q::Float64=100.0,
    Q_s::Float64=1e-2,
    Q_r::Float64=100.0,
    R=1e-2,
    R_a::Union{Float64, Vector{Float64}}=R,
    R_da::Union{Float64, Vector{Float64}}=R,
    R_dda::Union{Float64, Vector{Float64}}=R,
    constraints::Vector{<:AbstractConstraint}=AbstractConstraint[],
    piccolo_options::PiccoloOptions=PiccoloOptions(),
)
    if piccolo_options.verbose
        println("    constructing UnitaryVariationalProblem...")
        println("\tusing integrator: $(typeof(variational_integrator))")
        println("\ttotal variational parameters: $(length(system.G_vars))")
        if !isempty(robust_times)
            println("\trobust knot points: $(robust_times)")
        end
        if !isempty(sensitive_times)
            println("\tsensitive knot points: $(sensitive_times)")
        end
    end

    # Trajectory
    if !isnothing(init_trajectory)
        traj = deepcopy(init_trajectory)
    else
        traj = initialize_trajectory(
            goal,
            T,
            Δt,
            system.n_drives,
            (a_bounds, da_bounds, dda_bounds);
            state_name=state_name,
            control_name=control_name,
            timestep_name=timestep_name,
            Δt_bounds=(Δt_min, Δt_max),
            zero_initial_and_final_derivative=piccolo_options.zero_initial_and_final_derivative,
            geodesic=piccolo_options.geodesic,
            bound_state=piccolo_options.bound_state,
            rollout_integrator=piccolo_options.rollout_integrator,
            verbose=piccolo_options.verbose
        )
    end

    _, Ũ⃗_vars =  variational_unitary_rollout(
        traj, 
        system;
        unitary_name=state_name,
        drive_name=control_name
    )

    # Add variational components to the trajectory
    var_state_names = Tuple(
        Symbol(string(variational_state_name) * "$(i)") for i in eachindex(system.G_vars)
    )
    var_comps_inits = NamedTuple{var_state_names}(
        Ũ⃗_v[:, 1] / scale for (scale, Ũ⃗_v) in zip(variational_scales, Ũ⃗_vars)
    )
    var_comps_data = NamedTuple{var_state_names}(
        Ũ⃗_v / scale for (scale, Ũ⃗_v) in zip(variational_scales, Ũ⃗_vars)
    )
    traj = add_components(
        traj, 
        var_comps_data; 
        type=:state,
        initial=merge(traj.initial, var_comps_inits)
    )

    control_names = [
        name for name ∈ traj.names
            if endswith(string(name), string(control_name))
    ]

    # objective
    J = UnitaryInfidelityObjective(goal, state_name, traj; Q=Q)
    J += QuadraticRegularizer(control_names[1], traj, R_a)
    J += QuadraticRegularizer(control_names[2], traj, R_da)
    J += QuadraticRegularizer(control_names[3], traj, R_dda)

    # sensitivity
    for (name, scale, s, r) ∈ zip(
        var_state_names, 
        variational_scales, 
        sensitive_times, 
        robust_times
    )
        @assert isdisjoint(s, r)
        J += UnitarySensitivityObjective(
            name, 
            traj, 
            [s; r]; 
            Qs=[fill(-Q_s, length(s)); fill(Q_r, length(r))], 
            scale=scale
        )
    end
    
    # Optional Piccolo constraints and objectives
    J += apply_piccolo_options!(
        piccolo_options, constraints, traj;
        state_names=state_name,
        state_leakage_indices=goal isa EmbeddedOperator ? 
            get_iso_vec_leakage_indices(goal) :
            nothing
    )

    integrators = [
        variational_integrator(
            system, traj, state_name, [var_state_names...], control_name, 
            scales=variational_scales
        ),
        DerivativeIntegrator(traj, control_name, control_names[2]),
        DerivativeIntegrator(traj, control_names[2], control_names[3]),
    ]

    return DirectTrajOptProblem(
        traj,
        J,
        integrators;
        constraints=constraints
    )
end

# *************************************************************************** #

@testitem "Sensitive and robust" begin
    using LinearAlgebra
    using PiccoloQuantumObjects

    system = QuantumSystem([PAULIS.X, PAULIS.Y], 10.0, [1.0, 1.0])
    varsys = VariationalQuantumSystem([PAULIS.X, PAULIS.Y], [PAULIS.X], 10.0, [1.0, 1.0])
    N = 50
    Δt = 0.2

    sense_scale = 8.0
    sense_prob = UnitaryVariationalProblem(
        varsys, GATES.X, N, Δt, 
        variational_scales=[sense_scale], 
        sensitive_times=[[N]],
        piccolo_options=PiccoloOptions(verbose=false)
    )
    solve!(sense_prob, max_iter=20, print_level=1, verbose=false)

    rob_scale = 1 / 8.0
    rob_prob = UnitaryVariationalProblem(
        varsys, GATES.X, N, Δt, 
        variational_scales=[rob_scale], 
        robust_times=[[N]],
        piccolo_options=PiccoloOptions(verbose=false)
    )
    solve!(rob_prob, max_iter=20, print_level=1, verbose=false)

    sense_n = norm(sense_scale * sense_prob.trajectory.Ũ⃗ᵥ1)
    rob_n = norm(rob_scale * rob_prob.trajectory.Ũ⃗ᵥ1)
    @test sense_n > rob_n
end