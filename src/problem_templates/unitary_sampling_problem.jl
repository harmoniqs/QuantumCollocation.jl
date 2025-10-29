export UnitarySamplingProblem


@doc raw"""
    UnitarySamplingProblem(systemns, operator, T, Δt; kwargs...)

A `UnitarySamplingProblem` is a quantum control problem where the goal is to find a control
pulse that generates a target unitary operator for a set of quantum systems.
The controls are shared among all systems, and the optimization seeks to find the control 
pulse that achieves the operator for each system. The idea is to enforce a
robust solution by including multiple systems reflecting the problem uncertainty.

# Arguments
- `systems::AbstractVector{<:AbstractQuantumSystem}`: A vector of quantum systems.
- `operators::AbstractVector{<:AbstractPiccoloOperator}`: A vector of target operators.
- `T::Int`: The number of time steps.
- `Δt::Union{Float64, Vector{Float64}}`: The time step value or vector of time steps.

# Keyword Arguments
- `unitary_integrator=UnitaryIntegrator`: The integrator to use for unitary dynamics.
- `system_weights=fill(1.0, length(systems))`: The weights for each system.
- `init_trajectory::Union{NamedTrajectory, Nothing} = nothing`: The initial trajectory.
- `piccolo_options::PiccoloOptions = PiccoloOptions()`: The Piccolo options.
- `state_name::Symbol = :Ũ⃗`: The name of the state variable.
- `control_name::Symbol = :u`: The name of the control variable.
- `timestep_name::Symbol = :Δt`: The name of the timestep variable.
- `constraints::Vector{<:AbstractConstraint} = AbstractConstraint[]`: The constraints.
- `u_bound::Float64 = 1.0`: The bound for the control amplitudes.
- `u_bounds = fill(u_bound, systems[1].n_drives)`: The bounds for the control amplitudes.
- `u_guess::Union{Matrix{Float64}, Nothing} = nothing`: The initial guess for the control amplitudes.
- `du_bound::Float64 = Inf`: The bound for the control first derivatives.
- `du_bounds = fill(du_bound, systems[1].n_drives)`: The bounds for the control first derivatives.
- `ddu_bound::Float64 = 1.0`: The bound for the control second derivatives.
- `ddu_bounds = fill(ddu_bound, systems[1].n_drives)`: The bounds for the control second derivatives.
- `Δt_min::Float64 = Δt isa Float64 ? 0.5 * Δt : 0.5 * minimum(Δt)`: The minimum time step size.
- `Δt_max::Float64 = Δt isa Float64 ? 2.0 * Δt : 2.0 * maximum(Δt)`: The maximum time step size.
- `Q::Float64 = 100.0`: The fidelity weight.
- `R::Float64 = 1e-2`: The regularization weight.
- `R_u::Union{Float64, Vector{Float64}} = R`: The regularization weight for the control amplitudes.
- `R_du::Union{Float64, Vector{Float64}} = R`: The regularization weight for the control first derivatives.
- `R_ddu::Union{Float64, Vector{Float64}} = R`: The regularization weight for the control second derivatives.
- `piccolo_options::PiccoloOptions = PiccoloOptions()`: The Piccolo options.

"""
function UnitarySamplingProblem(
    systems::AbstractVector{<:AbstractQuantumSystem},
    operators::AbstractVector{<:AbstractPiccoloOperator},
    T::Int,
    Δt::Union{Float64,Vector{Float64}};
    unitary_integrator=UnitaryIntegrator,
    system_weights=fill(1.0, length(systems)),
    init_trajectory::Union{NamedTrajectory, Nothing} = nothing,
    piccolo_options::PiccoloOptions = PiccoloOptions(),
    state_name::Symbol = :Ũ⃗,
    control_name::Symbol=:u,
    timestep_name::Symbol=:Δt,
    constraints::Vector{<:AbstractConstraint}=AbstractConstraint[],
    u_bound::Float64=1.0,
    u_bounds=fill(u_bound, systems[1].n_drives),
    u_guess::Union{Matrix{Float64}, Nothing}=nothing,
    du_bound::Float64=Inf,
    du_bounds=fill(du_bound, systems[1].n_drives),
    ddu_bound::Float64=1.0,
    ddu_bounds=fill(ddu_bound, systems[1].n_drives),
    Δt_min::Float64=Δt isa Float64 ? 0.5 * Δt : 0.5 * minimum(Δt),
    Δt_max::Float64=Δt isa Float64 ? 2.0 * Δt : 2.0 * maximum(Δt),
    Q::Float64=100.0,
    R::Float64=1e-2,
    R_u::Union{Float64,Vector{Float64}}=R,
    R_du::Union{Float64,Vector{Float64}}=R,
    R_ddu::Union{Float64,Vector{Float64}}=R,
    kwargs...
)
    @assert length(systems) == length(operators)
    
    if piccolo_options.verbose
        println("    constructing UnitarySamplingProblem...")
        println("\tusing integrator: $(typeof(unitary_integrator))")
        println("\tusing $(length(systems)) systems")
    end

    # Trajectory
    state_names = [
        Symbol(string(state_name, "_system_", label)) for label ∈ 1:length(systems)
    ]

    if !isnothing(init_trajectory)
        traj = init_trajectory
    else
        trajs = map(zip(systems, operators, state_names)) do (sys, op, st)
            initialize_trajectory(
                op,
                T,
                Δt,
                sys.n_drives,
                (u_bounds, du_bounds, ddu_bounds);
                state_name=st,
                control_name=control_name,
                timestep_name=timestep_name,
                Δt_bounds=(Δt_min, Δt_max),
                geodesic=piccolo_options.geodesic,
                bound_state=piccolo_options.bound_state,
                u_guess=u_guess,
                system=sys,
                rollout_integrator=piccolo_options.rollout_integrator,
                verbose=false # loop
            )
        end

        traj = merge(
            trajs, merge_names=(u=1, du=1, ddu=1, Δt=1), timestep=timestep_name
        )
    end

    control_names = [
        name for name ∈ traj.names
        if endswith(string(name), string(control_name))
    ]

    # Objective
    J = QuadraticRegularizer(control_name, traj, R_u)
    J += QuadraticRegularizer(control_names[2], traj, R_du)
    J += QuadraticRegularizer(control_names[3], traj, R_ddu)

    for (weight, op, name) in zip(system_weights, operators, state_names)
        J += UnitaryInfidelityObjective(op, name, traj; Q=weight * Q)
    end

    # Optional Piccolo constraints and objectives
    J += apply_piccolo_options!(
        piccolo_options, constraints, traj;
        state_names=state_names,
        state_leakage_indices=all(op -> op isa EmbeddedOperator, operators) ?       
            get_iso_vec_leakage_indices.(operators) : 
            nothing
    )

    # Integrators
    integrators = [
        [unitary_integrator(sys, traj, n, control_name) for (sys, n) in zip(systems, state_names)]...,
        DerivativeIntegrator(traj, control_name, control_names[2]),
        DerivativeIntegrator(traj, control_names[2], control_names[3]),
    ]

    return DirectTrajOptProblem(
        traj,
        J,
        integrators;
        constraints=constraints,
    )
end

function UnitarySamplingProblem(
    systems::AbstractVector{<:AbstractQuantumSystem},
    operator::AbstractPiccoloOperator,
    T::Int,
    Δt::Union{Float64,Vector{Float64}};
    kwargs...
)
    # Broadcast the operator to all systems
    return UnitarySamplingProblem(
        systems,
        fill(operator, length(systems)),
        T,
        Δt;
        kwargs...
    )
end

# *************************************************************************** #

@testitem "Sample robustness test" begin
    using PiccoloQuantumObjects

    N = 50
    Δt = 0.2
    timesteps = fill(Δt, N)
    operator = GATES[:H]
    systems(ζ) = QuantumSystem(ζ * GATES[:Z], [GATES[:X], GATES[:Y]], 10.0, [1.0, 1.0])
    
    samples = [0.0, 0.1]
    prob = UnitarySamplingProblem(
        [systems(x) for x in samples], operator, N, Δt,
        piccolo_options=PiccoloOptions(verbose=false)
    )
    solve!(prob, max_iter=100, print_level=1, verbose=false)
    
    base_prob = UnitarySmoothPulseProblem(
        systems(samples[1]), operator, N,
        piccolo_options=PiccoloOptions(verbose=false)
    )
    solve!(base_prob, max_iter=100, verbose=false, print_level=1)
    
    fid = []
    base_fid = []
    for x in range(samples[1], samples[1], length=5)
        push!(fid, unitary_rollout_fidelity(prob.trajectory, systems(0.1), unitary_name=:Ũ⃗_system_1))
        push!(base_fid, unitary_rollout_fidelity(base_prob.trajectory, systems(0.1)))
    end
    @test sum(fid) > sum(base_fid)
end
