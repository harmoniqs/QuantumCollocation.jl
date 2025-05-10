export UnitaryFreePhaseProblem

"""

"""
function UnitaryFreePhaseProblem(
    system::AbstractQuantumSystem,
    goal::Function,
    T::Int,
    Δt::Union{Float64, <:AbstractVector{Float64}},
    initial_phases::AbstractVector{Float64};
    unitary_integrator=UnitaryIntegrator,
    state_name::Symbol = :Ũ⃗,
    control_name::Symbol = :a,
    timestep_name::Symbol = :Δt,
    phase_name::Symbol = :θ,
    init_trajectory::Union{NamedTrajectory, Nothing}=nothing,
    a_guess::Union{Matrix{Float64}, Nothing}=nothing,
    a_bound::Float64=1.0,
    a_bounds=fill(a_bound, system.n_drives),
    da_bound::Float64=Inf,
    da_bounds::Vector{Float64}=fill(da_bound, system.n_drives),
    dda_bound::Float64=1.0,
    dda_bounds::Vector{Float64}=fill(dda_bound, system.n_drives),
    Δt_min::Float64=Δt isa Float64 ? 0.5 * Δt : 0.5 * mean(Δt),
    Δt_max::Float64=Δt isa Float64 ? 1.5 * Δt : 1.5 * mean(Δt),
    Q::Float64=100.0,
    R=1e-2,
    R_a::Union{Float64, Vector{Float64}}=R,
    R_da::Union{Float64, Vector{Float64}}=R,
    R_dda::Union{Float64, Vector{Float64}}=R,
    constraints::Vector{<:AbstractConstraint}=AbstractConstraint[],
    piccolo_options::PiccoloOptions=PiccoloOptions(),
)
    if piccolo_options.verbose
        println("    constructing UnitaryFreePhaseProblem...")
        println("\tusing integrator: $(typeof(unitary_integrator))")
        println("\tinitial free phases: $(phase_name) = $(initial_phases)")
    end

    # Construct global phases
    x = Symbol("cos$(phase_name)")
    y = Symbol("sin$(phase_name)")
    phase_names = [x, y]
    phase_data = (; x => cos.(initial_phases), y => sin.(initial_phases))
    trig_phases = [phase_data[x]; phase_data[y]]
    if piccolo_options.verbose
        println("\tusing global names: ", phase_names)
    end

    @assert goal(trig_phases) isa AbstractPiccoloOperator "expected goal([cos(θ); sin(θ)])"
    eval_goal = goal(trig_phases)

    # Trajectory
    if !isnothing(init_trajectory)
        traj = init_trajectory
    else
        traj = initialize_trajectory(
            eval_goal,
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
            a_guess=a_guess,
            system=system,
            rollout_integrator=piccolo_options.rollout_integrator,
            verbose=piccolo_options.verbose,
            global_data=phase_data,
        )
    end

    # Objective
    J = UnitaryFreePhaseInfidelityObjective(
        goal, 
        state_name, 
        phase_names,
        traj;
        Q=Q
    )

    control_names = [
        name for name ∈ traj.names
            if endswith(string(name), string(control_name))
    ]

    J += QuadraticRegularizer(control_names[1], traj, R_a)
    J += QuadraticRegularizer(control_names[2], traj, R_da)
    J += QuadraticRegularizer(control_names[3], traj, R_dda)

    # Optional Piccolo constraints and objectives
    ProblemTemplates.apply_piccolo_options!(
        J, constraints, piccolo_options, traj, state_name, timestep_name;
        state_leakage_indices=eval_goal isa EmbeddedOperator ? get_leakage_indices(eval_goal) : nothing
    )
    
    # Phase constraint
    function phase_norm(z)
        x, y = z[1:length(z) ÷ 2], z[length(z) ÷ 2 + 1:end]
        return x .^ 2 + y .^2 .- 1
    end
    push!(constraints, NonlinearGlobalConstraint(phase_norm, phase_names, traj))

    integrators = [
        unitary_integrator(system, traj, state_name, control_name),
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