export UnitaryFreePhaseProblem


"""
    UnitaryFreePhaseProblem(system::AbstractQuantumSystem, goal::Function, T::Int, Δt::Union{Float64, AbstractVector{Float64}}; kwargs...)

Construct a `DirectTrajOptProblem` for a free-time unitary gate problem with smooth control pulses enforced by constraining the second derivative of the pulse trajectory. The problem follows the same structure as `UnitarySmoothPulseProblem`, but allows for free global phases on the goal unitary, via cosines and sines parameterizing phase variables.
    
The `goal` function should accept a vector of global phases `[cos(θ); sin(θ)]` and return an `AbstractPiccoloOperator`.

# Arguments
- `system::AbstractQuantumSystem`: the system to be controlled
- `goal::Function`: function that takes phase vector and returns target unitary
- `T::Int`: the number of knot points
- `Δt::Union{Float64, AbstractVector{Float64}}`: the time step size or vector of time steps

# Keyword Arguments
- `unitary_integrator=UnitaryIntegrator`: the integrator to use for unitary dynamics
- `state_name::Symbol = :Ũ⃗`: the name of the state
- `control_name::Symbol = :a`: the name of the control
- `timestep_name::Symbol = :Δt`: the name of the timestep
- `phase_name::Symbol = :θ`: the name of the phase variable
- `init_phases::Union{AbstractVector{Float64}, Nothing}=nothing`: initial phase values
- `init_trajectory::Union{NamedTrajectory, Nothing}=nothing`: an initial trajectory to use
- `a_guess::Union{Matrix{Float64}, Nothing}=nothing`: an initial guess for the control pulses
- `a_bound::Float64=1.0`: the bound on the control pulse
- `a_bounds=fill(a_bound, system.n_drives)`: the bounds on the control pulses
- `da_bound::Float64=Inf`: the bound on the control pulse derivative
- `da_bounds::Vector{Float64}=fill(da_bound, system.n_drives)`: the bounds on the control pulse derivatives
- `dda_bound::Float64=1.0`: the bound on the control pulse second derivative
- `dda_bounds::Vector{Float64}=fill(dda_bound, system.n_drives)`: the bounds on the control pulse second derivatives
- `Δt_min::Float64=Δt isa Float64 ? 0.5 * Δt : 0.5 * minimum(Δt)`: the minimum time step size
- `Δt_max::Float64=Δt isa Float64 ? 2.0 * Δt : 2.0 * maximum(Δt)`: the maximum time step size
- `Q::Float64=100.0`: the weight on the infidelity objective
- `R=1e-2`: the weight on the regularization terms
- `R_a::Union{Float64, Vector{Float64}}=R`: the weight on the regularization term for the control pulses
- `R_da::Union{Float64, Vector{Float64}}=R`: the weight on the regularization term for the control pulse derivatives
- `R_dda::Union{Float64, Vector{Float64}}=R`: the weight on the regularization term for the control pulse second derivatives
- `constraints::Vector{<:AbstractConstraint}=AbstractConstraint[]`: the constraints to enforce
- `piccolo_options::PiccoloOptions=PiccoloOptions()`: the options for the Piccolo solver
"""
function UnitaryFreePhaseProblem(
    system::AbstractQuantumSystem,
    goal::Function,
    T::Int,
    Δt::Union{Float64, AbstractVector{Float64}};
    unitary_integrator=UnitaryIntegrator,
    state_name::Symbol = :Ũ⃗,
    control_name::Symbol = :a,
    timestep_name::Symbol = :Δt,
    phase_name::Symbol = :θ,
    init_phases::Union{AbstractVector{Float64}, Nothing}=nothing,
    init_trajectory::Union{NamedTrajectory, Nothing}=nothing,
    a_guess::Union{Matrix{Float64}, Nothing}=nothing,
    a_bound::Float64=1.0,
    a_bounds=fill(a_bound, system.n_drives),
    da_bound::Float64=Inf,
    da_bounds::Vector{Float64}=fill(da_bound, system.n_drives),
    dda_bound::Float64=1.0,
    dda_bounds::Vector{Float64}=fill(dda_bound, system.n_drives),
    Δt_min::Float64=Δt isa Float64 ? 0.5 * Δt : 0.5 * minimum(Δt),
    Δt_max::Float64=Δt isa Float64 ? 2.0 * Δt : 2.0 * maximum(Δt),
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
        println("\tinitial free phases: $(phase_name) = $(init_phases)")
    end

    # Construct global phases
    x = Symbol("cos$(phase_name)")
    y = Symbol("sin$(phase_name)")
    phase_names = [x, y]
    if piccolo_options.verbose
        println("\tusing global names: ", phase_names)
    end

    # Trajectory
    if !isnothing(init_trajectory)
        trig_phases = [init_trajectory.global_data[x]; init_trajectory.global_data[y]]
        @assert goal(trig_phases) isa AbstractPiccoloOperator "expected goal([cos(θ); sin(θ)])"
        eval_goal = goal(trig_phases)
        traj = init_trajectory
    else
        phase_data = (; x => cos.(init_phases), y => sin.(init_phases))
        trig_phases = [phase_data[x]; phase_data[y]]
        @assert goal(trig_phases) isa AbstractPiccoloOperator "expected goal([cos(θ); sin(θ)])"
        eval_goal = goal(trig_phases)
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
            global_component_data=phase_data,
        )
    end

    # Objective
    J = UnitaryFreePhaseInfidelityObjective(goal,  state_name,  phase_names, traj; Q=Q)

    control_names = [
        name for name ∈ traj.names
            if endswith(string(name), string(control_name))
    ]

    J += QuadraticRegularizer(control_names[1], traj, R_a)
    J += QuadraticRegularizer(control_names[2], traj, R_da)
    J += QuadraticRegularizer(control_names[3], traj, R_dda)

    # Optional Piccolo constraints and objectives
    J += apply_piccolo_options!(
        piccolo_options, constraints, traj;
        state_names=state_name,
        state_leakage_indices=eval_goal isa EmbeddedOperator ? 
            get_iso_vec_leakage_indices(eval_goal) :
            nothing
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

# *************************************************************************** #

@testitem "UnitaryFreePhaseProblem: basic construction" begin
    using PiccoloQuantumObjects

    sys = QuantumSystem(0.3 * PAULIS.X, [PAULIS.Y], 10.0, [1.0])
    U_goal = GATES.Z
    N = 51
    Δt = 0.2

    function virtual_z(z::AbstractVector{<:Real})        
        x, y = z[1:length(z)÷2], z[1+length(z)÷2:end]
        # U_goal ≈ R * U
        R = reduce(kron, [xᵢ * PAULIS.I + im * yᵢ * PAULIS.Z for (xᵢ, yᵢ) in zip(x, y)])
        return R'U_goal
    end

    initial_phases = [pi/3]

    prob = UnitaryFreePhaseProblem(
        sys, virtual_z, N, Δt, 
        init_phases=initial_phases,
        piccolo_options=PiccoloOptions(verbose=false),
        phase_name=:ϕ,
    )

    @test prob isa DirectTrajOptProblem
    @test length(prob.trajectory.global_data) == 2length(initial_phases)
    @test prob.trajectory.global_names == (:cosϕ, :sinϕ)

    before = copy(prob.trajectory.global_data)
    solve!(prob, max_iter=10, verbose=false, print_level=1)
    @test prob.trajectory.global_data != before
end