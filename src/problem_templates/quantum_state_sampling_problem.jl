export QuantumStateSamplingProblem

"""

"""
function QuantumStateSamplingProblem end

function QuantumStateSamplingProblem(
    systems::AbstractVector{<:AbstractQuantumSystem},
    ψ_inits::AbstractVector{<:AbstractVector{<:AbstractVector{<:ComplexF64}}},
    ψ_goals::AbstractVector{<:AbstractVector{<:AbstractVector{<:ComplexF64}}},
    T::Int,
    Δt::Union{Float64, AbstractVector{Float64}};
    ket_integrator=KetIntegrator,
    system_weights=fill(1.0, length(systems)),
    init_trajectory::Union{NamedTrajectory,Nothing}=nothing,
    state_name::Symbol=:ψ̃,
    control_name::Symbol = :u,
    timestep_name::Symbol=:Δt,
    u_guess::Union{Matrix{Float64},Nothing}=nothing,
    du_bound::Float64 = Inf,
    du_bounds = fill(du_bound, systems[1].n_drives),
    ddu_bound::Float64 = 1.0,
    ddu_bounds = fill(ddu_bound, systems[1].n_drives),
    Δt_min::Float64=0.5 * minimum(Δt),
    Δt_max::Float64=2.0 * maximum(Δt),
    Q::Float64=100.0,
    R=1e-2,
    R_u::Union{Float64,Vector{Float64}} = R,
    R_du::Union{Float64,Vector{Float64}} = R,
    R_ddu::Union{Float64,Vector{Float64}} = R,
    state_leakage_indices::Union{Nothing,AbstractVector{Int}}=nothing,
    constraints::Vector{<:AbstractConstraint}=AbstractConstraint[],
    piccolo_options::PiccoloOptions=PiccoloOptions(),
)
    @assert length(ψ_inits) == length(ψ_goals)

    if piccolo_options.verbose
        println("    constructing QuantumStateSamplingProblem...")
        println("\tusing integrator: $(typeof(ket_integrator))")
        println("\tusing $(length(ψ_inits)) initial state(s)")
    end

    # Outer dimension is the system, inner dimension is the initial state
    state_names = [
        [Symbol(string(state_name) * "$(i)_system_$(j)") for i in eachindex(ψs)]
        for (j, ψs) ∈ zip(eachindex(systems), ψ_inits)
    ]
        
    # Trajectory
    if !isnothing(init_trajectory)
        traj = init_trajectory
    else
        # TODO(main issue with this is that it doesn't handle composite systems correctly, will come back to this.
        trajs = map(zip(systems, state_names, ψ_inits, ψ_goals)) do (sys, names, ψis, ψgs)
            initialize_trajectory(
                ψis,
                ψgs,
                T,
                Δt,
                sys.n_drives,
                (sys.drive_bounds, du_bounds, ddu_bounds);
                state_names=names,
                control_name=control_name,
                timestep_name=timestep_name,
                Δt_bounds=(Δt_min, Δt_max),
                zero_initial_and_final_derivative=piccolo_options.zero_initial_and_final_derivative,
                bound_state=piccolo_options.bound_state,
                u_guess = u_guess,
                system=sys,
                rollout_integrator=piccolo_options.rollout_integrator,
                verbose=false, # loop
            )
        end

        traj = merge(
            trajs,
            merge_names = (u = 1, du = 1, ddu = 1, Δt = 1),
            timestep = timestep_name,
        )
    end

    control_names = [
        name for name ∈ traj.names
        if endswith(string(name), string(control_name))
    ]

    # Objective
    J = QuadraticRegularizer(control_names[1], traj, R_u)
    J += QuadraticRegularizer(control_names[2], traj, R_du)
    J += QuadraticRegularizer(control_names[3], traj, R_ddu)

    for (weight, names) in zip(system_weights, state_names)
        for name in names
            J += KetInfidelityObjective(name, traj, Q=weight * Q)
        end
    end

    # Optional Piccolo constraints and objectives
    J += apply_piccolo_options!(
        piccolo_options, constraints, traj;
        state_names=vcat(state_names...),
        state_leakage_indices=isnothing(state_leakage_indices) ? nothing : fill(state_leakage_indices, length(vcat(state_names...))),
    )

    # Integrators
    state_integrators = []
    for (sys, names) ∈ zip(systems, state_names)
        for name ∈ names
            push!(state_integrators, ket_integrator(sys, traj, name, control_name))
        end
    end

    integrators = [
        state_integrators...,
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

function QuantumStateSamplingProblem(
    systems::AbstractVector{<:AbstractQuantumSystem},
    ψ_inits::AbstractVector{<:AbstractVector{<:ComplexF64}},
    ψ_goals::AbstractVector{<:AbstractVector{<:ComplexF64}},
    args...;
    kwargs...
)
    return QuantumStateSamplingProblem(
        systems, 
        fill(ψ_inits, length(systems)), 
        fill(ψ_goals, length(systems)),
        args...; 
        kwargs...
    )
end

function QuantumStateSamplingProblem(
    systems::AbstractVector{<:AbstractQuantumSystem},
    ψ_init::AbstractVector{<:ComplexF64},
    ψ_goal::AbstractVector{<:ComplexF64},
    args...;
    kwargs...
)
    # Broadcast the initial and target states to all systems
    return QuantumStateSamplingProblem(systems, [ψ_init], [ψ_goal], args...; kwargs...)
end

# *************************************************************************** #

@testitem "Sample systems with single initial, target" begin
    using PiccoloQuantumObjects

    T = 50
    T_max = 1.0
    u_bounds = [(-1.0, 1.0), (-1.0, 1.0)]
    Δt = 0.2
    sys1 = QuantumSystem(0.3 * GATES[:Z], [GATES[:X], GATES[:Y]], T_max, u_bounds)
    sys2 = QuantumSystem(-0.3 * GATES[:Z], [GATES[:X], GATES[:Y]], T_max, u_bounds)
    ψ_init = Vector{ComplexF64}([1.0, 0.0])
    ψ_target = Vector{ComplexF64}([0.0, 1.0])
    
    prob = QuantumStateSamplingProblem(
        [sys1, sys2], ψ_init, ψ_target, T, Δt;
        piccolo_options=PiccoloOptions(verbose=false)
    )
    
    state_name = :ψ̃
    state_names = [n for n ∈ prob.trajectory.names if startswith(string(n), string(state_name))]
    sys_state_names = [n for n ∈ state_names if endswith(string(n), "1")]
    
    # Separately compute all unique initial and goal state fidelities
    inits = []
    for sys in [sys1, sys2]
        push!(
            inits,
            [rollout_fidelity(prob.trajectory, sys; state_name = n) for n in state_names],
        )
    end
    
    solve!(prob, max_iter=20, print_level=1, verbose=false)
        
    for (init, sys) in zip(inits, [sys1, sys2])
        final =
            [rollout_fidelity(prob.trajectory, sys, state_name = n) for n in state_names]
        @test all(final .> init)
    end
end

@testitem "Sample systems with multiple initial, target" begin
    using PiccoloQuantumObjects

    T = 50
    T_max = 1.0
    u_bounds = [(-1.0, 1.0), (-1.0, 1.0)]
    Δt = 0.2
    sys1 = QuantumSystem(0.3 * GATES[:Z], [GATES[:X], GATES[:Y]], T_max, u_bounds)
    sys2 = QuantumSystem(-0.3 * GATES[:Z], [GATES[:X], GATES[:Y]], T_max, u_bounds)

    # Multiple initial and target states
    ψ_inits = Vector{ComplexF64}.([[1.0, 0.0], [0.0, 1.0]])
    ψ_targets = Vector{ComplexF64}.([[0.0, 1.0], [1.0, 0.0]])

    prob = QuantumStateSamplingProblem(
        [sys1, sys2], ψ_inits, ψ_targets, T, Δt;
        piccolo_options=PiccoloOptions(verbose=false)
    )

    state_name = :ψ̃
    state_names = [n for n ∈ prob.trajectory.names if startswith(string(n), string(state_name))]
    sys_state_names = [n for n ∈ state_names if endswith(string(n), "1")]

    # Separately compute all unique initial and goal state fidelities
    inits = []
    for sys in [sys1, sys2]
        push!(
            inits,
            [rollout_fidelity(prob.trajectory, sys; state_name = n) for n in state_names],
        )
    end

    solve!(prob, max_iter=20, print_level=1, verbose=false)

    for (init, sys) in zip(inits, [sys1, sys2])
        final =
            [rollout_fidelity(prob.trajectory, sys, state_name = n) for n in state_names]
        @test all(final .> init)
    end
end

# TODO: Test that u_guess can be used
