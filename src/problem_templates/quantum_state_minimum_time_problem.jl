export QuantumStateMinimumTimeProblem


"""
    QuantumStateMinimumTimeProblem(traj, sys, obj, integrators, constraints; kwargs...)
    QuantumStateMinimumTimeProblem(prob; kwargs...)

Construct a `DirectTrajOptProblem` for the minimum time problem of reaching a target state.

# Keyword Arguments
- `state_name::Symbol=:ψ̃`: The symbol for the state variables.
- `final_fidelity::Union{Real, Nothing}=nothing`: The final fidelity.
- `D=1.0`: The cost weight on the time.
- `piccolo_options::PiccoloOptions=PiccoloOptions()`: The Piccolo options.
"""
function QuantumStateMinimumTimeProblem end

function QuantumStateMinimumTimeProblem(
    trajectory::NamedTrajectory,
    ψ_goals::AbstractVector{<:AbstractVector{<:ComplexF64}},
    objective::Objective,
    dynamics::TrajectoryDynamics,
    constraints::Vector{<:AbstractConstraint};
    state_name::Symbol=:ψ̃,
    final_fidelity::Float64=1.0,
    D=100.0,
    piccolo_options::PiccoloOptions=PiccoloOptions(),
)
    state_names = [name for name in trajectory.names if startswith(string(name), string(state_name))]
    @assert length(state_names) ≥ 1 "No matching states found in trajectory"
    @assert length(state_names) == length(ψ_goals) "Number of states and goals must match"

    if piccolo_options.verbose
        println("    constructing QuantumStateMinimumTimeProblem...")
        println("\tfinal fidelity: $(final_fidelity)")
    end

    objective += MinimumTimeObjective(
        trajectory, D=D, timesteps_all_equal=piccolo_options.timesteps_all_equal
    )

    for (state_name, ψ_goal) in zip(state_names, ψ_goals)
        fidelity_constraint = FinalKetFidelityConstraint(
            ψ_goal,
            state_name,
            final_fidelity,
            trajectory,
        )

        push!(constraints, fidelity_constraint)
    end

    return DirectTrajOptProblem(
        trajectory,
        objective,
        dynamics,
        constraints
    )
end

# TODO: goals from trajectory?

function QuantumStateMinimumTimeProblem(
    prob::DirectTrajOptProblem,
    ψ_goals::AbstractVector{<:AbstractVector{<:ComplexF64}};
    objective::Objective=prob.objective,
    constraints::AbstractVector{<:AbstractConstraint}=prob.constraints,
    kwargs...
)
    return QuantumStateMinimumTimeProblem(
        prob.trajectory,
        ψ_goals,
        objective,
        prob.dynamics,
        constraints;
        kwargs...
    )
end

function QuantumStateMinimumTimeProblem(
    prob::DirectTrajOptProblem,
    ψ_goal::AbstractVector{<:ComplexF64};
    kwargs...
)
    return QuantumStateMinimumTimeProblem(prob, [ψ_goal]; kwargs...)
end

# *************************************************************************** #

@testitem "Test quantum state minimum time" begin
    using NamedTrajectories
    using PiccoloQuantumObjects 
    using QuantumCollocation

    T = 51
    Δt = 0.2
    sys = QuantumSystem(0.1 * GATES[:Z], [GATES[:X], GATES[:Y]])
    ψ_init = Vector{ComplexF64}[[1.0, 0.0]]
    ψ_target = Vector{ComplexF64}[[0.0, 1.0]]

    prob = QuantumStateSmoothPulseProblem(
        sys, ψ_init, ψ_target, T, Δt;
        piccolo_options=QuantumCollocation.Options.PiccoloOptions(verbose=false)
    )
    initial = sum(get_timesteps(prob.trajectory))

    min_prob = QuantumStateMinimumTimeProblem(
        prob, ψ_target,
        piccolo_options=PiccoloOptions(verbose=false)
    )
    solve!(min_prob, max_iter=50, verbose=false, print_level=1)
    final = sum(get_timesteps(min_prob.trajectory))

    @test final < initial
end
