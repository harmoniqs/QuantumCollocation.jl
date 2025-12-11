module ProblemTemplates

using ..QuantumObjectives
using ..QuantumConstraints
using ..QuantumIntegrators
using ..QuantumControlProblems: QuantumControlProblem
using ..Options

using TrajectoryIndexingUtils
using NamedTrajectories
using DirectTrajOpt
using PiccoloQuantumObjects
using PiccoloQuantumObjects: build_sampling_trajectory, build_ensemble_trajectory_from_trajectories, 
    get_ensemble_state_names, get_weights, update_base_trajectory, SamplingTrajectory, EnsembleTrajectory

using ExponentialAction
using LinearAlgebra
using SparseArrays
using TestItems

const ⊗ = kron

include("smooth_pulse_problem.jl")
include("minimum_time_problem.jl")
include("sampling_problem.jl")

function apply_piccolo_options!(
    piccolo_options::PiccoloOptions,
    constraints::AbstractVector{<:AbstractConstraint},
    traj::NamedTrajectory;
    state_names::Union{Nothing,Symbol,AbstractVector{Symbol}}=nothing,
    state_leakage_indices::Union{Nothing,AbstractVector{Int},AbstractVector{<:AbstractVector{Int}}}=nothing,
)
    J = NullObjective(traj)

    if piccolo_options.leakage_constraint
        val = piccolo_options.leakage_constraint_value
        if piccolo_options.verbose
            println("\tapplying leakage suppression: $(state_names) < $(val)")
        end

        if isnothing(state_leakage_indices)
            throw(ValueError("Leakage indices are required for leakage suppression."))
        end

        if state_names isa Symbol
            state_names = [state_names]
            state_leakage_indices = [state_leakage_indices]
        end

        for (name, indices) ∈ zip(state_names, state_leakage_indices)
            J += LeakageObjective(indices, name, traj, Qs=fill(piccolo_options.leakage_cost, traj.N))
            push!(constraints, LeakageConstraint(val, indices, name, traj))
        end
    end

    if piccolo_options.timesteps_all_equal
        if piccolo_options.verbose
            println("\tapplying timesteps_all_equal constraint: $(traj.timestep)")
        end
        push!(
            constraints,
            TimeStepsAllEqualConstraint()
        )
    end

    if !isnothing(piccolo_options.complex_control_norm_constraint_name)
        if piccolo_options.verbose
            println("\tapplying complex control norm constraint: $(piccolo_options.complex_control_norm_constraint_name)")
        end
        norm_con = NonlinearKnotPointConstraint(
            u -> [norm(u)^2 - piccolo_options.complex_control_norm_constraint_radius^2],
            piccolo_options.complex_control_norm_constraint_name,
            traj;
            equality=false,
        )
        push!(constraints, norm_con)
    end

    return J
end

end
