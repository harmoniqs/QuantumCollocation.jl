module QuantumConstraints

using ..QuantumObjectives
using ..QuantumObjectives: ket_fidelity_loss, unitary_fidelity_loss

using DirectTrajOpt
using LinearAlgebra
using NamedTrajectories
using PiccoloQuantumObjects

export FinalKetFidelityConstraint
export FinalUnitaryFidelityConstraint
export LeakageConstraint

# ---------------------------------------------------------
#                        Kets
# ---------------------------------------------------------

function FinalKetFidelityConstraint(
    ψ_goal::AbstractVector{<:Complex{Float64}},
    ψ̃_name::Symbol,
    final_fidelity::Float64,
    traj::NamedTrajectory
)
    terminal_constraint = ψ̃ -> [final_fidelity - ket_fidelity_loss(ψ̃, ψ_goal)]

    return NonlinearKnotPointConstraint(
        terminal_constraint,
        ψ̃_name,
        traj,
        equality=false,
        times=[traj.N]
    )
end

# ---------------------------------------------------------
#                        Unitaries
# ---------------------------------------------------------

function FinalUnitaryFidelityConstraint(
    U_goal::AbstractPiccoloOperator,
    Ũ⃗_name::Symbol,
    final_fidelity::Float64,
    traj::NamedTrajectory
)
    terminal_constraint = Ũ⃗ -> [final_fidelity - unitary_fidelity_loss(Ũ⃗, U_goal)]

    return NonlinearKnotPointConstraint(
        terminal_constraint,
        Ũ⃗_name,
        traj,
        equality=false,
        times=[traj.N]
    )
end

function FinalUnitaryFidelityConstraint(
    U_goal::Function,
    Ũ⃗_name::Symbol,
    θ_names::AbstractVector{Symbol},
    final_fidelity::Float64,
    traj::NamedTrajectory
)
    θ_dim = sum(traj.global_dims[n] for n in θ_names)
    function terminal_constraint(z)
        Ũ⃗, θ = z[1:end-θ_dim], z[end-θ_dim+1:end]
        return [final_fidelity - unitary_fidelity_loss(Ũ⃗, U_goal(θ))]
    end

    return NonlinearGlobalKnotPointConstraint(
        terminal_constraint,
        Ũ⃗_name,
        θ_names,
        traj,
        equality=false,
        times=[traj.N]
    )
end

# ---------------------------------------------------------
# Leakage Constraint
# ---------------------------------------------------------

"""
    LeakageConstraint(value, indices, name, traj::NamedTrajectory)

Construct a `KnotPointConstraint` that bounds leakage of `name` at the knot points specified by `times` at any `indices` that are outside the computational subspace.

"""
function LeakageConstraint(
    value::Float64,
    indices::AbstractVector{Int},
    name::Symbol,
    traj::NamedTrajectory;
    times=1:traj.N,
)
    leakage_constraint(x) = abs2.(x[indices]) .- value
    
    return NonlinearKnotPointConstraint(
        leakage_constraint,
        name,
        traj,
        equality=false,
        times=times,
    )
end

end