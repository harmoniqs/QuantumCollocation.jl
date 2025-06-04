module QuantumConstraints

using ..QuantumObjectives

using NamedTrajectories
using PiccoloQuantumObjects
using DirectTrajOpt

export FinalKetFidelityConstraint
export FinalUnitaryFidelityConstraint


# ---------------------------------------------------------
#                        Kets
# ---------------------------------------------------------

function FinalKetFidelityConstraint(
    ψ_goal::AbstractVector{<:Complex{Float64}},
    ψ̃_name::Symbol,
    final_fidelity::Float64,
    traj::NamedTrajectory
)
    terminal_constraint = ψ̃ -> [
        final_fidelity - QuantumObjectives.ket_fidelity_loss(ψ̃, ψ_goal)
    ]

    return NonlinearKnotPointConstraint(
        terminal_constraint,
        ψ̃_name,
        traj,
        equality=false,
        times=[traj.T]
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
    terminal_constraint = Ũ⃗ -> [
        final_fidelity - QuantumObjectives.unitary_fidelity_loss(Ũ⃗, U_goal)
    ]

    return NonlinearKnotPointConstraint(
        terminal_constraint,
        Ũ⃗_name,
        traj,
        equality=false,
        times=[traj.T]
    )
end

function FinalUnitaryFidelityConstraint(
    U_goal::Function,
    Ũ⃗_name::Symbol,
    θ_names::AbstractVector{Symbol},
    final_fidelity::Float64,
    traj::NamedTrajectory
)
    d = sum(traj.global_dims[n] for n in θ_names)
    function terminal_constraint(z)
        Ũ⃗, θ = z[1:end-d], z[end-d+1:end]
        return [
            # final_fidelity - QuantumObjectives.unitary_fidelity_loss(Ũ⃗, U_goal(θ))
            log(final_fidelity / QuantumObjectives.unitary_fidelity_loss(Ũ⃗, U_goal(θ)))
        ]
    end

    return NonlinearGlobalKnotPointConstraint(
        terminal_constraint,
        Ũ⃗_name,
        θ_names,
        traj,
        equality=false,
        times=[traj.T]
    )
end

end