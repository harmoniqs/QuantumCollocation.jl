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

end