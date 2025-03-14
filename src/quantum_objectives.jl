module QuantumObjectives

export ket_infidelity_loss
export unitary_infidelity_loss
export density_matrix_pure_state_infidelity_loss
export KetInfidelityLoss
export UnitaryInfidelityLoss
export DensityMatrixPureStateInfidelityLoss

using LinearAlgebra
using NamedTrajectories
using PiccoloQuantumObjects
using DirectTrajOpt

# --------------------------------------------------------- 
#                        Kets
# ---------------------------------------------------------

function ket_infidelity_loss(
    ψ̃::AbstractVector, 
    ψ_goal::AbstractVector{<:Complex{Float64}}
)
    ψ = iso_to_ket(ψ̃)
    ℱ = abs2(ψ_goal' * ψ)
    return abs(1 - ℱ) 
end 

function KetInfidelityLoss(
    ψ̃_name::Symbol,
    traj::NamedTrajectory;
    Q=100.0
)
    ψ_goal = iso_to_ket(traj.goal[ψ̃_name])
    ℓ = ψ̃ -> ket_infidelity_loss(ψ̃, ψ_goal)
    return TerminalLoss(ℓ, ψ̃_name, traj; Q=Q)
end


# ---------------------------------------------------------
#                        Unitaries
# ---------------------------------------------------------

function unitary_infidelity_loss(
    Ũ⃗::AbstractVector,
    U_goal::AbstractMatrix{<:Complex{Float64}}
)
    U = iso_vec_to_operator(Ũ⃗)
    n = size(U, 1)
    ℱ = abs2(tr(U_goal' * U)) / n^2
    return abs(1 - ℱ) 
end

function unitary_infidelity_loss(
    Ũ⃗::AbstractVector,
    op::EmbeddedOperator
)
    U_goal = unembed(op)
    U = iso_vec_to_operator(Ũ⃗)[op.subspace, op.subspace]
    n = length(op.subspace)
    M = U_goal'U
    ℱ = 1 / (n * (n + 1)) * (abs(tr(M'M)) + abs2(tr(M))) 
    return abs(1 - ℱ)
end

function UnitaryInfidelityLoss(
    U_goal::AbstractPiccoloOperator,
    Ũ⃗_name::Symbol,
    traj::NamedTrajectory;
    Q=100.0
)
    ℓ = Ũ⃗ -> unitary_infidelity_loss(Ũ⃗, U_goal)
    return TerminalLoss(ℓ, Ũ⃗_name, traj; Q=Q)
end

# ---------------------------------------------------------
#                        Density Matrices
# ---------------------------------------------------------

function density_matrix_pure_state_infidelity_loss(
    ρ̃::AbstractVector, 
    ψ::AbstractVector{<:Complex{Float64}}
)
    ρ = iso_vec_to_density(ρ̃)
    ℱ = real(ψ' * ρ * ψ)
    return abs(1 - ℱ)
end

function DensityMatrixPureStateInfidelityLoss(
    ρ̃_name::Symbol,
    ψ_goal::AbstractVector{<:Complex{Float64}},
    traj::NamedTrajectory;
    Q=100.0
)
    ℓ = ρ̃ -> density_matrix_pure_state_infidelity_loss(ρ̃, ψ_goal)
    return TerminalLoss(ℓ, ρ̃_name, traj; Q=Q)
end

end