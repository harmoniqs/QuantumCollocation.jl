module QuantumObjectives

export KetInfidelityObjective
export UnitaryInfidelityObjective
export DensityMatrixPureStateInfidelityObjective
export UnitarySensitivityObjective

using LinearAlgebra
using NamedTrajectories
using PiccoloQuantumObjects
using DirectTrajOpt
using QuantumCollocation
using TestItems

# using PiccoloQuantumObjects.EmbeddedOperators: get_leakage_indices
# using LinearAlgebra: norm
# using DirectTrajOpt: KnotPointObjective

# --------------------------------------------------------- 
#                        Kets
# ---------------------------------------------------------

function ket_fidelity_loss(
    ψ̃::AbstractVector, 
    ψ_goal::AbstractVector{<:Complex{Float64}}
)
    ψ = iso_to_ket(ψ̃)
    return abs2(ψ_goal' * ψ)
end 

function KetInfidelityObjective(
    ψ̃_name::Symbol,
    traj::NamedTrajectory;
    Q=100.0
)
    ψ_goal = iso_to_ket(traj.goal[ψ̃_name])
    ℓ = ψ̃ -> abs(1 - ket_fidelity_loss(ψ̃, ψ_goal))
    return TerminalObjective(ℓ, ψ̃_name, traj; Q=Q)
end


# ---------------------------------------------------------
#                        Unitaries
# ---------------------------------------------------------

function unitary_fidelity_loss(
    Ũ⃗::AbstractVector{<:Real},
    U_goal::AbstractMatrix{<:Complex{Float64}}
)
    U = iso_vec_to_operator(Ũ⃗)
    n = size(U, 1)
    return abs2(tr(U_goal' * U)) / n^2
end

function unitary_fidelity_loss(
    Ũ⃗::AbstractVector{<:Real},
    op::EmbeddedOperator
)
    U_goal = unembed(op)
    U = iso_vec_to_operator(Ũ⃗)[op.subspace, op.subspace]
    n = length(op.subspace)
    M = U_goal'U
    return 1 / (n * (n + 1)) * (abs(tr(M'M)) + abs2(tr(M))) 
end

function UnitaryInfidelityObjective(
    U_goal::AbstractPiccoloOperator,
    Ũ⃗_name::Symbol,
    traj::NamedTrajectory;
    Q=100.0
)
    ℓ = Ũ⃗ -> abs(1 - unitary_fidelity_loss(Ũ⃗, U_goal))
    return TerminalObjective(ℓ, Ũ⃗_name, traj; Q=Q)
end

# ---------------------------------------------------------
#                Leakage Suppression Objective
# ---------------------------------------------------------

"""
    LeakageSuppressionObjective(system; norm_type=:fro, kwargs...)

Construct a `KnotPointObjective` that penalizes leakage outside the computational subspace.

- `system`: The quantum system (must support `get_leakage_indices`).
- `norm_type`: The matrix norm to use (default: `:fro` for Frobenius norm).
- `kwargs...`: Passed to `KnotPointObjective`.

The objective is: 
    sum_{(i,j) ∈ I_leakage} norm(U[i, j], norm_type)
where `I_leakage` is given by `get_leakage_indices(system)`.
"""
function LeakageSuppressionObjective(system, name::Symbol, traj; norm_type=:fro, kwargs...)
    leakage_inds = get_leakage_indices(system)
    return KnotPointObjective(
        (U, args...) -> begin
            n = Int(sqrt(length(U)))
            @assert n^2 == length(U) "Input vector length is not a perfect square"
            Umat = reshape(U, n, n)
            sum(abs(Umat[i, j]) for (i, j) in leakage_inds)
        end,
        name,
        traj;
        kwargs...
    )
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

function DensityMatrixPureStateInfidelityObjective(
    ρ̃_name::Symbol,
    ψ_goal::AbstractVector{<:Complex{Float64}},
    traj::NamedTrajectory;
    Q=100.0
)
    ℓ = ρ̃ -> density_matrix_pure_state_infidelity_loss(ρ̃, ψ_goal)
    return TerminalObjective(ℓ, ρ̃_name, traj; Q=Q)
end

# ---------------------------------------------------------
#                        Sensitivity
# ---------------------------------------------------------

function unitary_fidelity_loss(
    Ũ⃗::AbstractVector{<:Real}
)
    U = iso_vec_to_operator(Ũ⃗)
    n = size(U, 1)
    return abs2(tr(U' * U)) / n^2
end

function UnitarySensitivityObjective(
    name::Symbol,
    traj::NamedTrajectory,
    times::AbstractVector{Int};
    Qs::AbstractVector{<:Float64}=fill(1.0, length(times)),
    scale::Float64=1.0,
)
    ℓ = Ũ⃗ -> scale^4 * unitary_fidelity_loss(Ũ⃗)

    return KnotPointObjective(
        ℓ,
        name,
        traj;
        Qs=Qs,
        times=times
    )
end

@testitem "LeakageSuppressionObjective basic functionality" begin
    using PiccoloQuantumObjects
    using LinearAlgebra: norm
    using NamedTrajectories

    # Dummy system with leakage indices
    struct DummySystem end
    PiccoloQuantumObjects.EmbeddedOperators.get_leakage_indices(::DummySystem) = [(1,2), (2,3)]
    
    # U with zeros everywhere (no leakage)
    U = zeros(3,3)
    traj = NamedTrajectory((; u = reshape(U, 9, 1)); controls=:u, timestep=1.0)
    obj = QuantumObjectives.LeakageSuppressionObjective(DummySystem(), :u, traj)
    @test obj.L(traj.datavec) == 0.0

    # U with leakage at (1,2) and (2,3)
    U[1,2] = 2.0
    U[2,3] = 3.0
    traj = NamedTrajectory((; u = reshape(U, 9, 1)); controls=:u, timestep=1.0)
    @test obj.L(traj.datavec) ≈ 2.0 + 3.0
end

end