module QuantumObjectives

export KetInfidelityObjective
export KetsCoherentInfidelityObjective
export UnitaryInfidelityObjective
export DensityMatrixPureStateInfidelityObjective
export UnitarySensitivityObjective
export UnitaryFreePhaseInfidelityObjective
export LeakageObjective

using LinearAlgebra
using NamedTrajectories
using PiccoloQuantumObjects
using DirectTrajOpt
using TestItems

# --------------------------------------------------------- 
#                       Kets
# ---------------------------------------------------------

function ket_fidelity_loss(
    ψ̃::AbstractVector, 
    ψ_goal::AbstractVector{<:Complex{Float64}}
)
    ψ = iso_to_ket(ψ̃)
    return abs2(ψ_goal' * ψ)
end 

"""
    KetInfidelityObjective(ψ̃_name, traj; Q=100.0)

Create a terminal objective for ket state infidelity, using the goal from `traj.goal[ψ̃_name]`.
"""
function KetInfidelityObjective(
    ψ̃_name::Symbol,
    traj::NamedTrajectory;
    Q=100.0
)
    ψ_goal = iso_to_ket(traj.goal[ψ̃_name])
    ℓ = ψ̃ -> abs(1 - ket_fidelity_loss(ψ̃, ψ_goal))
    return TerminalObjective(ℓ, ψ̃_name, traj; Q=Q)
end

"""
    KetInfidelityObjective(ψ_goal, ψ̃_name, traj; Q=100.0)

Create a terminal objective for ket state infidelity with an explicit goal state.

This variant is useful for SamplingProblem and EnsembleTrajectory where the goal
is shared across multiple state variables that don't have individual goals in `traj.goal`.

# Arguments
- `ψ_goal::AbstractVector{<:Complex}`: The target ket state (complex vector)
- `ψ̃_name::Symbol`: Name of the isomorphic state variable in the trajectory
- `traj::NamedTrajectory`: The trajectory

# Keyword Arguments
- `Q::Float64=100.0`: Weight on the infidelity objective
"""
function KetInfidelityObjective(
    ψ_goal::AbstractVector{<:Complex},
    ψ̃_name::Symbol,
    traj::NamedTrajectory;
    Q=100.0
)
    ℓ = ψ̃ -> abs(1 - ket_fidelity_loss(ψ̃, ComplexF64.(ψ_goal)))
    return TerminalObjective(ℓ, ψ̃_name, traj; Q=Q)
end

function _partitions(dims::AbstractVector{Int})
    ends = cumsum(dims)
    return UnitRange.(ends .- dims .+ 1, ends)
end

"""
    KetsCoherentInfidelityObjective(ψ_goals, ψ̃_names, traj; Q=100.0)

Phase-sensitive ensemble fidelity objective for a goal unitary via state preparations.

For an ensemble of N prepared states with final states |ϕᵢ⟩ and goals |gᵢ⟩, computes the 
coherent overlap ℱ = |∑ᵢ ⟨ϕᵢ | gᵢ⟩|² / N². If the prepared states form a complete  basis,
ℱ reduces to |Tr(U† U_target)|² / N².

# Arguments
- `ψ_goals::AbstractVector{<:AbstractVector{<:Complex}}`: The target ket states (complex vector)
- `ψ̃_names::Symbol`: Names of the isomorphic state variables in the trajectory
- `traj::NamedTrajectory`: The trajectory

# Keyword Arguments
- `Q::Float64=100.0`: Weight on the infidelity objective
"""
function KetsCoherentInfidelityObjective(
    ψ_goals::AbstractVector{<:AbstractVector{<:Complex}},
    ψ̃_names::AbstractVector{Symbol}, 
    traj::NamedTrajectory;
    Q=100.0
)
    N = length(ψ̃_names)
    @assert length(ψ_goals) == N
    idxs = _partitions([traj.dims[n] for n in ψ̃_names])
    function ℓ(x)
        ℱ = abs2(sum(ket_fidelity_loss(x[idx], ComplexF64.(g)) for (idx, g) in zip(idxs, ψ_goals))) / N^2
        return abs(1 - ℱ)
    end
    return KnotPointObjective(ℓ, ψ̃_names, traj, times=[traj.N], Qs=[Q])
end


# ---------------------------------------------------------
#                       Unitaries
# ---------------------------------------------------------

function unitary_fidelity_loss(
    Ũ⃗::AbstractVector{<:Real},
    U_goal::AbstractMatrix{<:Complex{<:Real}}
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

function UnitaryFreePhaseInfidelityObjective(
    U_goal::Function,
    Ũ⃗_name::Symbol,
    θ_names::AbstractVector{Symbol},
    traj::NamedTrajectory;
    Q=100.0
)
    d = sum(traj.global_dims[n] for n in θ_names)
    function ℓ(z)
        Ũ⃗, θ = z[1:end-d], z[end-d+1:end]
        return abs(1 - QuantumObjectives.unitary_fidelity_loss(Ũ⃗, U_goal(θ)))
    end
    return TerminalObjective(ℓ, Ũ⃗_name, θ_names, traj; Q=Q)
end

function UnitaryFreePhaseInfidelityObjective(
    U_goal::Function,
    Ũ⃗_name::Symbol,
    θ_name::Symbol,
    traj::NamedTrajectory;
    kwargs...
)
    return UnitaryFreePhaseInfidelityObjective(U_goal, Ũ⃗_name, [θ_name], traj; kwargs...)
end

# ---------------------------------------------------------
#                       Density Matrices
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
#                       Sensitivity
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

# ---------------------------------------------------------
#                       Leakage
# ---------------------------------------------------------

"""
    LeakageObjective(indices, name, traj::NamedTrajectory)

Construct a `KnotPointObjective` that penalizes leakage of `name` at the knot points specified by `times` at any `indices` that are outside the computational subspace.

"""
function LeakageObjective(
    indices::AbstractVector{Int},
    name::Symbol,
    traj::NamedTrajectory;
    times=1:traj.N,
    Qs::AbstractVector{<:Float64}=fill(1.0, length(times)),
)
    leakage_objective(x) = sum(abs2, x[indices]) / length(indices)

    return KnotPointObjective(
        leakage_objective,
        name,
        traj;
        Qs=Qs,
        times=times,
    )
end

# ---------------------------------------------------------
#                       Tests
# ---------------------------------------------------------

using TestItems

end