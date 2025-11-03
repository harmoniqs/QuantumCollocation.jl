module QuantumIntegrators

export KetIntegrator
export UnitaryIntegrator
export DensityMatrixIntegrator
export VariationalUnitaryIntegrator

using LinearAlgebra
using NamedTrajectories
using DirectTrajOpt
using PiccoloQuantumObjects
using SparseArrays

const ⊗ = kron

# ----------------------------------------------------------------------------- #
# Default Integrators
# ----------------------------------------------------------------------------- #

function KetIntegrator(
    sys::QuantumSystem,
    traj::NamedTrajectory,
    ψ̃::Symbol,
    u::Symbol
)
    Ĝ = u_ -> sys.G(u_, 0.0)
    return BilinearIntegrator(Ĝ, traj, ψ̃, u)
end

function UnitaryIntegrator(
    sys::QuantumSystem,
    traj::NamedTrajectory,
    Ũ⃗::Symbol,
    u::Symbol
)
    Ĝ = u_ -> I(sys.levels) ⊗ sys.G(u_, 0.0)
    return BilinearIntegrator(Ĝ, traj, Ũ⃗, u)
end

function DensityMatrixIntegrator(
    sys::OpenQuantumSystem,
    traj::NamedTrajectory,
    ρ̃::Symbol,
    u::Symbol
)
    return BilinearIntegrator(sys.𝒢, traj, ρ̃, u)
end

# ----------------------------------------------------------------------------- #
# Variational Integrators
# ----------------------------------------------------------------------------- #

function VariationalKetIntegrator(
    sys::VariationalQuantumSystem,
    traj::NamedTrajectory,
    ψ̃::Symbol,
    ψ̃_variations::AbstractVector{Symbol},
    u::Symbol;
    scale::Float64=1.0,
)
    var_ψ̃ = vcat(ψ̃, ψ̃_variations...)
    G = a -> Isomorphisms.var_G(sys.G(a), [G(a) / scale for G in sys.G_vars])
    return BilinearIntegrator(G, traj, var_ψ̃, u)
end

function VariationalUnitaryIntegrator(
    sys::VariationalQuantumSystem,
    traj::NamedTrajectory,
    Ũ⃗::Symbol,
    Ũ⃗_variations::AbstractVector{Symbol},
    u::Symbol;
    scales::AbstractVector{<:Float64}=fill(1.0, length(sys.G_vars)),
)
    var_Ũ⃗ = vcat(Ũ⃗, Ũ⃗_variations...)

    function Ĝ(u)
        G0 = sys.G(u)
        Gs = typeof(G0)[I(sys.levels) ⊗ G(u) / scale for (scale, G) in zip(scales, sys.G_vars)]
        return Isomorphisms.var_G(I(sys.levels) ⊗ G0, Gs)
    end
    return BilinearIntegrator(Ĝ, traj, var_Ũ⃗, u)
end


end
