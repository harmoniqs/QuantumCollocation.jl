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
    return BilinearIntegrator(u_ -> Isomorphisms.G(sys.H(u_, 0.0)), traj, ψ̃, u)
end

function UnitaryIntegrator(
    sys::QuantumSystem,
    traj::NamedTrajectory, 
    Ũ⃗::Symbol, 
    u::Symbol
) 
    Id = I(sys.levels)
    Ĝ = u_ -> Id ⊗ Isomorphisms.G(sys.H(u_, 0.0))
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
    a::Symbol;
    scale::Float64=1.0,
) 
    var_ψ̃ = vcat(ψ̃, ψ̃_variations...)
    G = a -> Isomorphisms.var_G(sys.G(a), [G(a) / scale for G in sys.G_vars])
    return BilinearIntegrator(G, traj, var_ψ̃, a)
end

function VariationalUnitaryIntegrator(
    sys::VariationalQuantumSystem,
    traj::NamedTrajectory, 
    Ũ⃗::Symbol, 
    Ũ⃗_variations::AbstractVector{Symbol},
    a::Symbol;
    scales::AbstractVector{<:Float64}=fill(1.0, length(sys.G_vars)),
)
    var_Ũ⃗ = vcat(Ũ⃗, Ũ⃗_variations...)

    function Ĝ(a)
        G0 = sys.G(a)
        Gs = typeof(G0)[I(sys.levels) ⊗ G(a) / scale for (scale, G) in zip(scales, sys.G_vars)]
        return Isomorphisms.var_G(I(sys.levels) ⊗ G0, Gs)
    end
    return BilinearIntegrator(Ĝ, traj, var_Ũ⃗, a)
end


end