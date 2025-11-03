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
using TestItems

import DirectTrajOpt: BilinearIntegrator

# Import QuantumTrajectories types (will be loaded before this module)
using ..QuantumTrajectories

const ⊗ = kron

# ----------------------------------------------------------------------------- #
# Default Integrators
# ----------------------------------------------------------------------------- #

# Dispatch on quantum trajectory types
function BilinearIntegrator(qtraj::UnitaryTrajectory)
    sys = system(qtraj)
    traj = trajectory(qtraj)
    Ĝ = u_ -> I(sys.levels) ⊗ sys.G(u_, 0.0)
    return BilinearIntegrator(Ĝ, state_name(qtraj), control_name(qtraj))
end

function BilinearIntegrator(qtraj::KetTrajectory)
    sys = system(qtraj)
    traj = trajectory(qtraj)
    Ĝ = u_ -> sys.G(u_, 0.0)
    
    # If only one state, return single integrator
    if length(qtraj.state_names) == 1
        return BilinearIntegrator(Ĝ, qtraj.state_names[1], control_name(qtraj))
    end
    
    # Multiple states: return vector of integrators, one for each state
    return [BilinearIntegrator(Ĝ, name, control_name(qtraj)) for name in qtraj.state_names]
end

function BilinearIntegrator(qtraj::DensityTrajectory)
    sys = system(qtraj)
    traj = trajectory(qtraj)
    return BilinearIntegrator(sys.𝒢, state_name(qtraj), control_name(qtraj))
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
    return BilinearIntegrator(Ĝ, traj, var_Ũ⃗, a)
end

# ----------------------------------------------------------------------------- #
# Tests
# ----------------------------------------------------------------------------- #

@testitem "BilinearIntegrator dispatch on UnitaryTrajectory" begin
    using PiccoloQuantumObjects
    using DirectTrajOpt
    
    sys = QuantumSystem(GATES[:Z], [GATES[:X], GATES[:Y]], 1.0, [1.0, 1.0])
    qtraj = UnitaryTrajectory(sys, GATES[:H], 10)
    
    integrator = BilinearIntegrator(qtraj)
    
    @test integrator isa BilinearIntegrator
end

@testitem "BilinearIntegrator dispatch on KetTrajectory" begin
    using PiccoloQuantumObjects
    using DirectTrajOpt
    
    sys = QuantumSystem(GATES[:Z], [GATES[:X], GATES[:Y]], 1.0, [1.0, 1.0])
    ψ_init = ComplexF64[1.0, 0.0]
    ψ_goal = ComplexF64[0.0, 1.0]
    qtraj = KetTrajectory(sys, ψ_init, ψ_goal, 10)
    
    integrator = BilinearIntegrator(qtraj)
    
    @test integrator isa BilinearIntegrator
end

@testitem "BilinearIntegrator dispatch on KetTrajectory (multiple states)" begin
    using PiccoloQuantumObjects
    using DirectTrajOpt
    
    sys = QuantumSystem(GATES[:Z], [GATES[:X], GATES[:Y]], 1.0, [1.0, 1.0])
    
    # Multiple initial and goal states
    ψ_inits = [
        ComplexF64[1.0, 0.0],
        ComplexF64[0.0, 1.0]
    ]
    ψ_goals = [
        ComplexF64[0.0, 1.0],
        ComplexF64[1.0, 0.0]
    ]
    
    qtraj = KetTrajectory(sys, ψ_inits, ψ_goals, 10)
    
    integrators = BilinearIntegrator(qtraj)
    
    @test integrators isa Vector{<:BilinearIntegrator}
    @test length(integrators) == 2
end

@testitem "BilinearIntegrator dispatch on DensityTrajectory" begin
    using PiccoloQuantumObjects
    using DirectTrajOpt
    
    sys = OpenQuantumSystem(GATES[:Z], [GATES[:X], GATES[:Y]], 1.0, [1.0, 1.0])
    ρ_init = ComplexF64[1.0 0.0; 0.0 0.0]
    ρ_goal = ComplexF64[0.0 0.0; 0.0 1.0]
    qtraj = DensityTrajectory(sys, ρ_init, ρ_goal, 10)
    
    integrator = BilinearIntegrator(qtraj)
    
    @test integrator isa BilinearIntegrator
end


end