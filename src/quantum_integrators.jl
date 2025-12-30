module QuantumIntegrators

using LinearAlgebra
using NamedTrajectories
using DirectTrajOpt
using PiccoloQuantumObjects
using PiccoloQuantumObjects: SamplingTrajectory, EnsembleTrajectory, 
    get_ensemble_state_names, build_sampling_trajectory, build_ensemble_trajectory_from_trajectories
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
    sys = get_system(qtraj)
    traj = get_trajectory(qtraj)
    Ĝ = u_ -> I(sys.levels) ⊗ sys.G(u_, 0.0)
    return BilinearIntegrator(Ĝ, get_state_name(qtraj), get_control_name(qtraj), traj)
end

function BilinearIntegrator(qtraj::KetTrajectory)
    sys = get_system(qtraj)
    traj = get_trajectory(qtraj)
    Ĝ = u_ -> sys.G(u_, 0.0)
    return BilinearIntegrator(Ĝ, get_state_name(qtraj), get_control_name(qtraj), traj)
end

function BilinearIntegrator(qtraj::DensityTrajectory)
    sys = get_system(qtraj)
    traj = get_trajectory(qtraj)
    return BilinearIntegrator(sys.𝒢, get_state_name(qtraj), get_control_name(qtraj), traj)
end

# ----------------------------------------------------------------------------- #
# SamplingTrajectory Integrators
# ----------------------------------------------------------------------------- #

"""
    BilinearIntegrator(qtraj::SamplingTrajectory, traj::NamedTrajectory)

Create a vector of BilinearIntegrators for each system in a SamplingTrajectory.

Each system in the sampling ensemble gets its own dynamics integrator, but they
all share the same control variables. The trajectory `traj` should be the 
expanded trajectory with sample state variables (e.g., from `build_sampling_trajectory`).

# Returns
- `Vector{BilinearIntegrator}`: One integrator per system in the ensemble
"""
function BilinearIntegrator(qtraj::SamplingTrajectory, traj::NamedTrajectory)
    state_names = get_ensemble_state_names(qtraj)
    control_sym = get_control_name(qtraj)
    systems = qtraj.systems
    
    return [
        _sampling_integrator(qtraj.base_trajectory, sys, traj, name, control_sym)
        for (sys, name) in zip(systems, state_names)
    ]
end

# Helper to create single integrator for sampling - dispatches on base trajectory type
function _sampling_integrator(
    base_qtraj::UnitaryTrajectory,
    sys::AbstractQuantumSystem,
    traj::NamedTrajectory,
    state_sym::Symbol,
    control_sym::Symbol
)
    Ĝ = u_ -> I(sys.levels) ⊗ sys.G(u_, 0.0)
    return BilinearIntegrator(Ĝ, state_sym, control_sym, traj)
end

function _sampling_integrator(
    base_qtraj::KetTrajectory,
    sys::AbstractQuantumSystem,
    traj::NamedTrajectory,
    state_sym::Symbol,
    control_sym::Symbol
)
    Ĝ = u_ -> sys.G(u_, 0.0)
    return BilinearIntegrator(Ĝ, state_sym, control_sym, traj)
end

function _sampling_integrator(
    base_qtraj::DensityTrajectory,
    sys::OpenQuantumSystem,
    traj::NamedTrajectory,
    state_sym::Symbol,
    control_sym::Symbol
)
    return BilinearIntegrator(sys.𝒢, state_sym, control_sym, traj)
end

# ----------------------------------------------------------------------------- #
# EnsembleTrajectory Integrators
# ----------------------------------------------------------------------------- #

"""
    BilinearIntegrator(qtraj::EnsembleTrajectory, traj::NamedTrajectory)

Create a vector of BilinearIntegrators for each trajectory in an EnsembleTrajectory.

Each trajectory in the ensemble gets its own dynamics integrator evolving under
the shared system, but with different state variables. The trajectory `traj` should
be the expanded trajectory with ensemble state variables (e.g., from 
`build_ensemble_trajectory_from_trajectories`).

# Returns
- `Vector{BilinearIntegrator}`: One integrator per trajectory in the ensemble
"""
function BilinearIntegrator(qtraj::EnsembleTrajectory, traj::NamedTrajectory)
    state_names = get_ensemble_state_names(qtraj)
    control_sym = get_control_name(qtraj)
    sys = get_system(qtraj)  # Shared system for all
    
    # Use the base trajectory type to determine integrator construction
    base_type = eltype(qtraj.trajectories)
    
    return [
        _ensemble_integrator(base_type, sys, traj, name, control_sym)
        for name in state_names
    ]
end

# Helper to create single integrator for ensemble - dispatches on trajectory type
function _ensemble_integrator(
    ::Type{<:UnitaryTrajectory},
    sys::AbstractQuantumSystem,
    traj::NamedTrajectory,
    state_sym::Symbol,
    control_sym::Symbol
)
    Ĝ = u_ -> I(sys.levels) ⊗ sys.G(u_, 0.0)
    return BilinearIntegrator(Ĝ, state_sym, control_sym, traj)
end

function _ensemble_integrator(
    ::Type{<:KetTrajectory},
    sys::AbstractQuantumSystem,
    traj::NamedTrajectory,
    state_sym::Symbol,
    control_sym::Symbol
)
    Ĝ = u_ -> sys.G(u_, 0.0)
    return BilinearIntegrator(Ĝ, state_sym, control_sym, traj)
end

function _ensemble_integrator(
    ::Type{<:DensityTrajectory},
    sys::OpenQuantumSystem,
    traj::NamedTrajectory,
    state_sym::Symbol,
    control_sym::Symbol
)
    return BilinearIntegrator(sys.𝒢, state_sym, control_sym, traj)
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
    G = u_ -> Isomorphisms.var_G(sys.G(u_), [G(u_) / scale for G in sys.G_vars])
    return BilinearIntegrator(G, var_ψ̃, u, traj)
end

function VariationalUnitaryIntegrator(
    sys::VariationalQuantumSystem,
    traj::NamedTrajectory,
    Ũ⃗::Symbol,
    Ũ⃗_variations::AbstractVector{Symbol},
    u::Symbol;
    scales::AbstractVector{<:Float64}=fill(1.0, length(sys.G_vars)),
)
    var_Ũ⃗ = vcat(Ũ⃗, Ũ⃗_variations...)

    function Ĝ(u_)
        G0 = sys.G(u_)
        Gs = typeof(G0)[I(sys.levels) ⊗ G(u_) / scale for (scale, G) in zip(scales, sys.G_vars)]
        return Isomorphisms.var_G(I(sys.levels) ⊗ G0, Gs)
    end
    return BilinearIntegrator(Ĝ, var_Ũ⃗, u, traj)
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

    test_integrator(integrator, get_trajectory(qtraj); atol=1e-3)
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
    test_integrator(integrator, get_trajectory(qtraj); atol=1e-3)
end

@testitem "BilinearIntegrator dispatch on DensityTrajectory" tags=[:experimental] begin
    using PiccoloQuantumObjects
    using DirectTrajOpt

    sys = OpenQuantumSystem(GATES[:Z], [GATES[:X], GATES[:Y]], 20.0, [1.0, 1.0])
    ρ_init = ComplexF64[1.0 0.0; 0.0 0.0]
    ρ_goal = ComplexF64[0.0 0.0; 0.0 1.0]
    qtraj = DensityTrajectory(sys, ρ_init, ρ_goal, 10)

    integrator = BilinearIntegrator(qtraj)

    @test integrator isa BilinearIntegrator
    test_integrator(integrator, get_trajectory(qtraj); atol=1e-2, show_hessian_diff=true)
end

@testitem "BilinearIntegrator dispatch on SamplingTrajectory (Unitary)" begin
    using PiccoloQuantumObjects
    using DirectTrajOpt

    # Create systems with parameter variation
    sys1 = QuantumSystem(GATES[:Z], [GATES[:X], GATES[:Y]], 1.0, [1.0, 1.0])
    sys2 = QuantumSystem(1.1 * GATES[:Z], [GATES[:X], GATES[:Y]], 1.0, [1.0, 1.0])

    # Create base trajectory and sampling trajectory
    base_qtraj = UnitaryTrajectory(sys1, GATES[:H], 10)
    sampling_qtraj = SamplingTrajectory(base_qtraj, [sys1, sys2])

    # Build expanded trajectory
    base_traj = get_trajectory(base_qtraj)
    expanded_traj, state_names = build_sampling_trajectory(base_traj, :Ũ⃗, 2)

    # Create integrators
    integrators = BilinearIntegrator(sampling_qtraj, expanded_traj)

    @test integrators isa Vector{<:BilinearIntegrator}
    @test length(integrators) == 2

    # Test each integrator
    for integrator in integrators
        test_integrator(integrator, expanded_traj; atol=1e-3)
    end
end

@testitem "BilinearIntegrator dispatch on SamplingTrajectory (Ket)" begin
    using PiccoloQuantumObjects
    using DirectTrajOpt

    # Create systems with parameter variation
    sys1 = QuantumSystem(GATES[:Z], [GATES[:X], GATES[:Y]], 1.0, [1.0, 1.0])
    sys2 = QuantumSystem(1.1 * GATES[:Z], [GATES[:X], GATES[:Y]], 1.0, [1.0, 1.0])

    ψ_init = ComplexF64[1.0, 0.0]
    ψ_goal = ComplexF64[0.0, 1.0]

    # Create base trajectory and sampling trajectory
    base_qtraj = KetTrajectory(sys1, ψ_init, ψ_goal, 10)
    sampling_qtraj = SamplingTrajectory(base_qtraj, [sys1, sys2])

    # Build expanded trajectory
    base_traj = get_trajectory(base_qtraj)
    expanded_traj, state_names = build_sampling_trajectory(base_traj, :ψ̃, 2)

    # Create integrators
    integrators = BilinearIntegrator(sampling_qtraj, expanded_traj)

    @test integrators isa Vector{<:BilinearIntegrator}
    @test length(integrators) == 2

    for integrator in integrators
        test_integrator(integrator, expanded_traj; atol=1e-3)
    end
end

@testitem "BilinearIntegrator dispatch on EnsembleTrajectory (Ket)" begin
    using PiccoloQuantumObjects
    using DirectTrajOpt

    # Shared system
    sys = QuantumSystem(GATES[:Z], [GATES[:X], GATES[:Y]], 1.0, [1.0, 1.0])

    # Different initial/goal states
    ψ0 = ComplexF64[1.0, 0.0]
    ψ1 = ComplexF64[0.0, 1.0]

    qtraj1 = KetTrajectory(sys, ψ0, ψ1, 10)  # |0⟩ → |1⟩
    qtraj2 = KetTrajectory(sys, ψ1, ψ0, 10)  # |1⟩ → |0⟩

    # Create ensemble trajectory
    ensemble_qtraj = EnsembleTrajectory([qtraj1, qtraj2])

    # Build expanded trajectory - pass quantum trajectories, not named trajectories
    expanded_traj, state_names = build_ensemble_trajectory_from_trajectories([qtraj1, qtraj2])

    # Create integrators
    integrators = BilinearIntegrator(ensemble_qtraj, expanded_traj)

    @test integrators isa Vector{<:BilinearIntegrator}
    @test length(integrators) == 2

    for integrator in integrators
        test_integrator(integrator, expanded_traj; atol=1e-3)
    end
end

@testitem "BilinearIntegrator dispatch on EnsembleTrajectory (Unitary)" begin
    using PiccoloQuantumObjects
    using DirectTrajOpt

    # Shared system
    sys = QuantumSystem(GATES[:Z], [GATES[:X], GATES[:Y]], 1.0, [1.0, 1.0])

    # Different target gates
    qtraj1 = UnitaryTrajectory(sys, GATES[:X], 10)
    qtraj2 = UnitaryTrajectory(sys, GATES[:H], 10)

    # Create ensemble trajectory
    ensemble_qtraj = EnsembleTrajectory([qtraj1, qtraj2])

    # Build expanded trajectory - pass quantum trajectories, not named trajectories
    expanded_traj, state_names = build_ensemble_trajectory_from_trajectories([qtraj1, qtraj2])

    # Create integrators
    integrators = BilinearIntegrator(ensemble_qtraj, expanded_traj)

    @test integrators isa Vector{<:BilinearIntegrator}
    @test length(integrators) == 2

    for integrator in integrators
        test_integrator(integrator, expanded_traj; atol=1e-3)
    end
end


end
