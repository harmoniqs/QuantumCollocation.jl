module QuantumTrajectories

export AbstractQuantumTrajectory
export UnitaryTrajectory
export KetTrajectory
export DensityTrajectory
export trajectory, system, goal, state_name, control_name, state, controls

using NamedTrajectories
using PiccoloQuantumObjects
using TestItems
using LinearAlgebra

# Import helper functions from TrajectoryInitialization
import ..TrajectoryInitialization: unitary_geodesic, unitary_linear_interpolation, linear_interpolation

"""
    AbstractQuantumTrajectory

Abstract type for quantum trajectories that wrap a `NamedTrajectory` with quantum-specific metadata.

Subtypes should implement:
- `trajectory(qtraj)`: Return the underlying `NamedTrajectory`
- `system(qtraj)`: Return the quantum system
- `state_name(qtraj)`: Return the state variable name
- `control_name(qtraj)`: Return the control variable name
- `goal(qtraj)`: Return the goal state/operator
"""
abstract type AbstractQuantumTrajectory end

# Accessor functions
trajectory(qtraj::AbstractQuantumTrajectory) = qtraj.trajectory
system(qtraj::AbstractQuantumTrajectory) = qtraj.system
state_name(qtraj::AbstractQuantumTrajectory) = qtraj.state_name
control_name(qtraj::AbstractQuantumTrajectory) = qtraj.control_name
goal(qtraj::AbstractQuantumTrajectory) = qtraj.goal

# Delegate common operations to underlying NamedTrajectory
Base.getindex(qtraj::AbstractQuantumTrajectory, key) = getindex(trajectory(qtraj), key)
Base.setindex!(qtraj::AbstractQuantumTrajectory, value, key) = setindex!(trajectory(qtraj), value, key)

# Delegate property access to underlying NamedTrajectory, except for AbstractQuantumTrajectory's own fields
function Base.getproperty(qtraj::AbstractQuantumTrajectory, symb::Symbol)
    # Access AbstractQuantumTrajectory's own fields
    if symb ∈ fieldnames(typeof(qtraj))
        return getfield(qtraj, symb)
    # Otherwise delegate to the underlying NamedTrajectory
    else
        return getproperty(getfield(qtraj, :trajectory), symb)
    end
end

function Base.propertynames(qtraj::AbstractQuantumTrajectory)
    # Combine properties from both the wrapper and the trajectory
    wrapper_fields = fieldnames(typeof(qtraj))
    traj_props = propertynames(getfield(qtraj, :trajectory))
    return tuple(wrapper_fields..., traj_props...)
end

# Convenience accessors
state(qtraj::AbstractQuantumTrajectory) = trajectory(qtraj)[state_name(qtraj)]
controls(qtraj::AbstractQuantumTrajectory) = trajectory(qtraj)[control_name(qtraj)]

"""
    UnitaryTrajectory <: AbstractQuantumTrajectory

A trajectory for unitary gate synthesis problems.

# Fields
- `trajectory::NamedTrajectory`: The underlying trajectory data (stored as a copy)
- `system::QuantumSystem`: The quantum system
- `state_name::Symbol`: Name of the state variable (typically `:Ũ⃗`)
- `control_name::Symbol`: Name of the control variable (typically `:u`)
- `goal::AbstractPiccoloOperator`: Target unitary operator
"""
struct UnitaryTrajectory <: AbstractQuantumTrajectory
    trajectory::NamedTrajectory
    system::QuantumSystem
    state_name::Symbol
    control_name::Symbol
    goal::AbstractPiccoloOperator
    
    function UnitaryTrajectory(
        sys::QuantumSystem,
        U_goal::AbstractMatrix{<:Number},
        N::Int;
        U_init::AbstractMatrix{<:Number}=Matrix{ComplexF64}(I(size(sys.H_drift, 1))),
        Δt_min::Float64=sys.T_max / (2 * (N-1)),
        Δt_max::Float64=2 * sys.T_max / (N-1),
        free_time::Bool=true,
        geodesic::Bool=true
    )
        Δt = sys.T_max / (N - 1)
        n_drives = sys.n_drives
        
        # Initialize unitary trajectory
        if geodesic
            H_drift = Matrix(get_drift(sys))
            Ũ⃗ = unitary_geodesic(U_init, U_goal, N, H_drift=H_drift)
        else
            Ũ⃗ = unitary_linear_interpolation(U_init, U_goal, N)
        end
        
        # Initialize controls (zero at boundaries)
        u = hcat(
            zeros(n_drives),
            randn(n_drives, N - 2) * 0.01,
            zeros(n_drives)
        )
        
        # Timesteps
        Δt_vec = fill(Δt, N)
        
        # Initial and final constraints
        Ũ⃗_init = operator_to_iso_vec(U_init)
        Ũ⃗_goal = operator_to_iso_vec(U_goal)
        
        initial = (Ũ⃗ = Ũ⃗_init, u = zeros(n_drives))
        final = (u = zeros(n_drives),)
        goal_constraint = (Ũ⃗ = Ũ⃗_goal,)
        
        # Time data (automatic for time-dependent systems)
        if sys.time_dependent
            t_data = cumsum([0.0; Δt_vec[1:end-1]])
            initial = merge(initial, (t = [0.0],))
        end
        
        # Bounds - convert drive_bounds from Vector{Tuple} to Tuple of Vectors
        u_lower = [sys.drive_bounds[i][1] for i in 1:n_drives]
        u_upper = [sys.drive_bounds[i][2] for i in 1:n_drives]
        Δt_bounds = free_time ? (Δt_min, Δt_max) : (Δt, Δt)
        bounds = (
            u = (u_lower, u_upper),
            Δt = Δt_bounds
        )
        
        # Build component data
        comps_data = (Ũ⃗ = Ũ⃗, u = u, Δt = reshape(Δt_vec, 1, N))
        
        if sys.time_dependent
            comps_data = merge(comps_data, (t = reshape(t_data, 1, N),))
        end
        
        traj = NamedTrajectory(
            comps_data;
            controls = (:u, :Δt),
            timestep = :Δt,
            initial = initial,
            final = final,
            goal = goal_constraint,
            bounds = bounds
        )
        
        return new(traj, sys, :Ũ⃗, :u, U_goal)
    end
end

"""
    KetTrajectory <: AbstractQuantumTrajectory

A trajectory for quantum state transfer problems.

# Fields
- `trajectory::NamedTrajectory`: The underlying trajectory data (stored as a copy)
- `system::QuantumSystem`: The quantum system
- `state_name::Symbol`: Name of the primary state variable (typically `:ψ̃` or `:ψ̃1`)
- `state_names::Vector{Symbol}`: Names of all state variables (e.g., `[:ψ̃1, :ψ̃2]` for multiple states)
- `control_name::Symbol`: Name of the control variable (typically `:u`)
- `goals::Vector{<:AbstractVector{ComplexF64}}`: Target ket states (can be multiple)
"""
struct KetTrajectory <: AbstractQuantumTrajectory
    trajectory::NamedTrajectory
    system::QuantumSystem
    state_name::Symbol
    state_names::Vector{Symbol}
    control_name::Symbol
    goals::Vector{<:AbstractVector{ComplexF64}}
    
    function KetTrajectory(
        sys::QuantumSystem,
        ψ_init::AbstractVector{ComplexF64},
        ψ_goal::AbstractVector{ComplexF64},
        N::Int;
        kwargs...
    )
        # Delegate to multi-state constructor
        return KetTrajectory(sys, [ψ_init], [ψ_goal], N; kwargs...)
    end
    
    # High-level constructor: multiple states
    function KetTrajectory(
        sys::QuantumSystem,
        ψ_inits::AbstractVector{<:AbstractVector{ComplexF64}},
        ψ_goals::AbstractVector{<:AbstractVector{ComplexF64}},
        N::Int;
        state_name::Symbol=:ψ̃,
        state_names::Union{AbstractVector{<:Symbol}, Nothing}=nothing,
        Δt_min::Float64=sys.T_max / (2 * (N-1)),
        Δt_max::Float64=2 * sys.T_max / (N-1),
        free_time::Bool=true
    )
        @assert length(ψ_inits) == length(ψ_goals) "ψ_inits and ψ_goals must have the same length"
        
        Δt = sys.T_max / (N - 1)
        n_drives = sys.n_drives
        n_states = length(ψ_inits)
        
        # Generate state names if not provided
        if isnothing(state_names)
            if n_states == 1
                state_names = [state_name]
            else
                state_names = [Symbol(string(state_name) * "$i") for i = 1:n_states]
            end
        else
            @assert length(state_names) == n_states "state_names must have same length as ψ_inits"
        end
        
        # Convert to iso representation
        ψ̃_inits = ket_to_iso.(ψ_inits)
        ψ̃_goals = ket_to_iso.(ψ_goals)
        
        # Linear interpolation of states
        ψ̃_trajs = [linear_interpolation(ψ̃_init, ψ̃_goal, N) for (ψ̃_init, ψ̃_goal) in zip(ψ̃_inits, ψ̃_goals)]
        
        # Initialize controls (zero at boundaries)
        u = hcat(
            zeros(n_drives),
            randn(n_drives, N - 2) * 0.01,
            zeros(n_drives)
        )
        
        # Timesteps
        Δt_vec = fill(Δt, N)
        
        # Initial and final constraints
        initial_states = NamedTuple{Tuple(state_names)}(Tuple(ψ̃_inits))
        goal_states = NamedTuple{Tuple(state_names)}(Tuple(ψ̃_goals))
        
        initial = merge(initial_states, (u = zeros(n_drives),))
        final = (u = zeros(n_drives),)
        goal_constraint = goal_states
        
        # Time data (automatic for time-dependent systems)
        if sys.time_dependent
            t_data = [0.0; cumsum(Δt_vec)[1:end-1]]
            initial = merge(initial, (t = [0.0],))
        end
        
        # Bounds - convert drive_bounds from Vector{Tuple} to Tuple of Vectors
        u_lower = [sys.drive_bounds[i][1] for i in 1:n_drives]
        u_upper = [sys.drive_bounds[i][2] for i in 1:n_drives]
        Δt_bounds = free_time ? (Δt_min, Δt_max) : (Δt, Δt)
        bounds = (
            u = (u_lower, u_upper),
            Δt = Δt_bounds
        )
        
        # Build component data
        state_data = NamedTuple{Tuple(state_names)}(Tuple(ψ̃_trajs))
        comps_data = merge(state_data, (u = u, Δt = reshape(Δt_vec, 1, N)))
        
        if sys.time_dependent
            comps_data = merge(comps_data, (t = reshape(t_data, 1, N),))
        end
        
        traj = NamedTrajectory(
            comps_data;
            controls = (:u, :Δt),
            timestep = :Δt,
            initial = initial,
            final = final,
            goal = goal_constraint,
            bounds = bounds
        )
        
        # Store all state names and use first as primary
        return new(traj, sys, state_names[1], collect(state_names), :u, ψ_goals)
    end
end

# Special accessor for KetTrajectory
goal(qtraj::KetTrajectory) = length(qtraj.goals) == 1 ? qtraj.goals[1] : qtraj.goals

"""
    DensityTrajectory <: AbstractQuantumTrajectory

A trajectory for open quantum system problems.

# Fields
- `trajectory::NamedTrajectory`: The underlying trajectory data (stored as a copy)
- `system::OpenQuantumSystem`: The open quantum system
- `state_name::Symbol`: Name of the state variable (typically `:ρ⃗̃`)
- `control_name::Symbol`: Name of the control variable (typically `:u`)
- `goal::AbstractMatrix`: Target density matrix
"""
struct DensityTrajectory <: AbstractQuantumTrajectory
    trajectory::NamedTrajectory
    system::OpenQuantumSystem
    state_name::Symbol
    control_name::Symbol
    goal::AbstractMatrix
    
    function DensityTrajectory(
        sys::OpenQuantumSystem,
        ρ_init::AbstractMatrix,
        ρ_goal::AbstractMatrix,
        N::Int;
        Δt_min::Float64=sys.T_max / (2 * (N-1)),
        Δt_max::Float64=2 * sys.T_max / (N-1),
        free_time::Bool=true
    )
        Δt = sys.T_max / (N - 1)
        n_drives = sys.n_drives
        
        # Convert to iso representation
        ρ⃗̃_init = density_to_iso_vec(ρ_init)
        ρ⃗̃_goal = density_to_iso_vec(ρ_goal)
        
        # Linear interpolation of state
        ρ⃗̃ = linear_interpolation(ρ⃗̃_init, ρ⃗̃_goal, N)
        
        # Initialize controls (zero at boundaries)
        u = hcat(
            zeros(n_drives),
            randn(n_drives, N - 2) * 0.01,
            zeros(n_drives)
        )
        
        # Timesteps
        Δt_vec = fill(Δt, N)
        
        # Initial and final constraints
        initial = (ρ⃗̃ = ρ⃗̃_init, u = zeros(n_drives))
        final = (u = zeros(n_drives),)
        goal_constraint = (ρ⃗̃ = ρ⃗̃_goal,)
        
        # Time data (automatic for time-dependent systems)
        if sys.time_dependent
            t_data = [0.0; cumsum(Δt_vec)[1:end-1]]
            initial = merge(initial, (t = [0.0],))
        end
        
        # Bounds - convert drive_bounds from Vector{Tuple} to Tuple of Vectors
        u_lower = [sys.drive_bounds[i][1] for i in 1:n_drives]
        u_upper = [sys.drive_bounds[i][2] for i in 1:n_drives]
        Δt_bounds = free_time ? (Δt_min, Δt_max) : (Δt, Δt)
        bounds = (
            u = (u_lower, u_upper),
            Δt = Δt_bounds
        )
        
        # Build component data
        comps_data = (Ũ⃗ = Ũ⃗, u = u, Δt = reshape(Δt_vec, 1, N))
        
        if sys.time_dependent
            comps_data = merge(comps_data, (t = reshape(t_data, 1, N),))
        end
        
        traj = NamedTrajectory(
            comps_data;
            controls = (:u, :Δt),
            timestep = :Δt,
            initial = initial,
            final = final,
            goal = goal_constraint,
            bounds = bounds
        )
        
        return new(traj, sys, :ρ⃗̃, :u, ρ_goal)
    end
end

# ============================================================================= #
# Tests
# ============================================================================= #


@testitem "UnitaryTrajectory high-level constructor" begin
    using PiccoloQuantumObjects
    using NamedTrajectories
    
    # Create a simple quantum system
    sys = QuantumSystem(
        GATES[:Z],              # H_drift
        [GATES[:X], GATES[:Y]], # H_drives
        1.0,                    # T_max
        [1.0, 1.0];             # drive_bounds
        time_dependent=false
    )
    
    N = 10
    U_goal = GATES[:H]
    
    # Test basic constructor
    qtraj = UnitaryTrajectory(sys, U_goal, N)
    @test qtraj isa UnitaryTrajectory
    @test size(qtraj[:Ũ⃗], 2) == N
    @test size(qtraj[:u], 2) == N
    @test size(qtraj[:u], 1) == 2  # 2 drives
    @test system(qtraj) === sys
    @test goal(qtraj) === U_goal
    @test state_name(qtraj) == :Ũ⃗
    @test control_name(qtraj) == :u
    
    # Test with custom initial unitary
    U_init = GATES[:I]
    qtraj2 = UnitaryTrajectory(sys, U_goal, N; U_init=U_init)
    @test qtraj2 isa UnitaryTrajectory
    @test size(qtraj2[:Ũ⃗], 2) == N
    
    # Test with fixed time (free_time=false)
    qtraj3 = UnitaryTrajectory(sys, U_goal, N; free_time=false)
    @test qtraj3 isa UnitaryTrajectory
    Δt_val = sys.T_max / (N - 1)
    @test qtraj3.bounds[:Δt][1][1] == Δt_val
    @test qtraj3.bounds[:Δt][2][1] == Δt_val
    
    # Test with custom Δt bounds
    qtraj4 = UnitaryTrajectory(sys, U_goal, N; Δt_min=0.05, Δt_max=0.2)
    @test qtraj4 isa UnitaryTrajectory
    @test qtraj4.bounds[:Δt][1][1] == 0.05
    @test qtraj4.bounds[:Δt][2][1] == 0.2
    
    # Test that time is NOT stored for non-time-dependent systems
    @test !haskey(qtraj.components, :t)
    @test :t ∉ keys(qtraj.components)
    
    # Test with linear interpolation (geodesic=false)
    qtraj6 = UnitaryTrajectory(sys, U_goal, N; geodesic=false)
    @test qtraj6 isa UnitaryTrajectory
    @test size(qtraj6[:Ũ⃗], 2) == N
end

@testitem "KetTrajectory high-level constructor" begin
    using PiccoloQuantumObjects
    using NamedTrajectories
    
    # Create a simple quantum system
    sys = QuantumSystem(
        GATES[:Z],              # H_drift
        [GATES[:X], GATES[:Y]], # H_drives
        1.0,                    # T_max
        [1.0, 1.0]             # drive_bounds
    )
    
    N = 10
    ψ_init = ComplexF64[1.0, 0.0]
    ψ_goal = ComplexF64[0.0, 1.0]
    
    # Test single state constructor
    qtraj = KetTrajectory(sys, ψ_init, ψ_goal, N)
    @test qtraj isa KetTrajectory
    @test size(qtraj[:ψ̃], 2) == N
    @test size(qtraj[:u], 2) == N
    @test size(qtraj[:u], 1) == 2  # 2 drives
    @test system(qtraj) === sys
    @test goal(qtraj) == ψ_goal
    @test state_name(qtraj) == :ψ̃
    @test control_name(qtraj) == :u
    
    # Test with fixed time
    qtraj3 = KetTrajectory(sys, ψ_init, ψ_goal, N; free_time=false)
    @test qtraj3 isa KetTrajectory
    Δt_val = sys.T_max / (N - 1)
    @test qtraj3.bounds[:Δt][1][1] == Δt_val
    @test qtraj3.bounds[:Δt][2][1] == Δt_val
    
    # Test with custom Δt bounds
    qtraj4 = KetTrajectory(sys, ψ_init, ψ_goal, N; Δt_min=0.05, Δt_max=0.2)
    @test qtraj4 isa KetTrajectory
    @test qtraj4.bounds[:Δt][1][1] == 0.05
    @test qtraj4.bounds[:Δt][2][1] == 0.2
    
    # Test with multiple states
    ψ2_init = ComplexF64[0.0, 1.0]
    ψ2_goal = ComplexF64[1.0, 0.0]
    qtraj5 = KetTrajectory(sys, [ψ_init, ψ2_init], [ψ_goal, ψ2_goal], N)
    @test qtraj5 isa KetTrajectory
    @test size(qtraj5[:ψ̃1], 2) == N
    @test size(qtraj5[:ψ̃2], 2) == N
    @test size(qtraj5[:u], 2) == N
    @test goal(qtraj5) == [ψ_goal, ψ2_goal]  # Multiple goals
    
    # Test with custom state names
    qtraj6 = KetTrajectory(sys, [ψ_init, ψ2_init], [ψ_goal, ψ2_goal], N;
        state_names=[:ψ̃_a, :ψ̃_b]
    )
    @test qtraj6 isa KetTrajectory
    @test size(qtraj6[:ψ̃_a], 2) == N
    @test size(qtraj6[:ψ̃_b], 2) == N
    @test state_name(qtraj6) == :ψ̃_a  # First state name
    
    # Test that time is NOT stored for non-time-dependent systems
    @test !haskey(qtraj.components, :t)
    @test :t ∉ keys(qtraj.components)
end

@testitem "DensityTrajectory high-level constructor" begin
    using PiccoloQuantumObjects
    using NamedTrajectories
    
    # Create an open quantum system
    sys = OpenQuantumSystem(
        GATES[:Z],              # H_drift
        [GATES[:X], GATES[:Y]], # H_drives
        1.0,                    # T_max
        [1.0, 1.0]             # drive_bounds
    )
    
    N = 10
    ρ_init = ComplexF64[1.0 0.0; 0.0 0.0]  # |0⟩⟨0|
    ρ_goal = ComplexF64[0.0 0.0; 0.0 1.0]  # |1⟩⟨1|
    
    # Test basic constructor
    qtraj = DensityTrajectory(sys, ρ_init, ρ_goal, N)
    @test qtraj isa DensityTrajectory
    @test size(qtraj[:ρ⃗̃], 2) == N
    @test size(qtraj[:u], 2) == N
    @test size(qtraj[:u], 1) == 2  # 2 drives
    @test system(qtraj) === sys
    @test goal(qtraj) == ρ_goal
    @test state_name(qtraj) == :ρ⃗̃
    @test control_name(qtraj) == :u
    
    # Test with fixed time
    qtraj3 = DensityTrajectory(sys, ρ_init, ρ_goal, N; free_time=false)
    @test qtraj3 isa DensityTrajectory
    Δt_val = sys.T_max / (N - 1)
    @test qtraj3.bounds[:Δt][1][1] == Δt_val
    @test qtraj3.bounds[:Δt][2][1] == Δt_val
    
    # Test with custom Δt bounds
    qtraj4 = DensityTrajectory(sys, ρ_init, ρ_goal, N; Δt_min=0.05, Δt_max=0.2)
    @test qtraj4 isa DensityTrajectory
    @test qtraj4.bounds[:Δt][1][1] == 0.05
    @test qtraj4.bounds[:Δt][2][1] == 0.2
    
    # Test that time is NOT stored for non-time-dependent systems
    @test !haskey(qtraj.components, :t)
    @test :t ∉ keys(qtraj.components)
end

@testitem "Time-dependent Hamiltonians with UnitaryTrajectory" begin
    using PiccoloQuantumObjects
    using NamedTrajectories
    using LinearAlgebra
    
    # Create time-dependent Hamiltonian: H(u, t) = H_drift + u(t) * cos(ω*t) * H_drive
    ω = 2π * 5.0  # Drive frequency
    H_drift = GATES[:Z]
    H_drive = GATES[:X]
    
    H(u, t) = H_drift + u[1] * cos(ω * t) * H_drive
    
    # Create system with time-dependent flag
    sys = QuantumSystem(H, 1.0, [1.0]; time_dependent=true)
    
    N = 10
    U_goal = GATES[:H]
    
    # Test that time storage is automatic for time-dependent systems
    qtraj = UnitaryTrajectory(sys, U_goal, N)
    @test qtraj isa UnitaryTrajectory
    @test haskey(qtraj.components, :t)
    @test size(qtraj[:t], 2) == N
    @test qtraj[:t][1] ≈ 0.0
    @test qtraj.initial[:t][1] ≈ 0.0
    
    # Verify time values are cumulative sums of Δt
    Δt_cumsum = [0.0; cumsum(qtraj[:Δt][:])[1:end-1]]
    @test qtraj[:t][:] ≈ Δt_cumsum
    
    # Test with custom time bounds
    qtraj2 = UnitaryTrajectory(sys, U_goal, N; Δt_min=0.05, Δt_max=0.15)
    @test qtraj2 isa UnitaryTrajectory
    @test haskey(qtraj2.components, :t)
    
    # Test that time is included in components (but not controls)
    @test :t ∈ keys(qtraj.components)
    @test :t ∉ qtraj.control_names
end

@testitem "Time-dependent Hamiltonians with KetTrajectory" begin
    using PiccoloQuantumObjects
    using NamedTrajectories
    using LinearAlgebra
    
    # Create time-dependent Hamiltonian
    ω = 2π * 5.0
    H_drift = GATES[:Z]
    H_drive = GATES[:X]
    
    H(u, t) = H_drift + u[1] * cos(ω * t) * H_drive
    
    # Create system with time-dependent flag
    sys = QuantumSystem(H, 1.0, [1.0]; time_dependent=true)
    
    N = 10
    ψ_init = ComplexF64[1.0, 0.0]
    ψ_goal = ComplexF64[0.0, 1.0]
    
    # Test single state with time-dependent system (automatic time storage)
    qtraj = KetTrajectory(sys, ψ_init, ψ_goal, N)
    @test qtraj isa KetTrajectory
    @test haskey(qtraj.components, :t)
    @test size(qtraj[:t], 2) == N
    @test qtraj[:t][1] ≈ 0.0
    @test qtraj.initial[:t][1] ≈ 0.0
    
    # Verify time values are cumulative sums of Δt
    Δt_cumsum = [0.0; cumsum(qtraj[:Δt][:])[1:end-1]]
    @test qtraj[:t][:] ≈ Δt_cumsum
    
    # Test with multiple states
    ψ2_init = ComplexF64[0.0, 1.0]
    ψ2_goal = ComplexF64[1.0, 0.0]
    qtraj2 = KetTrajectory(sys, [ψ_init, ψ2_init], [ψ_goal, ψ2_goal], N)
    @test qtraj2 isa KetTrajectory
    @test haskey(qtraj2.components, :t)
    @test size(qtraj2[:ψ̃1], 2) == N
    @test size(qtraj2[:ψ̃2], 2) == N
    @test size(qtraj2[:t], 2) == N
    
    # Test that time is included in components (but not controls)
    @test :t ∈ keys(qtraj.components)
    @test :t ∉ qtraj.control_names
end

@testitem "Time-dependent Hamiltonians with DensityTrajectory" begin
    using PiccoloQuantumObjects
    using NamedTrajectories
    using LinearAlgebra
    
    # Create time-dependent Hamiltonian
    ω = 2π * 5.0
    H_drift = GATES[:Z]
    H_drive = GATES[:X]
    
    H(u, t) = H_drift + u[1] * cos(ω * t) * H_drive
    
    # Create open system with time-dependent Hamiltonian
    sys = OpenQuantumSystem(H, 1.0, [1.0]; time_dependent=true)
    
    N = 10
    ρ_init = ComplexF64[1.0 0.0; 0.0 0.0]  # |0⟩⟨0|
    ρ_goal = ComplexF64[0.0 0.0; 0.0 1.0]  # |1⟩⟨1|
    
    # Test with time-dependent system (automatic time storage)
    qtraj = DensityTrajectory(sys, ρ_init, ρ_goal, N)
    @test qtraj isa DensityTrajectory
    @test haskey(qtraj.components, :t)
    @test size(qtraj[:t], 2) == N
    @test qtraj[:t][1] ≈ 0.0
    @test qtraj.initial[:t][1] ≈ 0.0
    
    # Verify time values are cumulative sums of Δt
    Δt_cumsum = [0.0; cumsum(qtraj[:Δt][:])[1:end-1]]
    @test qtraj[:t][:] ≈ Δt_cumsum
    
    # Test that time is included in components (but not controls)
    @test :t ∈ keys(qtraj.components)
    @test :t ∉ qtraj.control_names
end

@testitem "Multiple drives with time-dependent Hamiltonians" begin
    using PiccoloQuantumObjects
    using NamedTrajectories
    using LinearAlgebra
    
    # Create time-dependent Hamiltonian with multiple drives
    ω1 = 2π * 5.0
    ω2 = 2π * 3.0
    H_drift = GATES[:Z]
    H_drives = [GATES[:X], GATES[:Y]]
    
    H(u, t) = H_drift + u[1] * cos(ω1 * t) * H_drives[1] + u[2] * cos(ω2 * t) * H_drives[2]
    
    # Create system with multiple drives
    sys = QuantumSystem(H, 1.0, [1.0, 1.0]; time_dependent=true)
    
    N = 10
    U_goal = GATES[:H]
    
    # Test with multiple drives (automatic time storage)
    qtraj = UnitaryTrajectory(sys, U_goal, N)
    @test qtraj isa UnitaryTrajectory
    @test size(qtraj[:u], 1) == 2  # 2 drives
    @test haskey(qtraj.components, :t)
    @test size(qtraj[:t], 2) == N
    
    # Test initial and final control constraints
    @test all(qtraj[:u][:, 1] .== 0.0)  # Initial controls are zero
    @test all(qtraj[:u][:, end] .== 0.0)  # Final controls are zero
    
    # Test bounds on multiple drives
    @test qtraj.bounds[:u][1] == [-1.0, -1.0]  # Lower bounds
    @test qtraj.bounds[:u][2] == [1.0, 1.0]    # Upper bounds
end

end