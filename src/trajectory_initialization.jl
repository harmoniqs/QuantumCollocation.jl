module TrajectoryInitialization

export unitary_geodesic
export linear_interpolation
export unitary_linear_interpolation
export initialize_trajectory
export unitary_trajectory
export ket_trajectory
export density_trajectory

using NamedTrajectories
import NamedTrajectories.StructNamedTrajectory: ScalarBound, VectorBound
using PiccoloQuantumObjects

using Distributions
using ExponentialAction
using LinearAlgebra
using TestItems


# ----------------------------------------------------------------------------- #
#                           Initial states                                      #
# ----------------------------------------------------------------------------- #

linear_interpolation(x::AbstractVector, y::AbstractVector, n::Int) = hcat(range(x, y, n)...)

"""
    unitary_linear_interpolation(
        U_init::AbstractMatrix,
        U_goal::AbstractMatrix,
        samples::Int
    )

Compute a linear interpolation of unitary operators with `samples` samples.
"""
function unitary_linear_interpolation(
    U_init::AbstractMatrix{<:Number},
    U_goal::AbstractMatrix{<:Number},
    samples::Int
)
    Ũ⃗_init = operator_to_iso_vec(U_init)
    Ũ⃗_goal = operator_to_iso_vec(U_goal)
    Ũ⃗s = [Ũ⃗_init + (Ũ⃗_goal - Ũ⃗_init) * t for t ∈ range(0, 1, length=samples)]
    Ũ⃗ = hcat(Ũ⃗s...)
    return Ũ⃗
end

function unitary_linear_interpolation(
    U_init::AbstractMatrix{<:Number},
    U_goal::EmbeddedOperator,
    samples::Int
)
    return unitary_linear_interpolation(U_init, U_goal.operator, samples)
end

"""
    unitary_geodesic(U_init, U_goal, times; kwargs...)

Compute the geodesic connecting U_init and U_goal at the specified times.

# Arguments
- `U_init::AbstractMatrix{<:Number}`: The initial unitary operator.
- `U_goal::AbstractMatrix{<:Number}`: The goal unitary operator.
- `times::AbstractVector{<:Number}`: The times at which to evaluate the geodesic.

# Keyword Arguments
- `return_unitary_isos::Bool=true`: If true returns a matrix where each column is a unitary 
    isovec, i.e. vec(vcat(real(U), imag(U))). If false, returns a vector of unitary matrices.
- `return_generator::Bool=false`: If true, returns the effective Hamiltonian generating 
    the geodesic.
"""
function unitary_geodesic end

function unitary_geodesic(
    U_init::AbstractMatrix{<:Number},
    U_goal::AbstractMatrix{<:Number},
    times::AbstractVector{<:Number};
    return_unitary_isos=true,
    return_generator=false,
    H_drift::AbstractMatrix{<:Number}=zeros(size(U_init)),
)
    t₀ = times[1]
    T = times[end] - t₀

    U_drift(t) = exp(-im * H_drift * t)
    H = im * log(U_drift(T)' * (U_goal * U_init')) / T
    # -im prefactor is not included in H
    U_geo = [U_drift(t) * exp(-im * H * (t - t₀)) * U_init for t ∈ times]

    if !return_unitary_isos
        if return_generator
            return U_geo, H
        else
            return U_geo
        end
    else
        Ũ⃗_geo = stack(operator_to_iso_vec.(U_geo), dims=2)
        if return_generator
            return Ũ⃗_geo, H
        else
            return Ũ⃗_geo
        end
    end
end

function unitary_geodesic(
    U_goal::AbstractPiccoloOperator,
    samples::Int;
    kwargs...
)
    return unitary_geodesic(
        I(size(U_goal, 1)),
        U_goal,
        samples;
        kwargs...
    )
end

function unitary_geodesic(
    U_init::AbstractMatrix{<:Number},
    U_goal::EmbeddedOperator,
    samples::Int;
    H_drift::AbstractMatrix{<:Number}=zeros(size(U_init)),
    kwargs...
)
    H_drift = unembed(H_drift, U_goal)
    U1 = unembed(U_init, U_goal)
    U2 = unembed(U_goal)
    Ũ⃗ = unitary_geodesic(U1, U2, samples; H_drift=H_drift, kwargs...)
    return hcat([
        operator_to_iso_vec(embed(iso_vec_to_operator(Ũ⃗ₜ), U_goal))
        for Ũ⃗ₜ ∈ eachcol(Ũ⃗)
    ]...)
end

function unitary_geodesic(
    U_init::AbstractMatrix{<:Number},
    U_goal::AbstractMatrix{<:Number},
    samples::Int;
    kwargs...
)
    return unitary_geodesic(U_init, U_goal, range(0, 1, samples); kwargs...)
end

linear_interpolation(X::AbstractMatrix, Y::AbstractMatrix, n::Int) =
    hcat([X + (Y - X) * t for t in range(0, 1, length=n)]...)

# ============================================================================= #

"""
    initialize_unitary_trajectory(
        U_init::AbstractMatrix{<:Number},
        U_goal::AbstractPiccoloOperator,
        N::Int;
        geodesic::Bool=true,
        system::Union{AbstractQuantumSystem, Nothing}=nothing
    )

Generate an initial unitary trajectory from `U_init` to `U_goal`.

# Arguments
- `U_init::AbstractMatrix{<:Number}`: Initial unitary operator
- `U_goal::AbstractPiccoloOperator`: Target unitary operator
- `N::Int`: Number of time steps

# Keyword Arguments
- `geodesic::Bool=true`: Use geodesic interpolation (vs. linear interpolation)
- `system::Union{AbstractQuantumSystem, Nothing}=nothing`: System for drift Hamiltonian

# Returns
- `Matrix{Float64}`: Trajectory of unitaries in iso-vec representation (column per timestep)

# Notes
- Geodesic interpolation follows the shortest path on the unitary manifold
- If a system is provided, the drift Hamiltonian is used in geodesic computation
"""
function initialize_unitary_trajectory(
    U_init::AbstractMatrix{<:Number},
    U_goal::AbstractPiccoloOperator,
    N::Int;
    geodesic::Bool=true,
    system::Union{AbstractQuantumSystem, Nothing}=nothing
)
    if geodesic
        if system isa AbstractQuantumSystem
            H_drift = Matrix(get_drift(system))
        else
            H_drift = zeros(size(U_init))
        end
        Ũ⃗ = unitary_geodesic(U_init, U_goal, N, H_drift=H_drift)
    else
        Ũ⃗ = unitary_linear_interpolation(U_init, U_goal, N)
    end
    return Ũ⃗
end

# ----------------------------------------------------------------------------- #
#                           Initial controls                                    #
# ----------------------------------------------------------------------------- #

"""
    initialize_control_trajectory(
        n_drives::Int,
        n_derivatives::Int,
        N::Int,
        bounds::VectorBound,
        drive_derivative_σ::Float64
    )

Generate random initial control trajectories with derivatives.

Creates smooth control trajectories by randomly sampling the base controls within bounds
and generating higher derivatives from a Gaussian distribution.

# Arguments
- `n_drives::Int`: Number of independent control drives
- `n_derivatives::Int`: Number of derivatives to generate (e.g., 2 for u, du, ddu)
- `N::Int`: Number of time steps
- `bounds::VectorBound`: Bounds for the control amplitudes (vector or tuple of bounds)
- `drive_derivative_σ::Float64`: Standard deviation for random derivative initialization

# Returns
- `Vector{Matrix{Float64}}`: Vector of control matrices [u, du, ddu, ...]

# Notes
- Base controls are zero at initial and final timesteps for smooth boundary conditions
- Derivatives are sampled from N(0, drive_derivative_σ²)
"""
function initialize_control_trajectory(
    n_drives::Int,
    n_derivatives::Int,
    N::Int,
    bounds::VectorBound,
    drive_derivative_σ::Float64,
)
    if bounds isa AbstractVector
        a_dists = [Uniform(-bounds[i], bounds[i]) for i = 1:n_drives]
    elseif bounds isa Tuple
        a_dists = [Uniform(aᵢ_lb, aᵢ_ub) for (aᵢ_lb, aᵢ_ub) ∈ zip(bounds...)]
    else
        error("bounds must be a Vector or Tuple")
    end

    controls = Matrix{Float64}[]

    a = hcat([
        zeros(n_drives),
        vcat([rand(a_dists[i], 1, N - 2) for i = 1:n_drives]...),
        zeros(n_drives)
    ]...)
    push!(controls, a)

    for _ in 1:n_derivatives
        push!(controls, randn(n_drives, N) * drive_derivative_σ)
    end

    return controls
end

"""
    initialize_control_trajectory(
        u::AbstractMatrix,
        Δt::AbstractVecOrMat,
        n_derivatives::Int
    )

Generate control derivatives from a provided control trajectory.

Takes a given control trajectory and computes its time derivatives using finite differences.
Ensures smooth transitions at boundaries to avoid constraint violations.

# Arguments
- `u::AbstractMatrix`: Control trajectory (n_drives × N)
- `Δt::AbstractVecOrMat`: Time step size(s)
- `n_derivatives::Int`: Number of derivatives to compute

# Returns
- `Vector{Matrix{Float64}}`: Vector of control matrices [u, du, ddu, ...]

# Notes
- Uses finite difference approximation for derivatives
- Adjusts penultimate point to ensure smooth final derivative
"""
function initialize_control_trajectory(
    a::AbstractMatrix,
    Δt::AbstractVecOrMat,
    n_derivatives::Int
)
    controls = Matrix{Float64}[a]

    for n in 1:n_derivatives
        # next derivative
        push!(controls,  derivative(controls[end], Δt))

        # to avoid constraint violation error at initial iteration for da, dda, ...
        if n > 1
            controls[end-1][:, end] =
                controls[end-1][:, end-1] + Δt[end-1] * controls[end][:, end-1]
        end
    end
    return controls
end

initialize_control_trajectory(a::AbstractMatrix, Δt::Real, n_derivatives::Int) =
    initialize_control_trajectory(a, fill(Δt, size(a, 2)), n_derivatives)

# ----------------------------------------------------------------------------- #
#                           Trajectory initialization                           #
# ----------------------------------------------------------------------------- #

"""
    initialize_trajectory(
        state_data::Vector{<:AbstractMatrix{Float64}},
        state_inits::Vector{<:AbstractVector{Float64}},
        state_goals::Vector{<:AbstractVector{Float64}},
        state_names::AbstractVector{Symbol},
        N::Int,
        Δt::Union{Float64, AbstractVecOrMat{<:Float64}},
        n_drives::Int,
        control_bounds::Tuple{Vararg{VectorBound}};
        kwargs...
    )

Initialize a trajectory for a quantum control problem with custom state data.

# Arguments
- `state_data::Vector{<:AbstractMatrix{Float64}}`: Pre-computed state trajectories (one matrix per state)
- `state_inits::Vector{<:AbstractVector{Float64}}`: Initial state values
- `state_goals::Vector{<:AbstractVector{Float64}}`: Target state values
- `state_names::AbstractVector{Symbol}`: Names for each state component
- `N::Int`: Number of time steps
- `Δt::Union{Float64, AbstractVecOrMat{<:Float64}}`: Time step size(s)
- `n_drives::Int`: Number of control drives
- `control_bounds::Tuple{Vararg{VectorBound}}`: Bounds for controls and their derivatives

# Keyword Arguments
- `bound_state::Bool=false`: Whether to bound the state variables
- `control_name::Symbol=:u`: Name for the control variable
- `n_control_derivatives::Int=length(control_bounds) - 1`: Number of control derivatives
- `zero_initial_and_final_derivative::Bool=false`: Enforce zero derivatives at boundaries
- `timestep_name::Symbol=:Δt`: Name for the timestep variable
- `Δt_bounds::ScalarBound=(0.5 * Δt, 1.5 * Δt)`: Bounds for the timestep
- `drive_derivative_σ::Float64=0.1`: Standard deviation for random control derivatives
- `u_guess::Union{AbstractMatrix{<:Float64}, Nothing}=nothing`: Initial guess for controls
- `global_component_data::NamedTuple=NamedTuple()`: Additional global trajectory components
- `verbose::Bool=false`: Print detailed initialization information
- `store_times::Bool=false`: Store cumulative time values in the trajectory

# Returns
- `NamedTrajectory`: Initialized trajectory with states, controls, and timesteps
"""
function initialize_trajectory(
    state_data::Vector{<:AbstractMatrix{Float64}},
    state_inits::Vector{<:AbstractVector{Float64}},
    state_goals::Vector{<:AbstractVector{Float64}},
    state_names::AbstractVector{Symbol},
    N::Int,
    Δt::Union{Float64, AbstractVecOrMat{<:Float64}},
    n_drives::Int,
    control_bounds::Tuple{Vararg{VectorBound}};
    bound_state=false,
    control_name=:u,
    n_control_derivatives::Int=length(control_bounds) - 1,
    zero_initial_and_final_derivative=false,
    timestep_name=:Δt,
    Δt_bounds::ScalarBound=(0.5 * Δt, 1.5 * Δt),
    drive_derivative_σ::Float64=0.1,
    u_guess::Union{AbstractMatrix{<:Float64}, Nothing}=nothing,
    global_component_data::NamedTuple{gname, <:Tuple{Vararg{AbstractVector{<:Real}}}} where gname=(;),
    verbose=false,
    store_times=false,
)
    @assert length(state_data) == length(state_names) == length(state_inits) == length(state_goals) "state_data, state_names, state_inits, and state_goals must have the same length"
    @assert length(control_bounds) == n_control_derivatives + 1 "control_bounds must have $n_control_derivatives + 1 elements"

    # assert that state names are unique
    @assert length(state_names) == length(Set(state_names)) "state_names must be unique"

    # Control data
    control_derivative_names = [
        Symbol("d"^i * string(control_name)) for i = 1:n_control_derivatives
    ]
    if verbose
        println("\tcontrol derivative names: $control_derivative_names")
    end

    control_names = (control_name, control_derivative_names...)

    control_bounds = NamedTuple{control_names}(control_bounds)

    # Timestep data
    if Δt isa Real
        timestep_data = fill(Δt, 1, N)
    elseif Δt isa AbstractVector
        timestep_data = reshape(Δt, 1, :)
    else
        timestep_data = Δt
        @assert size(Δt) == (1, N) "Δt must be a Real, AbstractVector, or 1x$(N) AbstractMatrix"
    end
    timestep = timestep_name

    # Constraints
    initial = (;
        (state_names .=> state_inits)...,
        control_name => zeros(n_drives),
    )

    if store_times
        initial = merge(initial, (; t=[0.0]))
        t_data = cumsum(timestep_data, dims=2)
    end

    final = (;
        control_name => zeros(n_drives),
    )

    if zero_initial_and_final_derivative
        initial = merge(initial, (; control_derivative_names[1] => zeros(n_drives),))
        final = merge(final, (; control_derivative_names[1] => zeros(n_drives),))
    end

    goal = (; (state_names .=> state_goals)...)

    # Bounds
    bounds = control_bounds

    bounds = merge(bounds, (; timestep_name => Δt_bounds,))

    # Put unit box bounds on the state if bound_state is true
    if bound_state
        state_dim = length(state_inits[1])
        state_bounds = repeat([(-ones(state_dim), ones(state_dim))], length(state_names))
        bounds = merge(bounds, (; (state_names .=> state_bounds)...))
    end

    # Trajectory
    if isnothing(u_guess)
        # Randomly sample controls
        control_data = initialize_control_trajectory(
            n_drives,
            n_control_derivatives,
            N,
            bounds[control_name],
            drive_derivative_σ
        )
    else
        # Use provided controls and take derivatives
        control_data = initialize_control_trajectory(u_guess, Δt, n_control_derivatives)
    end

    names = [state_names..., control_names..., timestep_name]
    values = [state_data..., control_data..., timestep_data]
    controls = (control_names[end], timestep_name)

    if store_times
        names = [names..., :t]
        values = [values..., t_data]
        controls = (controls..., :t)
    end

    return NamedTrajectory(
        (; (names .=> values)...),
        global_component_data;
        controls=controls,
        timestep=timestep,
        bounds=bounds,
        initial=initial,
        final=final,
        goal=goal,
    )
end

"""
    initialize_trajectory(
        U_goal::AbstractPiccoloOperator,
        N::Int,
        Δt::Union{Real, AbstractVecOrMat{<:Real}},
        n_drives::Int,
        control_bounds::Tuple{Vararg{VectorBound}};
        kwargs...
    )

Initialize a trajectory for unitary gate synthesis problems.

Constructs a trajectory that evolves from an initial unitary (default: identity) to a target
unitary gate. The trajectory can use geodesic interpolation or rollout-based initialization.

# Arguments
- `U_goal::AbstractPiccoloOperator`: Target unitary operator (can be `EmbeddedOperator`)
- `N::Int`: Number of time steps
- `Δt::Union{Real, AbstractVecOrMat{<:Real}}`: Time step size(s)
- `n_drives::Int`: Number of control drives
- `control_bounds::Tuple{Vararg{VectorBound}}`: Bounds for controls and their derivatives

# Keyword Arguments
- `state_name::Symbol=:Ũ⃗`: Name for the unitary state variable (iso-vec representation)
- `U_init::AbstractMatrix{<:Number}=I`: Initial unitary operator
- `u_guess::Union{AbstractMatrix{<:Float64}, Nothing}=nothing`: Initial guess for controls
- `system::Union{AbstractQuantumSystem, Nothing}=nothing`: Quantum system for rollout
- `rollout_integrator::Function=expv`: Integrator for unitary dynamics
- `geodesic=true`: Use geodesic interpolation between unitaries
- Additional kwargs passed to the base `initialize_trajectory` method

# Returns
- `NamedTrajectory`: Initialized trajectory with unitary states, controls, and timesteps

# Notes
- If `u_guess` is provided, the trajectory is computed via rollout using the quantum system
- If `u_guess` is `nothing`, geodesic interpolation is used (requires `geodesic=true`)
- The unitary is stored in iso-vec representation for efficient optimization
"""
function initialize_trajectory(
    U_goal::AbstractPiccoloOperator,
    N::Int,
    Δt::Union{Real, AbstractVecOrMat{<:Real}},
    args...;
    state_name::Symbol=:Ũ⃗,
    U_init::AbstractMatrix{<:Number}=Matrix{ComplexF64}(I(size(U_goal, 1))),
    u_guess::Union{AbstractMatrix{<:Float64}, Nothing}=nothing,
    system::Union{AbstractQuantumSystem, Nothing}=nothing,
    rollout_integrator::Function=expv,
    geodesic=true,
    kwargs...
)
    # Construct timesteps
    if Δt isa AbstractMatrix
        timesteps = vec(Δt)
    elseif Δt isa Float64
        timesteps = fill(Δt, N)
    else
        timesteps = Δt
    end

    # Initial state and goal
    Ũ⃗_init = operator_to_iso_vec(U_init)

    if U_goal isa EmbeddedOperator
        Ũ⃗_goal = operator_to_iso_vec(U_goal.operator)
    else
        Ũ⃗_goal = operator_to_iso_vec(U_goal)
    end

    # Construct state data
    if isnothing(u_guess)
        Ũ⃗_traj = initialize_unitary_trajectory(
            U_init, 
            U_goal, 
            N; 
            geodesic=geodesic, 
            system=system
        )
    else
        @assert !isnothing(system) "System must be provided if u_guess is provided."
        Ũ⃗_traj = unitary_rollout(Ũ⃗_init, u_guess, timesteps, system; integrator=rollout_integrator)
    end
    
    return initialize_trajectory(
        [Ũ⃗_traj],
        [Ũ⃗_init],
        [Ũ⃗_goal],
        [state_name],
        N,
        Δt,
        args...;
        u_guess=u_guess,
        kwargs...
    )
end



"""
    initialize_trajectory(
        ψ_goals::AbstractVector{<:AbstractVector{ComplexF64}},
        ψ_inits::AbstractVector{<:AbstractVector{ComplexF64}},
        N::Int,
        Δt::Union{Real, AbstractVector{<:Real}},
        n_drives::Int,
        control_bounds::Tuple{Vararg{VectorBound}};
        kwargs...
    )

Initialize a trajectory for quantum state transfer problems.

Constructs a trajectory that evolves one or more quantum states from initial states to target
states. Supports multiple simultaneous state trajectories with shared controls.

# Arguments
- `ψ_goals::AbstractVector{<:AbstractVector{ComplexF64}}`: Target quantum state(s)
- `ψ_inits::AbstractVector{<:AbstractVector{ComplexF64}}`: Initial quantum state(s)
- `N::Int`: Number of time steps
- `Δt::Union{Real, AbstractVector{<:Real}}`: Time step size(s)
- `n_drives::Int`: Number of control drives
- `control_bounds::Tuple{Vararg{VectorBound}}`: Bounds for controls and their derivatives

# Keyword Arguments
- `state_name::Symbol=:ψ̃`: Base name for state variables (iso representation)
- `state_names::AbstractVector{<:Symbol}`: Explicit names for each state (auto-generated if not provided)
- `u_guess::Union{AbstractMatrix{<:Float64}, Nothing}=nothing`: Initial guess for controls
- `system::Union{AbstractQuantumSystem, Nothing}=nothing`: Quantum system for rollout
- `rollout_integrator::Function=expv`: Integrator for state dynamics
- Additional kwargs passed to the base `initialize_trajectory` method

# Returns
- `NamedTrajectory`: Initialized trajectory with quantum states, controls, and timesteps

# Notes
- States are stored in iso representation (real-valued vectors) for optimization
- If `u_guess` is provided, trajectories are computed via rollout
- If `u_guess` is `nothing`, states are linearly interpolated
- Multiple states share the same control trajectory
"""
function initialize_trajectory(
    ψ_goals::AbstractVector{<:AbstractVector{ComplexF64}},
    ψ_inits::AbstractVector{<:AbstractVector{ComplexF64}},
    N::Int,
    Δt::Union{Real, AbstractVector{<:Real}},
    args...;
    state_name=:ψ̃,
    state_names::AbstractVector{<:Symbol}=length(ψ_goals) == 1 ?
        [state_name] :
        [Symbol(string(state_name) * "$i") for i = 1:length(ψ_goals)],
    u_guess::Union{AbstractMatrix{<:Float64}, Nothing}=nothing,
    system::Union{AbstractQuantumSystem, Nothing}=nothing,
    rollout_integrator::Function=expv,
    kwargs...
)
    @assert length(ψ_inits) == length(ψ_goals) "ψ_inits and ψ_goals must have the same length"
    @assert length(state_names) == length(ψ_goals) "state_names and ψ_goals must have the same length"

    ψ̃_goals = ket_to_iso.(ψ_goals)
    ψ̃_inits = ket_to_iso.(ψ_inits)

    # Construct timesteps
    if Δt isa AbstractMatrix
        timesteps = vec(Δt)
    elseif Δt isa Float64
        timesteps = fill(Δt, N)
    else
        timesteps = Δt
    end

    # Construct state data
    ψ̃_trajs = Matrix{Float64}[]
    if isnothing(u_guess)
        for (ψ̃_init, ψ̃_goal) ∈ zip(ψ̃_inits, ψ̃_goals)
            ψ̃_traj = linear_interpolation(ψ̃_init, ψ̃_goal, N)
            push!(ψ̃_trajs, ψ̃_traj)
        end
        if system isa AbstractVector
            ψ̃_trajs = repeat(ψ̃_trajs, length(system))
        end
    else
        for ψ̃_init ∈ ψ̃_inits
            ψ̃_traj = rollout(ψ̃_init, u_guess, timesteps, system; integrator=rollout_integrator)
            push!(ψ̃_trajs, ψ̃_traj)
        end
    end

    return initialize_trajectory(
        ψ̃_trajs,
        ψ̃_inits,
        ψ̃_goals,
        state_names,
        N,
        Δt,
        args...;
        u_guess=u_guess,
        kwargs...
    )
end

"""
    initialize_trajectory(
        ρ_init,
        ρ_goal,
        N::Int,
        Δt::Union{Real, AbstractVecOrMat{<:Real}},
        args...;
        state_name::Symbol=:ρ⃗̃,
        u_guess::Union{AbstractMatrix{<:Float64}, Nothing}=nothing,
        system::Union{OpenQuantumSystem, Nothing}=nothing,
        rollout_integrator::Function=expv,
        kwargs...
    )

Initialize a trajectory for open quantum system density matrix evolution.

Constructs a trajectory for evolving a density matrix from an initial state to a target
state, supporting both unitary and open system dynamics. Density matrices are stored in
iso-vectorized form for optimization.

# Arguments
- `ρ_init`: Initial density matrix
- `ρ_goal`: Target density matrix
- `N::Int`: Number of time steps
- `Δt::Union{Real, AbstractVecOrMat{<:Real}}`: Time step size(s)
- `args...`: Additional arguments passed to base `initialize_trajectory`

# Keyword Arguments
- `state_name::Symbol=:ρ⃗̃`: Name for the density matrix state variable
- `u_guess::Union{AbstractMatrix{<:Float64}, Nothing}=nothing`: Initial control guess
- `system::Union{OpenQuantumSystem, Nothing}=nothing`: Open quantum system for rollout
- `rollout_integrator::Function=expv`: Integrator for open system dynamics
- Additional kwargs passed to the base `initialize_trajectory` method

# Returns
- `NamedTrajectory`: Initialized trajectory with density matrix state, controls, and timesteps

# Notes
- Density matrices are stored in iso-vectorized representation
- If `u_guess` is provided, requires `system` and uses rollout for state trajectory
- If `u_guess` is `nothing`, uses linear interpolation between initial and target states
- Uses `open_rollout` for open system dynamics with Lindblad master equation
"""
function initialize_trajectory(
    ρ_init,
    ρ_goal,
    N::Int,
    Δt::Union{Real, AbstractVecOrMat{<:Real}},
    args...;
    state_name::Symbol=:ρ⃗̃,
    u_guess::Union{AbstractMatrix{<:Float64}, Nothing}=nothing,
    system::Union{OpenQuantumSystem, Nothing}=nothing,
    rollout_integrator::Function=expv,
    kwargs...
)
    # Construct timesteps
    if Δt isa AbstractMatrix
        timesteps = vec(Δt)
    elseif Δt isa Float64
        timesteps = fill(Δt, N)
    else
        timesteps = Δt
    end

    # Initial state and goal
    ρ⃗̃_init = density_to_iso_vec(ρ_init)
    ρ⃗̃_goal = density_to_iso_vec(ρ_goal)

    # Construct state data
    if isnothing(u_guess)
        ρ⃗̃_traj = linear_interpolation(ρ_init, ρ_goal, N)
    else
        @assert !isnothing(system) "System must be provided if u_guess is provided."

        ρ⃗̃_traj = open_rollout(
            ρ_init,
            u_guess,
            timesteps,
            system;
            integrator=rollout_integrator
        )
    end

    return initialize_trajectory(
        [ρ⃗̃_traj],
        [ρ⃗̃_init],
        [ρ⃗̃_goal],
        [state_name],
        N,
        Δt,
        args...;
        u_guess=u_guess,
        kwargs...
    )
end

# ============================================================================= #
#                        Convenience trajectory creators                        #
# ============================================================================= #

"""
    unitary_trajectory(
        sys::QuantumSystem,
        U_goal::AbstractMatrix{<:Number},
        N::Int;
        U_init::AbstractMatrix{<:Number}=I(size(sys.H_drift, 1)),
        Δt_min::Float64=sys.T_max / (2 * (N-1)),
        Δt_max::Float64=2 * sys.T_max / (N-1),
        free_time::Bool=true,
        geodesic::Bool=true,
        store_times::Bool=false
    )

Create a unitary trajectory initialized from a quantum system.

# Arguments
- `sys::QuantumSystem`: The quantum system
- `U_goal::AbstractMatrix`: Target unitary operator
- `N::Int`: Number of knot points

# Keyword Arguments
- `U_init::AbstractMatrix`: Initial unitary (default: identity)
- `Δt_min::Float64`: Minimum timestep (default: T_max / (2*(N-1)))
- `Δt_max::Float64`: Maximum timestep (default: 2*T_max / (N-1))
- `free_time::Bool`: Whether timesteps are free or fixed (default: true)
- `geodesic::Bool`: Use geodesic interpolation (default: true)
- `store_times::Bool`: Store cumulative time values in the trajectory (default: false)

# Returns
- `UnitaryTrajectory`: Initialized unitary trajectory with quantum metadata
"""
function unitary_trajectory(
    sys::QuantumSystem,
    U_goal::AbstractMatrix{<:Number},
    N::Int;
    U_init::AbstractMatrix{<:Number}=Matrix{ComplexF64}(I(size(sys.H_drift, 1))),
    Δt_min::Float64=sys.T_max / (2 * (N-1)),
    Δt_max::Float64=2 * sys.T_max / (N-1),
    free_time::Bool=true,
    geodesic::Bool=true,
    store_times::Bool=false
)
    Δt = sys.T_max / (N - 1)
    n_drives = length(sys.H_drives)
    
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
    goal = (Ũ⃗ = Ũ⃗_goal,)
    
    # Time data
    if store_times
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
    controls = (:u, :Δt)
    
    if store_times
        comps_data = merge(comps_data, (t = reshape(t_data, 1, N),))
        controls = (controls..., :t)
    end
    
    traj = NamedTrajectory(
        comps_data;
        controls = controls,
        timestep = :Δt,
        initial = initial,
        final = final,
        goal = goal,
        bounds = bounds
    )
    
    return UnitaryTrajectory(traj, sys, :Ũ⃗, :u, U_goal)
end

"""
    ket_trajectory(
        sys::QuantumSystem,
        ψ_inits::AbstractVector{<:AbstractVector{<:ComplexF64}},
        ψ_goals::AbstractVector{<:AbstractVector{<:ComplexF64}},
        N::Int;
        state_name::Symbol=:ψ̃,
        state_names::Union{AbstractVector{<:Symbol}, Nothing}=nothing,
        Δt_min::Float64=sys.T_max / (2 * (N-1)),
        Δt_max::Float64=2 * sys.T_max / (N-1),
        free_time::Bool=true,
        store_times::Bool=false
    )

Create a ket state trajectory initialized from a quantum system.

Supports multiple simultaneous state trajectories with shared controls.

# Arguments
- `sys::QuantumSystem`: The quantum system
- `ψ_inits::AbstractVector`: Vector of initial ket states
- `ψ_goals::AbstractVector`: Vector of target ket states
- `N::Int`: Number of knot points

# Keyword Arguments
- `state_name::Symbol`: Base name for state variables (default: :ψ̃)
- `state_names::Union{AbstractVector{<:Symbol}, Nothing}`: Explicit names for each state (auto-generated if not provided)
- `Δt_min::Float64`: Minimum timestep (default: T_max / (2*(N-1)))
- `Δt_max::Float64`: Maximum timestep (default: 2*T_max / (N-1))
- `free_time::Bool`: Whether timesteps are free or fixed (default: true)
- `store_times::Bool`: Store cumulative time values in the trajectory (default: false)

# Returns
- `KetTrajectory`: Initialized ket trajectory with quantum metadata

# Examples
```julia
# Single state
traj = ket_trajectory(sys, [ψ_init], [ψ_goal], 10)

# Multiple states with shared controls
traj = ket_trajectory(sys, [ψ1_init, ψ2_init], [ψ1_goal, ψ2_goal], 10)
```
"""
function ket_trajectory(
    sys::QuantumSystem,
    ψ_inits::AbstractVector{<:AbstractVector{<:ComplexF64}},
    ψ_goals::AbstractVector{<:AbstractVector{<:ComplexF64}},
    N::Int;
    state_name::Symbol=:ψ̃,
    state_names::Union{AbstractVector{<:Symbol}, Nothing}=nothing,
    Δt_min::Float64=sys.T_max / (2 * (N-1)),
    Δt_max::Float64=2 * sys.T_max / (N-1),
    free_time::Bool=true,
    store_times::Bool=false
)
    @assert length(ψ_inits) == length(ψ_goals) "ψ_inits and ψ_goals must have the same length"
    
    Δt = sys.T_max / (N - 1)
    n_drives = length(sys.H_drives)
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
    goal = goal_states
    
    # Time data
    if store_times
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
    controls = (:u, :Δt)
    
    if store_times
        comps_data = merge(comps_data, (t = reshape(t_data, 1, N),))
        controls = (controls..., :t)
    end
    
    traj = NamedTrajectory(
        comps_data;
        controls = controls,
        timestep = :Δt,
        initial = initial,
        final = final,
        goal = goal,
        bounds = bounds
    )
    
    # Return wrapped trajectory with metadata
    # Use first state name for the trajectory wrapper
    return KetTrajectory(traj, sys, state_names[1], :u, ψ_goals)
end

"""
    ket_trajectory(
        sys::QuantumSystem,
        ψ_init::AbstractVector{<:ComplexF64},
        ψ_goal::AbstractVector{<:ComplexF64},
        N::Int;
        kwargs...
    )

Convenience constructor for a single ket state trajectory.

# Arguments
- `sys::QuantumSystem`: The quantum system
- `ψ_init::AbstractVector`: Initial ket state
- `ψ_goal::AbstractVector`: Target ket state
- `N::Int`: Number of knot points

# Keyword Arguments
- `kwargs...`: Additional arguments passed to the main `ket_trajectory` method

# Returns
- `KetTrajectory`: Initialized ket trajectory with quantum metadata
"""
function ket_trajectory(
    sys::QuantumSystem,
    ψ_init::AbstractVector{<:ComplexF64},
    ψ_goal::AbstractVector{<:ComplexF64},
    N::Int;
    kwargs...
)
    return ket_trajectory(sys, [ψ_init], [ψ_goal], N; kwargs...)
end

"""
    density_trajectory(
        sys::OpenQuantumSystem,
        ρ_init::AbstractMatrix,
        ρ_goal::AbstractMatrix,
        N::Int;
        Δt_min::Float64=sys.T_max / (2 * (N-1)),
        Δt_max::Float64=2 * sys.T_max / (N-1),
        free_time::Bool=true,
        store_times::Bool=false
    )

Create a density matrix trajectory initialized from an open quantum system.

# Arguments
- `sys::OpenQuantumSystem`: The open quantum system
- `ρ_init::AbstractMatrix`: Initial density matrix
- `ρ_goal::AbstractMatrix`: Target density matrix
- `N::Int`: Number of knot points

# Keyword Arguments
- `Δt_min::Float64`: Minimum timestep (default: T_max / (2*(N-1)))
- `Δt_max::Float64`: Maximum timestep (default: 2*T_max / (N-1))
- `free_time::Bool`: Whether timesteps are free or fixed (default: true)
- `store_times::Bool`: Store cumulative time values in the trajectory (default: false)

# Returns
- `DensityTrajectory`: Initialized density matrix trajectory with quantum metadata
"""
function density_trajectory(
    sys::OpenQuantumSystem,
    ρ_init::AbstractMatrix,
    ρ_goal::AbstractMatrix,
    N::Int;
    Δt_min::Float64=sys.T_max / (2 * (N-1)),
    Δt_max::Float64=2 * sys.T_max / (N-1),
    free_time::Bool=true,
    store_times::Bool=false
)
    Δt = sys.T_max / (N - 1)
    n_drives = length(sys.H_drives)
    
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
    goal = (ρ⃗̃ = ρ⃗̃_goal,)
    
    # Time data
    if store_times
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
    comps_data = (ρ⃗̃ = ρ⃗̃, u = u, Δt = reshape(Δt_vec, 1, N))
    controls = (:u, :Δt)
    
    if store_times
        comps_data = merge(comps_data, (t = reshape(t_data, 1, N),))
        controls = (controls..., :t)
    end
    
    traj = NamedTrajectory(
        comps_data;
        controls = controls,
        timestep = :Δt,
        initial = initial,
        final = final,
        goal = goal,
        bounds = bounds
    )
    
    return DensityTrajectory(traj, sys, :ρ⃗̃, :u, ρ_goal)
end


# ============================================================================= #

@testitem "Random drive initialization" begin
    T = 10
    n_drives = 2
    n_derivates = 2
    drive_bounds = [1.0, 2.0]
    drive_derivative_σ = 0.01

    a, da, dda = TrajectoryInitialization.initialize_control_trajectory(n_drives, n_derivates, T, drive_bounds, drive_derivative_σ)

    @test size(a) == (n_drives, T)
    @test size(da) == (n_drives, T)
    @test size(dda) == (n_drives, T)
    @test all([-drive_bounds[i] < minimum(a[i, :]) < drive_bounds[i] for i in 1:n_drives])
end

@testitem "Geodesic" begin
    using LinearAlgebra
    using PiccoloQuantumObjects 

    ## Group 1: identity to X (π rotation)

    # Test π rotation
    U_α = GATES[:I]
    U_ω = GATES[:X]
    Us, H = unitary_geodesic(
        U_α, U_ω, range(0, 1, 4), return_generator=true
    )

    @test size(Us, 2) == 4
    @test Us[:, 1] ≈ operator_to_iso_vec(U_α)
    @test Us[:, end] ≈ operator_to_iso_vec(U_ω)
    @test H' - H ≈ zeros(2, 2)
    @test norm(H) ≈ π

    # Test modified timesteps (10x)
    Us10, H10 = unitary_geodesic(
        U_α, U_ω, range(-5, 5, 4), return_generator=true
    )

    @test size(Us10, 2) == 4
    @test Us10[:, 1] ≈ operator_to_iso_vec(U_α)
    @test Us10[:, end] ≈ operator_to_iso_vec(U_ω)
    @test H10' - H10 ≈ zeros(2, 2)
    @test norm(H10) ≈ π/10

    # Test wrapped call
    Us_wrap, H_wrap = unitary_geodesic(U_ω, 10, return_generator=true)
    @test Us_wrap[:, 1] ≈ operator_to_iso_vec(GATES[:I])
    @test Us_wrap[:, end] ≈ operator_to_iso_vec(U_ω)
    rotation = [exp(-im * H_wrap * t) for t ∈ range(0, 1, 10)]
    Us_test = stack(operator_to_iso_vec.(rotation), dims=2)
    @test isapprox(Us_wrap, Us_test)


    ## Group 2: √X to X (π/2 rotation)

    # Test geodesic not at identity
    U₀ = sqrt(GATES[:X])
    U₁ = GATES[:X]
    Us, H = unitary_geodesic(U₀, U₁, 10, return_generator=true)
    @test Us[:, 1] ≈ operator_to_iso_vec(U₀)
    @test Us[:, end] ≈ operator_to_iso_vec(U_ω)

    rotation = [exp(-im * H * t) * U₀ for t ∈ range(0, 1, 10)]
    Us_test = stack(operator_to_iso_vec.(rotation), dims=2)
    @test isapprox(Us, Us_test)
    Us_wrap = unitary_geodesic(U_ω, 4)
    @test Us_wrap[:, 1] ≈ operator_to_iso_vec(GATES[:I])
    @test Us_wrap[:, end] ≈ operator_to_iso_vec(U_ω)

end

@testitem "unitary trajectory initialization" begin
    using NamedTrajectories
    using PiccoloQuantumObjects 

    U_goal = GATES[:X]
    T = 10
    Δt = 0.1
    n_drives = 2
    a_bounds = ([1.0, 1.0],)

    traj = initialize_trajectory(
        U_goal, T, Δt, n_drives, a_bounds
    )

    @test traj isa NamedTrajectory
end

@testitem "quantum state trajectory initialization" begin
    using NamedTrajectories

    ψ_init = Vector{ComplexF64}([0.0, 1.0])
    ψ_goal = Vector{ComplexF64}([1.0, 0.0])

    T = 10
    Δt = 0.1
    n_drives = 2
    all_a_bounds = ([1.0, 1.0],)

    traj = initialize_trajectory(
        [ψ_goal], [ψ_init], T, Δt, n_drives, all_a_bounds
    )

    @test traj isa NamedTrajectory
end

@testitem "unitary_linear_interpolation direct" begin
    using PiccoloQuantumObjects
    U_init = GATES[:I]
    U_goal = GATES[:X]
    samples = 5
    # Direct matrix
    Ũ⃗ = TrajectoryInitialization.unitary_linear_interpolation(U_init, U_goal, samples)
    @test size(Ũ⃗, 2) == samples
    # EmbeddedOperator
    U_init_emb = EmbeddedOperator(U_init, [1,2], [2,2])
    U_goal_emb = EmbeddedOperator(U_goal, [1,2], [2,2])
    Ũ⃗2 = TrajectoryInitialization.unitary_linear_interpolation(U_init_emb.operator, U_goal_emb, samples)
    @test size(Ũ⃗2, 2) == samples
end

@testitem "initialize_unitary_trajectory geodesic=false" begin
    using PiccoloQuantumObjects
    U_init = GATES[:I]
    U_goal = GATES[:X]
    T = 4
    Ũ⃗ = TrajectoryInitialization.initialize_unitary_trajectory(U_init, U_goal, T; geodesic=false)
    @test size(Ũ⃗, 2) == T
end

@testitem "initialize_control_trajectory with a, Δt, n_derivatives" begin
    n_drives = 2
    T = 5
    n_derivatives = 2
    a = randn(n_drives, T)
    Δt = fill(0.1, T)
    controls = TrajectoryInitialization.initialize_control_trajectory(a, Δt, n_derivatives)
    @test length(controls) == n_derivatives + 1
    @test size(controls[1]) == (n_drives, T)
    # Real Δt version
    controls2 = TrajectoryInitialization.initialize_control_trajectory(a, 0.1, n_derivatives)
    @test length(controls2) == n_derivatives + 1
end

@testitem "initialize_trajectory with bound_state and zero_initial_and_final_derivative" begin
    using NamedTrajectories: NamedTrajectory
    state_data = [rand(2, 4)]
    state_inits = [rand(2)]
    state_goals = [rand(2)]
    state_names = [:x]
    T = 4
    Δt = 0.1
    n_drives = 1
    control_bounds = ([1.0], [1.0])
    traj = TrajectoryInitialization.initialize_trajectory(
        state_data, state_inits, state_goals, state_names, T, Δt, n_drives, control_bounds;
        bound_state=true, zero_initial_and_final_derivative=true
    )
    @test traj isa NamedTrajectory
end

@testitem "initialize_trajectory error branches" begin
    state_data = [rand(2, 4)]
    state_inits = [rand(2)]
    state_goals = [rand(2)]
    state_names = [:x]
    T = 4
    Δt = 0.1
    n_drives = 1
    control_bounds = ([1.0], [1.0])
    # state_names not unique
    @test_throws AssertionError TrajectoryInitialization.initialize_trajectory(
        state_data, state_inits, state_goals, [:x, :x], T, Δt, n_drives, control_bounds
    )
    # control_bounds wrong length
    @test_throws AssertionError TrajectoryInitialization.initialize_trajectory(
        state_data, state_inits, state_goals, state_names, T, Δt, n_drives, ([1.0],); n_control_derivatives=1
    )
    # bounds wrong type
    @test_throws MethodError TrajectoryInitialization.initialize_control_trajectory(
        n_drives, 2, T, "notabounds", 0.1
    )
end

@testitem "linear_interpolation for matrices" begin
    X = [1.0 2.0; 3.0 4.0]
    Y = [5.0 6.0; 7.0 8.0]
    n = 3
    result = linear_interpolation(X, Y, n)
    @test size(result) == (2, 2 * n)
    @test result[:, 1:2] ≈ X
    @test result[:, 5:6] ≈ Y
    @test result[:, 3:4] ≈ (X + Y) / 2
end

@testitem "unitary_trajectory convenience function" begin
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
    
    # Test with default parameters (identity to identity)
    U_goal = GATES[:I]
    qtraj = unitary_trajectory(sys, U_goal, N)
    @test qtraj isa UnitaryTrajectory
    @test size(qtraj[:Ũ⃗], 2) == N
    @test size(qtraj[:u], 2) == N
    @test size(qtraj[:u], 1) == 2  # 2 drives
    @test system(qtraj) === sys
    @test goal(qtraj) === U_goal
    @test state_name(qtraj) == :Ũ⃗
    @test control_name(qtraj) == :u
    
    # Test with custom initial and goal unitaries
    U_init = GATES[:I]
    U_goal = GATES[:X]
    qtraj2 = unitary_trajectory(sys, U_goal, N; U_init=U_init)
    @test qtraj2 isa UnitaryTrajectory
    @test size(qtraj2[:Ũ⃗], 2) == N
    
    # Test with fixed time (free_time=false)
    qtraj3 = unitary_trajectory(sys, U_goal, N; free_time=false)
    @test qtraj3 isa UnitaryTrajectory
    # Check that Δt bounds are equal (fixed timestep)
    Δt_val = sys.T_max / (N - 1)
    @test qtraj3.bounds[:Δt][1][1] == Δt_val
    @test qtraj3.bounds[:Δt][2][1] == Δt_val
    
    # Test with custom Δt bounds
    qtraj4 = unitary_trajectory(sys, U_goal, N; Δt_min=0.05, Δt_max=0.2)
    @test qtraj4 isa UnitaryTrajectory
    @test qtraj4.bounds[:Δt][1][1] == 0.05
    @test qtraj4.bounds[:Δt][2][1] == 0.2
    
    # Test with store_times=true
    qtraj5 = unitary_trajectory(sys, U_goal, N; store_times=true)
    @test qtraj5 isa UnitaryTrajectory
    @test haskey(qtraj5.components, :t)
    @test size(qtraj5[:t], 2) == N
    @test qtraj5[:t][1] ≈ 0.0
    @test qtraj5.initial[:t][1] ≈ 0.0
end

@testitem "ket_trajectory convenience function" begin
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
    
    # Test with specified initial and goal states (single state)
    qtraj = ket_trajectory(sys, ψ_init, ψ_goal, N)
    @test qtraj isa KetTrajectory
    @test size(qtraj[:ψ̃], 2) == N
    @test size(qtraj[:u], 2) == N
    @test size(qtraj[:u], 1) == 2  # 2 drives
    @test system(qtraj) === sys
    @test goal(qtraj) == ψ_goal
    @test state_name(qtraj) == :ψ̃
    @test control_name(qtraj) == :u
    
    # Test with fixed time
    qtraj3 = ket_trajectory(sys, ψ_init, ψ_goal, N; free_time=false)
    @test qtraj3 isa KetTrajectory
    Δt_val = sys.T_max / (N - 1)
    @test qtraj3.bounds[:Δt][1][1] == Δt_val
    @test qtraj3.bounds[:Δt][2][1] == Δt_val
    
    # Test with custom Δt bounds
    qtraj4 = ket_trajectory(sys, ψ_init, ψ_goal, N; Δt_min=0.05, Δt_max=0.2)
    @test qtraj4 isa KetTrajectory
    @test qtraj4.bounds[:Δt][1][1] == 0.05
    @test qtraj4.bounds[:Δt][2][1] == 0.2
    
    # Test with multiple states
    ψ2_init = ComplexF64[0.0, 1.0]
    ψ2_goal = ComplexF64[1.0, 0.0]
    qtraj5 = ket_trajectory(sys, [ψ_init, ψ2_init], [ψ_goal, ψ2_goal], N)
    @test qtraj5 isa KetTrajectory
    @test size(qtraj5[:ψ̃1], 2) == N
    @test size(qtraj5[:ψ̃2], 2) == N
    @test size(qtraj5[:u], 2) == N
    @test goal(qtraj5) == [ψ_goal, ψ2_goal]  # Multiple goals
    
    # Test with custom state names
    qtraj6 = ket_trajectory(sys, [ψ_init, ψ2_init], [ψ_goal, ψ2_goal], N;
        state_names=[:ψ̃_a, :ψ̃_b]
    )
    @test qtraj6 isa KetTrajectory
    @test size(qtraj6[:ψ̃_a], 2) == N
    @test size(qtraj6[:ψ̃_b], 2) == N
    @test state_name(qtraj6) == :ψ̃_a  # First state name
    
    # Test with store_times=true
    qtraj7 = ket_trajectory(sys, ψ_init, ψ_goal, N; store_times=true)
    @test qtraj7 isa KetTrajectory
    @test haskey(qtraj7.components, :t)
    @test size(qtraj7[:t], 2) == N
    @test qtraj7[:t][1] ≈ 0.0
    @test qtraj7.initial[:t][1] ≈ 0.0
end

@testitem "density_trajectory convenience function" begin
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
    
    # Test with specified initial and goal states
    qtraj = density_trajectory(sys, ρ_init, ρ_goal, N)
    @test qtraj isa DensityTrajectory
    @test size(qtraj[:ρ⃗̃], 2) == N
    @test size(qtraj[:u], 2) == N
    @test size(qtraj[:u], 1) == 2  # 2 drives
    @test system(qtraj) === sys
    @test goal(qtraj) == ρ_goal
    @test state_name(qtraj) == :ρ⃗̃
    @test control_name(qtraj) == :u
    
    # Test with fixed time
    qtraj3 = density_trajectory(sys, ρ_init, ρ_goal, N; free_time=false)
    @test qtraj3 isa DensityTrajectory
    Δt_val = sys.T_max / (N - 1)
    @test qtraj3.bounds[:Δt][1][1] == Δt_val
    @test qtraj3.bounds[:Δt][2][1] == Δt_val
    
    # Test with custom Δt bounds
    qtraj4 = density_trajectory(sys, ρ_init, ρ_goal, N; Δt_min=0.05, Δt_max=0.2)
    @test qtraj4 isa DensityTrajectory
    @test qtraj4.bounds[:Δt][1][1] == 0.05
    @test qtraj4.bounds[:Δt][2][1] == 0.2
    
    # Test with store_times=true
    qtraj5 = density_trajectory(sys, ρ_init, ρ_goal, N; store_times=true)
    @test qtraj5 isa DensityTrajectory
    @test haskey(qtraj5.components, :t)
    @test size(qtraj5[:t], 2) == N
    @test qtraj5[:t][1] ≈ 0.0
    @test qtraj5.initial[:t][1] ≈ 0.0
end


end
