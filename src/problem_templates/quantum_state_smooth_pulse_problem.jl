export QuantumStateSmoothPulseProblem


"""
    QuantumStateSmoothPulseProblem(system, ψ_inits, ψ_goals, T, Δt; kwargs...)
    QuantumStateSmoothPulseProblem(system, ψ_init, ψ_goal, T, Δt; kwargs...)
    QuantumStateSmoothPulseProblem(H_drift, H_drives, args...; kwargs...)

Create a quantum state smooth pulse problem. The goal is to find a control pulse `a(t)` 
that drives all of the initial states `ψ_inits` to the corresponding target states 
`ψ_goals` using `T` timesteps of size `Δt`. This problem also controls the  first and 
second derivatives of the control pulse, `da(t)` and `dda(t)`, to ensure smoothness.

# Arguments
- `system::AbstractQuantumSystem`: The quantum system.
or
- `H_drift::AbstractMatrix{<:Number}`: The drift Hamiltonian.
- `H_drives::Vector{<:AbstractMatrix{<:Number}}`: The control Hamiltonians.
with
- `ψ_inits::Vector{<:AbstractVector{<:ComplexF64}}`: The initial states.
- `ψ_goals::Vector{<:AbstractVector{<:ComplexF64}}`: The target states.
or
- `ψ_init::AbstractVector{<:ComplexF64}`: The initial state.
- `ψ_goal::AbstractVector{<:ComplexF64}`: The target state.
with
- `T::Int`: The number of timesteps.
- `Δt::Float64`: The timestep size.


# Keyword Arguments
- `state_name::Symbol=:ψ̃`: The name of the state variable.
- `control_name::Symbol=:a`: The name of the control variable.
- `timestep_name::Symbol=:Δt`: The name of the timestep variable.
- `init_trajectory::Union{NamedTrajectory, Nothing}=nothing`: The initial trajectory.
- `a_bound::Float64=1.0`: The bound on the control pulse.
- `a_bounds::Vector{Float64}=fill(a_bound, length(system.G_drives))`: The bounds on the control pulse.
- `a_guess::Union{Matrix{Float64}, Nothing}=nothing`: The initial guess for the control pulse.
- `da_bound::Float64=Inf`: The bound on the first derivative of the control pulse.
- `da_bounds::Vector{Float64}=fill(da_bound, length(system.G_drives))`: The bounds on the first derivative of the control pulse.
- `dda_bound::Float64=1.0`: The bound on the second derivative of the control pulse.
- `dda_bounds::Vector{Float64}=fill(dda_bound, length(system.G_drives))`: The bounds on the second derivative of the control pulse.
- `Δt_min::Float64=0.5 * Δt`: The minimum timestep size.
- `Δt_max::Float64=1.5 * Δt`: The maximum timestep size.
- `drive_derivative_σ::Float64=0.01`: The standard deviation of the drive derivative random initialization.
- `Q::Float64=100.0`: The weight on the state objective.
- `R=1e-2`: The weight on the control pulse and its derivatives.
- `R_a::Union{Float64, Vector{Float64}}=R`: The weight on the control pulse.
- `R_da::Union{Float64, Vector{Float64}}=R`: The weight on the first derivative of the control pulse.
- `R_dda::Union{Float64, Vector{Float64}}=R`: The weight on the second derivative of the control pulse.
- `constraints::Vector{<:AbstractConstraint}=AbstractConstraint[]`: The constraints.
- `piccolo_options::PiccoloOptions=PiccoloOptions()`: The Piccolo options.
"""
function QuantumStateSmoothPulseProblem end

function QuantumStateSmoothPulseProblem(
    sys::AbstractQuantumSystem,
    ψ_inits::Vector{<:AbstractVector{<:ComplexF64}},
    ψ_goals::Vector{<:AbstractVector{<:ComplexF64}},
    T::Int,
    Δt::Union{Float64, <:AbstractVector{Float64}};
    ket_integrator=KetIntegrator,
    state_name::Symbol=:ψ̃,
    control_name::Symbol=:a,
    timestep_name::Symbol=:Δt,
    init_trajectory::Union{NamedTrajectory, Nothing}=nothing,
    a_bound::Float64=1.0,
    a_bounds::Vector{Float64}=fill(a_bound, sys.n_drives),
    a_guess::Union{Matrix{Float64}, Nothing}=nothing,
    da_bound::Float64=Inf,
    da_bounds::Vector{Float64}=fill(da_bound, sys.n_drives),
    dda_bound::Float64=1.0,
    dda_bounds::Vector{Float64}=fill(dda_bound, sys.n_drives),
    Δt_min::Float64=0.001 * Δt,
    Δt_max::Float64=2.0 * Δt,
    Q::Float64=100.0,
    R=1e-2,
    R_a::Union{Float64, Vector{Float64}}=R,
    R_da::Union{Float64, Vector{Float64}}=R,
    R_dda::Union{Float64, Vector{Float64}}=R,
    constraints::Vector{<:AbstractConstraint}=AbstractConstraint[],
    piccolo_options::PiccoloOptions=PiccoloOptions(),
)
    @assert length(ψ_inits) == length(ψ_goals)

    if piccolo_options.verbose
        println("    constructing QuantumStateSmoothPulseProblem...")
        println("\tusing integrator: $(typeof(ket_integrator))")
        println("\tusing $(length(ψ_inits)) initial state(s)")
    end

    # Trajectory
    if !isnothing(init_trajectory)
        traj = init_trajectory
    else
        traj = initialize_trajectory(
            ψ_goals,
            ψ_inits,
            T,
            Δt,
            sys.n_drives,
            (a_bounds, da_bounds, dda_bounds);
            state_name=state_name,
            control_name=control_name,
            timestep_name=timestep_name,
            zero_initial_and_final_derivative=piccolo_options.zero_initial_and_final_derivative,
            Δt_bounds=(Δt_min, Δt_max),
            bound_state=piccolo_options.bound_state,
            a_guess=a_guess,
            system=sys,
            rollout_integrator=piccolo_options.rollout_integrator,
        )
    end

    state_names = [
        name for name ∈ traj.names
            if startswith(string(name), string(state_name))
    ]
    @assert length(state_names) == length(ψ_inits) "Number of states must match number of initial states"

    control_names = [
        name for name ∈ traj.names
            if endswith(string(name), string(control_name))
    ]

    # Objective
    J = QuadraticRegularizer(control_names[1], traj, R_a)
    J += QuadraticRegularizer(control_names[2], traj, R_da)
    J += QuadraticRegularizer(control_names[3], traj, R_dda)

    for name ∈ state_names
        J += KetInfidelityLoss(name, traj; Q=Q)
    end

    # Optional Piccolo constraints and objectives
    apply_piccolo_options!(
        J, constraints, piccolo_options, traj, state_name, timestep_name;
        state_leakage_indices=piccolo_options.state_leakage_indices
    )

    state_names = [
        name for name ∈ traj.names
            if startswith(string(name), string(state_name))
    ]

    state_integrators = []

    for name ∈ state_names
        push!(
            state_integrators, 
            ket_integrator(sys, traj, name, control_name)
        )
    end

    integrators = [
        state_integrators...,
        DerivativeIntegrator(traj, control_name, control_names[2]),
        DerivativeIntegrator(traj, control_names[2], control_names[3])
    ]

    return DirectTrajOptProblem(
        traj,
        J,
        integrators;
        constraints=constraints,
    )
end

function QuantumStateSmoothPulseProblem(
    system::AbstractQuantumSystem,
    ψ_init::AbstractVector{<:ComplexF64},
    ψ_goal::AbstractVector{<:ComplexF64},
    args...;
    kwargs...
)
    return QuantumStateSmoothPulseProblem(system, [ψ_init], [ψ_goal], args...; kwargs...)
end

function QuantumStateSmoothPulseProblem(
    H_drift::AbstractMatrix{<:Number},
    H_drives::Vector{<:AbstractMatrix{<:Number}},
    args...;
    kwargs...
)
    system = QuantumSystem(H_drift, H_drives)
    return QuantumStateSmoothPulseProblem(system, args...; kwargs...)
end

# *************************************************************************** #

@testitem "Test quantum state smooth pulse" begin
    using PiccoloQuantumObjects 

    T = 51
    Δt = 0.2
    sys = QuantumSystem(0.1 * GATES[:Z], [GATES[:X], GATES[:Y]])
    ψ_init = Vector{ComplexF64}([1.0, 0.0])
    ψ_target = Vector{ComplexF64}([0.0, 1.0])
    
    # Single initial and target states
    # --------------------------------
    prob = QuantumStateSmoothPulseProblem(
        sys, ψ_init, ψ_target, T, Δt;
        piccolo_options=PiccoloOptions(verbose=false)
    )
    initial = rollout_fidelity(prob.trajectory, sys)
    solve!(prob, max_iter=50, print_level=1, verbose=false)
    final = rollout_fidelity(prob.trajectory, sys)
    @test final > initial
end

@testitem "Test multiple quantum states smooth pulse" begin
    using PiccoloQuantumObjects 

    T = 50
    Δt = 0.2
    sys = QuantumSystem(0.1 * GATES[:Z], [GATES[:X], GATES[:Y]])
    ψ_inits = Vector{ComplexF64}.([[1.0, 0.0], [0.0, 1.0]])
    ψ_targets = Vector{ComplexF64}.([[0.0, 1.0], [1.0, 0.0]])

    prob = QuantumStateSmoothPulseProblem(
        sys, ψ_inits, ψ_targets, T, Δt;
        piccolo_options=PiccoloOptions(verbose=false)
    )
    initial = rollout_fidelity(prob.trajectory, sys)
    solve!(prob, max_iter=50, print_level=1, verbose=false)
    final = rollout_fidelity(prob.trajectory, sys)
    final, initial
    @test all(final .> initial)
end
