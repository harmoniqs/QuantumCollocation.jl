export UnitarySmoothPulseProblem


@doc raw"""
    UnitarySmoothPulseProblem(system::AbstractQuantumSystem, operator::AbstractPiccoloOperator, N::Int; kwargs...)
    UnitarySmoothPulseProblem(H_drift, H_drives, operator, N; kwargs...)

Construct a `DirectTrajOptProblem` for a free-time unitary gate problem with smooth control pulses enforced by constraining the second derivative of the pulse trajectory, i.e.,

```math
\begin{aligned}
\underset{\vec{\tilde{U}}, a, \dot{a}, \ddot{a}, \Delta t}{\text{minimize}} & \quad
Q \cdot \ell\qty(\vec{\tilde{U}}_N, \vec{\tilde{U}}_{\text{goal}}) + \frac{1}{2} \sum_t \qty(R_a a_t^2 + R_{\dot{a}} \dot{a}_t^2 + R_{\ddot{a}} \ddot{a}_t^2) \\
\text{ subject to } & \quad \vb{P}^{(n)}\qty(\vec{\tilde{U}}_{t+1}, \vec{\tilde{U}}_t, a_t, \Delta t_t) = 0 \\
& \quad a_{t+1} - a_t - \dot{a}_t \Delta t_t = 0 \\
& \quad \dot{a}_{t+1} - \dot{a}_t - \ddot{a}_t \Delta t_t = 0 \\
& \quad |a_t| \leq a_{\text{bound}} \\
& \quad |\ddot{a}_t| \leq \ddot{a}_{\text{bound}} \\
& \quad \Delta t_{\text{min}} \leq \Delta t_t \leq \Delta t_{\text{max}} \\
\end{aligned}
```

where, for $U \in SU(N)$,

```math
\ell\qty(\vec{\tilde{U}}_N, \vec{\tilde{U}}_{\text{goal}}) =
\abs{1 - \frac{1}{N} \abs{ \tr \qty(U_{\text{goal}}, U_N)} }
```

is the *infidelity* objective function, $Q$ is a weight, $R_a$, $R_{\dot{a}}$, and $R_{\ddot{a}}$ are weights on the regularization terms, and $\vb{P}^{(n)}$ is the $n$th-order Pade integrator.

# Arguments

- `system::AbstractQuantumSystem`: the system to be controlled
or
- `H_drift::AbstractMatrix{<:Number}`: the drift hamiltonian
- `H_drives::Vector{<:AbstractMatrix{<:Number}}`: the control hamiltonians
with
- `goal::AbstractPiccoloOperator`: the target unitary, either in the form of an `EmbeddedOperator` or a `Matrix{ComplexF64}
- `N::Int`: the number of knot points
- `Δt::Float64`: the (initial) time step size

# Keyword Arguments
- `piccolo_options::PiccoloOptions=PiccoloOptions()`: the options for the Piccolo solver
- `state_name::Symbol = :Ũ⃗`: the name of the state
- `control_name::Symbol = :a`: the name of the control
- `timestep_name::Symbol = :Δt`: the name of the timestep
- `init_trajectory::Union{NamedTrajectory, Nothing}=nothing`: an initial trajectory to use
- `a_guess::Union{Matrix{Float64}, Nothing}=nothing`: an initial guess for the control pulses
- `a_bound::Float64=1.0`: the bound on the control pulse
- `a_bounds=fill(a_bound, length(system.G_drives))`: the bounds on the control pulses, one for each drive
- `da_bound::Float64=Inf`: the bound on the control pulse derivative
- `da_bounds=fill(da_bound, length(system.G_drives))`: the bounds on the control pulse derivatives, one for each drive
- `dda_bound::Float64=1.0`: the bound on the control pulse second derivative
- `dda_bounds=fill(dda_bound, length(system.G_drives))`: the bounds on the control pulse second derivatives, one for each drive
- `Δt_min::Float64=Δt isa Float64 ? 0.5 * Δt : 0.5 * mean(Δt)`: the minimum time step size
- `Δt_max::Float64=Δt isa Float64 ? 1.5 * Δt : 1.5 * mean(Δt)`: the maximum time step size
- `Q::Float64=100.0`: the weight on the infidelity objective
- `R=1e-2`: the weight on the regularization terms
- `R_a::Union{Float64, Vector{Float64}}=R`: the weight on the regularization term for the control pulses
- `R_da::Union{Float64, Vector{Float64}}=R`: the weight on the regularization term for the control pulse derivatives
- `R_dda::Union{Float64, Vector{Float64}}=R`: the weight on the regularization term for the control pulse second derivatives
- `constraints::Vector{<:AbstractConstraint}=AbstractConstraint[]`: the constraints to enforce

"""
function UnitarySmoothPulseProblem end

function UnitarySmoothPulseProblem(
    system::AbstractQuantumSystem,
    goal::AbstractPiccoloOperator,
    N::Int;
    unitary_integrator=UnitaryIntegrator,
    state_name::Symbol = :Ũ⃗,
    control_name::Symbol = :a,
    timestep_name::Symbol = :Δt,
    init_trajectory::Union{NamedTrajectory, Nothing}=nothing,
    a_guess::Union{Matrix{Float64}, Nothing}=nothing,
    da_bound::Float64=Inf,
    da_bounds=fill(da_bound, system.n_drives),
    dda_bound::Float64=1.0,
    dda_bounds=fill(dda_bound, system.n_drives),
    Δt_min::Float64=0.5 * minimum(Δt),
    Δt_max::Float64=2.0 * maximum(Δt),
    Q::Float64=100.0,
    R=1e-2,
    R_a::Union{Float64, Vector{Float64}}=R,
    R_da::Union{Float64, Vector{Float64}}=R,
    R_dda::Union{Float64, Vector{Float64}}=R,
    constraints::Vector{<:AbstractConstraint}=AbstractConstraint[],
    piccolo_options::PiccoloOptions=PiccoloOptions(),
)
    if piccolo_options.verbose
        println("    constructing UnitarySmoothPulseProblem...")
        println("\tusing integrator: $(typeof(unitary_integrator))")
    end

    # Trajectory
    if !isnothing(init_trajectory)
        traj = init_trajectory
    else
        a_bounds = (
            [system.drive_bounds[j][1] for j in 1:length(system.drive_bounds)], 
            [system.drive_bounds[j][2] for j in 1:length(system.drive_bounds)]
        )
        traj = initialize_trajectory(
            goal,
            N,
            system.T_max / N,
            system.n_drives,
            (a_bounds, da_bounds, dda_bounds);
            state_name=state_name,
            control_name=control_name,
            timestep_name=timestep_name,
            Δt_bounds=(Δt_min, Δt_max),
            zero_initial_and_final_derivative=piccolo_options.zero_initial_and_final_derivative,
            geodesic=piccolo_options.geodesic,
            bound_state=piccolo_options.bound_state,
            a_guess=a_guess,
            system=system,
            rollout_integrator=piccolo_options.rollout_integrator,
            verbose=piccolo_options.verbose
        )
    end

    # Objective
    J = UnitaryInfidelityObjective(goal, state_name, traj; Q=Q)

    control_names = [
        name for name ∈ traj.names
            if endswith(string(name), string(control_name))
    ]

    J += QuadraticRegularizer(control_names[1], traj, R_a)
    J += QuadraticRegularizer(control_names[2], traj, R_da)
    J += QuadraticRegularizer(control_names[3], traj, R_dda)

    # Optional Piccolo constraints and objectives
    J += apply_piccolo_options!(
        piccolo_options, constraints, traj; 
        state_names=state_name,
        state_leakage_indices=goal isa EmbeddedOperator ? 
            get_iso_vec_leakage_indices(goal) : 
            nothing
    )

    integrators = [
        unitary_integrator(system, traj, state_name, control_name),
        DerivativeIntegrator(traj, control_name, control_names[2]),
        DerivativeIntegrator(traj, control_names[2], control_names[3]),
    ]

    # TODO: see if using a linear constraint here is the right choice
    # derivative_constraints = [
        # DerivativeConstraint(control_name, control_names[2], traj),
        # DerivativeConstraint(control_names[2], control_names[3], traj),
    # ]

    # constraints = vcat(constraints, derivative_constraints)

    return DirectTrajOptProblem(
        traj,
        J,
        integrators;
        constraints=constraints
    )
end

function UnitarySmoothPulseProblem(
    H_drift::AbstractMatrix{<:Number},
    H_drives::Vector{<:AbstractMatrix{<:Number}},
    args...;
    kwargs...
)
    system = QuantumSystem(H_drift, H_drives)
    return UnitarySmoothPulseProblem(system, args...; kwargs...)
end

# *************************************************************************** #

@testitem "Hadamard gate improvement" begin
    using PiccoloQuantumObjects 

    sys = QuantumSystem(GATES[:Z], [GATES[:X], GATES[:Y]])
    U_goal = GATES[:H]
    T = 51
    Δt = 0.2

    prob = UnitarySmoothPulseProblem(
        sys, U_goal, T, Δt;
        da_bound=1.0,
        piccolo_options=PiccoloOptions(verbose=false)
    )

    initial = unitary_rollout_fidelity(prob.trajectory, sys)
    solve!(prob, max_iter=100, verbose=false, print_level=1)
    @test unitary_rollout_fidelity(prob.trajectory, sys) > initial
end

@testitem "Bound states and control norm constraint" begin
    using PiccoloQuantumObjects 

    sys = QuantumSystem(GATES[:Z], [GATES[:X], GATES[:Y]])
    U_goal = GATES[:H]
    T = 51
    Δt = 0.2

    prob = UnitarySmoothPulseProblem(
        sys, U_goal, T, Δt,
        piccolo_options=PiccoloOptions(
            verbose=false,
            bound_state=true,
            complex_control_norm_constraint_name=:a
        )
    )

    initial = copy(prob.trajectory.data)
    solve!(prob, max_iter=5, verbose=false, print_level=1)
    @test prob.trajectory.data != initial
end

@testitem "EmbeddedOperator tests" begin
    using PiccoloQuantumObjects 

    a = annihilate(3)
    sys = QuantumSystem([(a + a')/2, (a - a')/(2im)])
    U_goal = EmbeddedOperator(GATES[:H], sys)
    T = 51
    Δt = 0.2

    @testset "EmbeddedOperator: solve gate" begin
        prob = UnitarySmoothPulseProblem(
            sys, U_goal, T, Δt,
            piccolo_options=PiccoloOptions(verbose=false)
        )

        initial = copy(prob.trajectory.data)
        solve!(prob, max_iter=5, verbose=false, print_level=1)
        @test prob.trajectory.data != initial
    end

    @testset "EmbeddedOperator: leakage constraint" begin
        prob = UnitarySmoothPulseProblem(
            sys, U_goal, T, Δt;
            da_bound=1.0,
            piccolo_options=PiccoloOptions(
                leakage_constraint=true, 
                leakage_constraint_value=5e-2, 
                leakage_cost=1e-1,
                verbose=false
            )
        )
        initial = copy(prob.trajectory.data)
        solve!(prob, max_iter=5, verbose=false, print_level=1)
        @test prob.trajectory.data != initial
    end
end

# TODO: Test changing names of control, state, and timestep
