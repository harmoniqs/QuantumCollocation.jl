export UnitaryMinimumTimeProblem


@doc raw"""
    UnitaryMinimumTimeProblem(
        goal::AbstractPiccoloOperator,
        trajectory::NamedTrajectory,
        objective::Objective,
        dynamics::TrajectoryDynamics,
        constraints::AbstractVector{<:AbstractConstraint};
        kwargs...
    )

    UnitaryMinimumTimeProblem(
        goal::AbstractPiccoloOperator,
        prob::DirectTrajOptProblem;
        kwargs...
    )

Create a minimum-time problem for unitary control.

```math
\begin{aligned}
\underset{\vec{\tilde{U}}, a, \dot{a}, \ddot{a}, \Delta t}{\text{minimize}} & \quad
J(\vec{\tilde{U}}, a, \dot{a}, \ddot{a}) + D \sum_t \Delta t_t \\
\text{ subject to } & \quad \vb{P}^{(n)}\qty(\vec{\tilde{U}}_{t+1}, \vec{\tilde{U}}_t, a_t, \Delta t_t) = 0 \\
& c(\vec{\tilde{U}}, a, \dot{a}, \ddot{a}) = 0 \\
& \quad \Delta t_{\text{min}} \leq \Delta t_t \leq \Delta t_{\text{max}} \\
\end{aligned}
```

# Keyword Arguments
- `piccolo_options::PiccoloOptions=PiccoloOptions()`: The Piccolo options.
- `unitary_name::Symbol=:Ũ⃗`: The name of the unitary for the goal.
- `final_fidelity::Float64=1.0`: The final fidelity constraint.
- `D::Float64=1.0`: The scaling factor for the minimum-time objective.
"""
function UnitaryMinimumTimeProblem end

function UnitaryMinimumTimeProblem(
    trajectory::NamedTrajectory,
    goal::AbstractPiccoloOperator,
    objective::Objective,
    dynamics::TrajectoryDynamics,
    constraints::AbstractVector{<:AbstractConstraint};
    unitary_name::Symbol=:Ũ⃗,
    final_fidelity::Float64=1.0,
    D::Float64=100.0,
    piccolo_options::PiccoloOptions=PiccoloOptions(),
)
    if piccolo_options.verbose
        println("    constructing UnitaryMinimumTimeProblem...")
        println("\tfinal fidelity: $(final_fidelity)")
    end

    objective += MinimumTimeObjective(
        trajectory; D=D, timesteps_all_equal=piccolo_options.timesteps_all_equal
    )

    fidelity_constraint = FinalUnitaryFidelityConstraint(
        goal, unitary_name, final_fidelity, trajectory
    )

    constraints = push!(constraints, fidelity_constraint)

    return DirectTrajOptProblem(
        trajectory,
        objective,
        dynamics,
        constraints
    )
end

# TODO: how to handle apply_piccolo_options?
# TODO: goal from trajectory?

function UnitaryMinimumTimeProblem(
    prob::DirectTrajOptProblem,
    goal::AbstractPiccoloOperator;
    objective::Objective=prob.objective,
    constraints::AbstractVector{<:AbstractConstraint}=deepcopy(prob.constraints),
    kwargs...
)
    return UnitaryMinimumTimeProblem(
        prob.trajectory,
        goal,
        objective,
        prob.dynamics,
        constraints;
        kwargs...
    )
end

# --------------------------------------------------------------------------- #
# Free phases
# --------------------------------------------------------------------------- #

function UnitaryMinimumTimeProblem(
    trajectory::NamedTrajectory,
    goal::Function,
    objective::Objective,
    dynamics::TrajectoryDynamics,
    constraints::AbstractVector{<:AbstractConstraint};
    unitary_name::Symbol=:Ũ⃗,
    phase_name::Symbol=:θ,
    final_fidelity::Float64=1.0,
    D::Float64=100.0,
    piccolo_options::PiccoloOptions=PiccoloOptions(),
)
    # Collect phase names
    phase_names = [
        n for n in keys(trajectory.global_data) if endswith(string(n), string(phase_name))
    ]

    if piccolo_options.verbose
        println("    constructing UnitaryMinimumTimeProblem...")
        println("\tfinal fidelity: $(final_fidelity)")
        println("\tphase names: $(phase_names)")
    end

    objective += MinimumTimeObjective(
        trajectory; D=D, timesteps_all_equal=piccolo_options.timesteps_all_equal
    )

    fidelity_constraint = FinalUnitaryFidelityConstraint(
        goal, unitary_name, phase_names, final_fidelity, trajectory
    )

    constraints = push!(constraints, fidelity_constraint)

    return DirectTrajOptProblem(
        trajectory,
        objective,
        dynamics,
        constraints
    )
end

function UnitaryMinimumTimeProblem(
    prob::DirectTrajOptProblem,
    goal::Function;
    objective::Objective=prob.objective,
    constraints::AbstractVector{<:AbstractConstraint}=deepcopy(prob.constraints),
    kwargs...
)
    return UnitaryMinimumTimeProblem(
        prob.trajectory,
        goal,
        objective,
        prob.dynamics,
        constraints;
        kwargs...
    )
end

# *************************************************************************** #

@testitem "Minimum time Hadamard gate" begin
    using NamedTrajectories
    using PiccoloQuantumObjects 

    H_drift = PAULIS[:Z]
    H_drives = [PAULIS[:X], PAULIS[:Y]]
    U_goal = GATES[:H]
    T = 51
    Δt = 0.2

    sys = QuantumSystem(H_drift, H_drives)

    prob = UnitarySmoothPulseProblem(
        sys, U_goal, T, Δt,
        piccolo_options=PiccoloOptions(verbose=false)
    )

    before = unitary_rollout_fidelity(prob.trajectory, sys)
    solve!(prob; max_iter=100, verbose=false, print_level=1)
    after = unitary_rollout_fidelity(prob.trajectory, sys)
    @test after > before

    # soft fidelity constraint
    min_prob = UnitaryMinimumTimeProblem(
        prob, U_goal,
        piccolo_options=PiccoloOptions(verbose=false)
    )
    solve!(min_prob; max_iter=100, verbose=false, print_level=1)

    # test fidelity has stayed above the constraint
    constraint_tol = 0.95
    final_fidelity = minimum([0.99, after])
    @test unitary_rollout_fidelity(min_prob.trajectory, sys) ≥ constraint_tol * final_fidelity
    duration_after = sum(get_timesteps(min_prob.trajectory))
    duration_before = sum(get_timesteps(prob.trajectory))
    @test duration_after < duration_before
end

@testitem "Test relaxed final_fidelity constraint" begin
    final_fidelity = 0.95
    @test_broken false
end
