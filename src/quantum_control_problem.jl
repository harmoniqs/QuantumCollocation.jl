module QuantumControlProblems

using DirectTrajOpt
using NamedTrajectories
using PiccoloQuantumObjects

import PiccoloQuantumObjects: get_trajectory, get_system, get_goal, get_state_name, get_control_name
import DirectTrajOpt.Solvers: solve!

export QuantumControlProblem
export get_trajectory, get_system, get_goal, get_state_name, get_control_name
export solve!
# Note: solve! is NOT exported to avoid ambiguity with SciMLBase.solve!
# Users should use: using DirectTrajOpt (to get solve!)

"""
    QuantumControlProblem{QT<:AbstractQuantumTrajectory}

Wrapper combining quantum trajectory information with trajectory optimization problem.

This type enables:
- Type-stable dispatch on quantum trajectory type (Unitary, Ket, Density)
- Clean separation of quantum information (system, goal) from optimization details
- Composable problem transformations (e.g., SmoothPulseProblem → MinimumTimeProblem)

# Fields
- `qtraj::QT`: Quantum trajectory containing system, goal, and quantum state information
- `prob::DirectTrajOptProblem`: Direct trajectory optimization problem with objective, dynamics, constraints

# Construction
Typically created via problem templates:
```julia
qtraj = UnitaryTrajectory(sys, U_goal, N)
qcp = SmoothPulseProblem(qtraj; Q=100.0, R=1e-2)
```

# Accessors
- `get_trajectory(qcp)`: Get the NamedTrajectory
- `get_system(qcp)`: Get the QuantumSystem
- `get_goal(qcp)`: Get the goal state/unitary
- `get_state_name(qcp)`: Get the state variable name
- `get_control_name(qcp)`: Get the control variable name

# Solving
```julia
solve!(qcp; max_iter=100, verbose=true)
```
"""
struct QuantumControlProblem{QT<:AbstractQuantumTrajectory}
    qtraj::QT
    prob::DirectTrajOptProblem
end

# ============================================================================= #
# Convenience accessors - extend PiccoloQuantumObjects methods
# ============================================================================= #

"""
    get_trajectory(qcp::QuantumControlProblem)

Get the NamedTrajectory from the optimization problem.
"""
get_trajectory(qcp::QuantumControlProblem) = qcp.prob.trajectory

"""
    get_system(qcp::QuantumControlProblem)

Get the QuantumSystem from the quantum trajectory.
"""
get_system(qcp::QuantumControlProblem) = get_system(qcp.qtraj)

"""
    get_goal(qcp::QuantumControlProblem)

Get the goal state/operator from the quantum trajectory.
"""
get_goal(qcp::QuantumControlProblem) = get_goal(qcp.qtraj)

"""
    get_state_name(qcp::QuantumControlProblem)

Get the state variable name from the quantum trajectory.
"""
get_state_name(qcp::QuantumControlProblem) = get_state_name(qcp.qtraj)

"""
    get_control_name(qcp::QuantumControlProblem)

Get the control variable name from the quantum trajectory.
"""
get_control_name(qcp::QuantumControlProblem) = get_control_name(qcp.qtraj)

# ============================================================================= #
# Forward DirectTrajOptProblem methods
# ============================================================================= #

"""
    solve!(qcp::QuantumControlProblem; kwargs...)

Solve the quantum control problem by forwarding to the inner DirectTrajOptProblem.

All keyword arguments are passed to the DirectTrajOpt solver.
"""
solve!(qcp::QuantumControlProblem; kwargs...) = solve!(qcp.prob; kwargs...)
# TODO: make sure qtraj is updated after solve!

# Forward other common DirectTrajOptProblem accessors
Base.getproperty(qcp::QuantumControlProblem, s::Symbol) = begin
    if s === :qtraj
        getfield(qcp, :qtraj)
    elseif s === :prob
        getfield(qcp, :prob)
    # Forward to prob for common fields
    elseif s in (:objective, :dynamics, :constraints, :trajectory)
        getproperty(qcp.prob, s)
    else
        # Fall back to default behavior
        getfield(qcp, s)
    end
end

# ============================================================================= #
# Display
# ============================================================================= #

function Base.show(io::IO, qcp::QuantumControlProblem{QT}) where {QT}
    println(io, "QuantumControlProblem{$QT}")
    println(io, "  System: $(typeof(get_system(qcp)))")
    println(io, "  Goal: $(typeof(get_goal(qcp)))")
    println(io, "  Trajectory: $(qcp.prob.trajectory.N) knots")
    println(io, "  State: $(get_state_name(qcp))")
    print(io, "  Controls: $(get_control_name(qcp))")
end

end
