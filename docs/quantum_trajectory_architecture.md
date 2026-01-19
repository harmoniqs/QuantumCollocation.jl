# Quantum Trajectory Architecture

## Overview

We've introduced `AbstractQuantumTrajectory` types that wrap `NamedTrajectory` with quantum-specific metadata. This enables type-based dispatch for problem templates and cleaner code organization.

## Type Hierarchy

```julia
abstract type AbstractQuantumTrajectory end

struct UnitaryTrajectory <: AbstractQuantumTrajectory
    trajectory::NamedTrajectory
    system::QuantumSystem
    state_name::Symbol    # :Ũ⃗
    control_name::Symbol  # :u
    goal::AbstractPiccoloOperator
end

struct KetTrajectory <: AbstractQuantumTrajectory
    trajectory::NamedTrajectory
    system::QuantumSystem
    state_name::Symbol    # :ψ̃
    control_name::Symbol  # :u
    goals::Vector{<:AbstractVector{ComplexF64}}
end

struct DensityTrajectory <: AbstractQuantumTrajectory
    trajectory::NamedTrajectory
    system::OpenQuantumSystem
    state_name::Symbol    # :ρ⃗̃
    control_name::Symbol  # :u
    goal::AbstractMatrix
end
```

## Usage Example

```julia
using QuantumCollocation
using PiccoloQuantumObjects

# Step 1: Create quantum system
sys = QuantumSystem(GATES[:Z], [GATES[:X], GATES[:Y]], [1.0, 1.0])

# Step 2: Create quantum trajectory (returns wrapped type)
qtraj = unitary_trajectory(sys, GATES[:H], 51; store_times=true)
# Returns: UnitaryTrajectory

# Access underlying data
qtraj[:Ũ⃗]  # state data
qtraj[:u]   # control data

# Access metadata
get_system(qtraj)       # the QuantumSystem
get_goal(qtraj)         # target operator
get_state_name(qtraj)   # :Ũ⃗
get_control_name(qtraj) # :u

# Convenience accessors
state(qtraj)     # qtraj[:Ũ⃗]
controls(qtraj)  # qtraj[:u]
```

## Benefits

1. **Type Safety**: Trajectory type determines which problem templates and integrators apply
2. **Cleaner API**: Metadata travels with trajectory, no need to pass separately
3. **Delegation**: Common operations delegate to underlying `NamedTrajectory`
4. **Multiple Dispatch**: Problem templates can dispatch on quantum trajectory type

## Next Steps

### 1. Integrator Partial Construction

Create integrators that can be configured without being bound to a trajectory:

```julia
# Configure integrator with kwargs
int_spec = UnitarySplineIntegrator(spline_order=3, tol=1e-8)

# Later, bind to trajectory
int_bound = int_spec(qtraj)  # or int_spec(sys, traj, :Ũ⃗, :u)
```

### 2. SmoothPulseProblem Template

Single problem template that works for all quantum trajectory types:

```julia
# Dispatches on UnitaryTrajectory
prob = SmoothPulseProblem(
    qtraj::UnitaryTrajectory;
    integrator=UnitarySplineIntegrator(spline_order=3, tol=1e-8),
    Q=100.0,
    R=1e-2
)

# Internally:
# - Adds derivatives to trajectory
# - Creates appropriate objective (UnitaryInfidelityObjective)
# - Instantiates integrators
# - Assembles DirectTrajOptProblem
```

### 3. Trajectory Augmentation

Add derivatives within problem template:

```julia
function SmoothPulseProblem(qtraj::AbstractQuantumTrajectory; kwargs...)
    # Add control derivatives
    traj_smooth = add_derivatives(qtraj.trajectory, :u, 2; bounds=...)
    
    # Create objectives, integrators, etc.
    ...
end
```

## Implementation Status

✅ Created `AbstractQuantumTrajectory` type hierarchy  
✅ Updated trajectory creators to return wrapped types  
✅ Implemented delegation to underlying `NamedTrajectory`  
✅ Added accessor functions  
✅ Updated tests

⏳ Next: Integrator partial construction pattern  
⏳ Next: Refactor problem templates to use quantum trajectories  
⏳ Next: Implement `SmoothPulseProblem` with type dispatch
