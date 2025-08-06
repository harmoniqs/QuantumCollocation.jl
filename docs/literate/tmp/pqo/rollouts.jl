# ```@meta
# CollapsedDocStrings = true
# ```

# # Rollouts and fidelity (and more!!!)

using PiccoloQuantumObjects
using NamedTrajectories
using SparseArrays # for visualization
using LinearAlgebra

#=

Rollouts are simulations of a quantum system. In a rollout, controls are integrated forward
in time using the dynamics implied by a provided quantum systems. The defualt is to use 
zero-order hold integration to simulate the evolution---that is, the controls are held
constant between time steps. For quantum states, the Schrödinger equation is used:

```math
\psi(t + \Delta t) = \exp\left(-i H(\mathbf{a}(t)) \Delta t\right) \psi(t)
```

The visited states are collected into a matrix of size `(2n, T)`, for the isomorphic Hilbert
space dimension `2n` and timesteps `T`.  

!!! note
    All of the returned rollout are assumed to be real valued. It is helpful to use the 
    appropriate isomorphisms to convert between real and complex quantum 
    objects, and [eachcol](https://docs.julialang.org/en/v1/base/arrays/#Base.eachcol)
    to iterate over the rollout columns.

There are rollouts for each kind of quantum object:
_quantum states_, _unitary operators_, and _density operators_.

A fidelity function is also provided for each kind of quantum objectand a rollout fidelity
function compares the final state of a rollout to a goal state.

=#

#=
## Fidelity functions

The fidelity functions are used to measure how close two quantum states or operators are to
each other. 

- [`fidelity`](@ref) calculates the fidelity between two quantum states.
- [`unitary_fidelity`](@ref) calculates the fidelity between two unitary operators.


=#

# _State fidelity_.
ψ = GATES.X * [1.0, 0.0]
ψ_goal = [0.0, 1.0]
fidelity(ψ, ψ_goal)


# _Unitary fidelity of orthogonal operations._
U = GATES.Y 
U_goal = GATES.X
unitary_fidelity(U, U_goal)

#=
## Quantum State Rollouts

The `rollout` function simulates the evolution of a real valued quantum state under given
controls and quantum system. 

```@docs; canonical = false
rollout
```

```@docs; canonical = false
rollout_fidelity
```

=#

# _The rollout is a matrix of size `(2n, T)`._
T = 10
ψ_init = ComplexF64[1.0, 0.0]
controls = rand(2, T)
Δt = fill(0.1, T)
system = QuantumSystem(PAULIS[:Z], [PAULIS[:X], PAULIS[:Y]])
ψ̃_rollout = rollout(ψ_init, controls, Δt, system) 
ψ̃_rollout |> size

# ### Quantum State Rollout Fidelity

# _States must be cast to complex for the rollout to know the difference between real and isomorphic states._
ψ_goal = ComplexF64[0.0, 1.0]
rollout_fidelity(ψ_init, ψ_goal, controls, Δt, system) 

#=

!!! warning
    Don't forget to convert the quantum state to the appropriate isomorphism before
    calculating the fidelity.

=#

fidelity(iso_to_ket(ψ̃_rollout[:, end]), ψ_goal)

# _The initial state and goal are often inferred from the properly configured trajectory of a control problem._
components = (ψ̃ = zeros(Float64, size(ψ̃_rollout)), a = controls, Δt = Δt)
traj = NamedTrajectory(
    components; 
    timestep=:Δt, 
    controls=:a,
    initial=(ψ̃ = ket_to_iso(ψ_init),),
    goal=(ψ̃ = ket_to_iso(ψ_goal),),
)
rollout_fidelity(traj, system)

#=
!!! note
    The rollout fidelity is not the same thing as the fidelity of the final trajectory state.
=#

fidelity(iso_to_ket(traj.ψ̃[:, end]), ψ_goal)

#=
## Unitary Rollouts

```@docs; canonical = false
unitary_rollout
```

```@docs; canonical = false
unitary_rollout_fidelity
```

=#

Ũ⃗_rollout = unitary_rollout(controls, Δt, system) 
Ũ⃗_rollout |> size

# _Convert to unitary operators, and have a look at the initial unitary._

iso_vec_to_operator.(eachcol(Ũ⃗_rollout[:, 1])) |> first


#=
## Open Quantum System Rollouts

```@docs; canonical = false
open_rollout
```

```@docs; canonical = false
open_rollout_fidelity
```

=#

# _Open rollouts require open quantum systems_.
open_system = OpenQuantumSystem(system)

ρ_init = ψ_init * ψ_init'
ρ̃⃗_rollout = open_rollout(ρ_init, controls, Δt, open_system) 
ρ̃⃗_rollout |> size
