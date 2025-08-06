# ```@meta
# CollapsedDocStrings = true
# ```

# # `Abstract Quantum Systems`

using PiccoloQuantumObjects
using SparseArrays # for visualization
⊗ = kron;

#=
```@docs; canonical = false
AbstractQuantumSystem
```
=#

#=

## Quantum Systems

The [`QuantumSystem`](@ref) type is used to represent a quantum system with a drift 
Hamiltonian and a set of drive Hamiltonians,

```math
H = H_{\text{drift}} + \sum_i a_i H_{\text{drives}}^{(i)}
```

```@docs; canonical = false
QuantumSystem
```

`QuantumSystem`'s are containers for quantum dynamics. Internally, they compute the
necessary isomorphisms to perform the dynamics in a real vector space.

=#

H_drift = PAULIS[:Z]
H_drives = [PAULIS[:X], PAULIS[:Y]]
system = QuantumSystem(H_drift, H_drives)

a_drives = [1, 0]
system.H(a_drives)

#=
To extract the drift and drive Hamiltonians from a `QuantumSystem`, use the 
[`get_drift`](@ref) and [`get_drives`](@ref) functions. 

=#

get_drift(system) |> sparse

# _Get the X drive._
drives = get_drives(system)
drives[1] |> sparse

# _And the Y drive._
drives[2] |> sparse

#=
!!! note
    We can also construct a `QuantumSystem` directly from a Hamiltonian function. Internally,
    `ForwardDiff.jl` is used to compute the drives.
=#

H(a) = PAULIS[:Z] + a[1] * PAULIS[:X] + a[2] * PAULIS[:Y]
system = QuantumSystem(H, 2)
get_drives(system)[1] |> sparse

# _Create a noise model with a confusion matrix._
function H(a; C::Matrix{Float64}=[1.0 0.0; 0.0 1.0])
    b = C * a
    return b[1] * PAULIS.X + b[2] * PAULIS.Y
end

C_matrix = [0.99 0.01; -0.01 1.01]
system = QuantumSystem(a -> H(a, C=C_matrix), 2; params=Dict(:C => C_matrix))
confused_drives = get_drives(system)
confused_drives[1] |> sparse

# 
confused_drives[2] |> sparse

#=
## Open quantum systems

We can also construct an [`OpenQuantumSystem`](@ref) with Lindblad dynamics, enabling
a user to pass a list of dissipation operators.

```@docs; canonical = false
OpenQuantumSystem
```
=#

# _Add a dephasing and annihilation error channel._
H_drives = [PAULIS[:X]]
a = annihilate(2)
dissipation_operators = [a'a, a]
system = OpenQuantumSystem(H_drives, dissipation_operators=dissipation_operators)
system.dissipation_operators[1] |> sparse

# 
system.dissipation_operators[2] |> sparse

#=
!!! warning
    The Hamiltonian part `system.H` excludes the Lindblad operators. This is also true
    for functions that report properties of `system.H`, such as [`get_drift`](@ref), 
    [`get_drives`](@ref), and [`is_reachable`](@ref).
=#

get_drift(system) |> sparse

#=
## Time Dependent Quantum Systems
A [`TimeDependentQuantumSystem`](@ref) is a `QuantumSystem` with time-dependent Hamiltonians.
```@docs; canonical = false
TimeDependentQuantumSystem
```

A function `H(a, t)` or carrier and phase kwargs are used to specify time-dependent drives,
```math
    H(a, t) = H_{\text{drift}} + \sum_i a_i \cos(\omega_i t + \phi_i) H_{\text{drives}}^{(i)}
```
=#
# _Create a time-dependent Hamiltonian with a time-dependent drive._
H(a, t) = PAULIS.Z + a[1] * cos(t) * PAULIS.X
system = TimeDependentQuantumSystem(H, 1)

# _The drift Hamiltonian is the Z operator, but its now a function of time!_
get_drift(system)(0.0) |> sparse

# _The drive Hamiltonian is the X operator, but its now a function of time!_
get_drives(system)[1](0.0) |> sparse

# _Change the time to π._
get_drives(system)[1](π) |> sparse

# _Similar matrix constructors exist, but with carrier and phase kwargs._
system = TimeDependentQuantumSystem(PAULIS.Z, [PAULIS.X], carriers=[1.0], phases=[0.0])

# _This is the same as before, t=0.0:_
get_drives(system)[1](0.0) |> sparse

# _and at π:_
get_drives(system)[1](π) |> sparse


#=
## Composite quantum systems

A [`CompositeQuantumSystem`](@ref) is constructed from a list of subsystems and their 
interactions. The interaction, in the form of drift or drive Hamiltonian, acts on the full
Hilbert space. The subsystems, with their own drift and drive Hamiltonians, are internally
lifted to the full Hilbert space.

=#

system_1 = QuantumSystem([PAULIS[:X]])
system_2 = QuantumSystem([PAULIS[:Y]])
H_drift = PAULIS[:Z] ⊗ PAULIS[:Z]
system = CompositeQuantumSystem(H_drift, [system_1, system_2]);

# _The drift Hamiltonian is the ZZ coupling._
get_drift(system) |> sparse

# _The drives are the X and Y operators on the first and second subsystems._
drives = get_drives(system)
drives[1] |> sparse

#
drives[2] |> sparse

#=
### The `lift_operator` function

To lift operators acting on a subsystem into the full Hilbert space, use [`lift_operator`](@ref).
```@docs; canonical = false
lift_operator
```
=#

# _Create an `a + a'` operator acting on the 1st subsystem of a qutrit and qubit system._
subspace_levels = [3, 2]
lift_operator(create(3) + annihilate(3), 1, subspace_levels) .|> real |> sparse

# _Create IXI operator on the 2nd qubit in a 3-qubit system._
lift_operator(PAULIS[:X], 2, 3) .|> real |> sparse

# _Create an XX operator acting on qubits 3 and 4 in a 4-qubit system._
lift_operator([PAULIS[:X], PAULIS[:X]], [3, 4], 4) .|> real |> sparse

#=
We can also lift an operator that entangles different subspaces by passing the indices
of the entangled subsystems.
=#

#_Here's another way to create an XX operator acting on qubits 3 and 4 in a 4-qubit system._
lift_operator(kron(PAULIS[:X], PAULIS[:X]), [3, 4], 4) .|> real |> sparse

# _Lift a CX gate acting on the 1st and 3rd qubits in a 3-qubit system._
# _The result is independent of the state of the second qubit._
lift_operator(GATES[:CX], [1, 3], 3) .|> real |> sparse


#=
# Reachability tests

Whether a quantum system can be used to reach a target state or operator can be tested
by computing the dynamical Lie algebra. Access to this calculation is provided by the 
[`is_reachable`](@ref) function.
```@docs; canonical = false
is_reachable
```
=#

# _Y can be reached by commuting Z and X._
system = QuantumSystem(PAULIS[:Z], [PAULIS[:X]])
is_reachable(PAULIS[:Y], system)

# _Y cannot be reached by X alone._
system = QuantumSystem([PAULIS[:X]])
is_reachable(PAULIS[:Y], system)

#=
# Direct sums

The direct sum of two quantum systems is constructed with the [`direct_sum`](@ref) function.
```@docs; canonical = false
direct_sum
```
=#

# _Create a pair of non-interacting qubits._
system_1 = QuantumSystem(PAULIS[:Z], [PAULIS[:X], PAULIS[:Y]])
system_2 = QuantumSystem(PAULIS[:Z], [PAULIS[:X], PAULIS[:Y]])
system = direct_sum(system_1, system_2)
get_drift(system) |> sparse
