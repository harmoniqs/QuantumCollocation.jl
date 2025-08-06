# ```@meta
# CollapsedDocStrings = true
# ```

# # Quantum Objects

using PiccoloQuantumObjects
using SparseArrays # for visualization
⊗ = kron;

#=

Quantum states and operators are represented as complex vectors and matrices. We provide a 
number of convenient ways to construct these objects. We also provide some tools for 
working with these objects, such as embedding operators in larger Hilbert spaces and 
selecting subspace indices.

=#

#=
## Quantum states

We can construct quantum states from bitstrings or string representations. The string 
representations use atomic notation (ground state `g`, excited state `e`, etc.).
=#

# _Ground state in a 2-level system._
ket_from_string("g", [2])

# _Superposition state coupled to a ground state in two 2-level systems._
ket_from_string("(g+e)g", [2,2])

# |01⟩ _in a 2-qubit system._
ket_from_bitstring("01")

#=
## Quantum operators

Frequently used operators are provided in [`PAULIS`](@ref) and [`GATES`](@ref).
```@docs; canonical = false
GATES
```
=#

# _Quantum operators can also be constructed from strings._
operator_from_string("X")
#

operator_from_string("XZ")

# _Annihilation and creation operators are provided for oscillator systems._
a = annihilate(3)

#
a⁺ = create(3)

#
a'a

# ### Random operators

# The [`haar_random`](@ref) function draws random unitary operators according to the Haar
# measure.

haar_random(3)

# If we want to generate random operations that are close to the identity, we can use the 
# [`haar_identity`](@ref) function.

haar_identity(2, 0.1)

# _A smaller radius means the random operator is closer to the identity._
haar_identity(2, 0.01)

#=

## Embedded operators
Sometimes we want to embed a quantum operator into a larger Hilbert space, $\mathcal{H}$,
which we decompose into subspace and leakage components:
```math
    \mathcal{H} = \mathcal{H}_{\text{subspace}} \oplus \mathcal{H}_{\text{leakage}},
```
In quantum computing, the computation is encoded in a `subspace`, while the remaining
`leakage` states should be avoided.

### The `embed` and `unembed` functions

The [`embed`](@ref) function allows to embed a quantum operator in a larger Hilbert space.
```@docs; canonical = false
embed
```

The [`unembed`](@ref) function allows to unembed a quantum operator from a larger Hilbert 
space.
```@docs; canonical = false
unembed
```
=#

# _Embed a two-level X gate into a multilevel system._
levels = 3
X = GATES[:X]
subspace_indices = 1:2
X_embedded = embed(X, subspace_indices, levels)

# _Unembed to retrieve the original operator._
X_original = unembed(X_embedded, subspace_indices)

#=
### The `EmbeddedOperator` type
The [`EmbeddedOperator`](@ref) type stores information about an operator embedded in the 
subspace of a larger quantum system.
```@docs; canonical = false
EmbeddedOperator
```

We construct an embedded operator in the same manner as the `embed` function.
```@docs; canonical = false
EmbeddedOperator(subspace_operator::Matrix{<:Number}, subspace::AbstractVector{Int}, subsystem_levels::AbstractVector{Int})
```
=#

# _Embed an X gate in the first qubit's subspace within two 3-level systems._
gate = GATES[:X] ⊗ GATES[:I]
subsystem_levels = [3, 3]
subspace_indices = get_subspace_indices([1:2, 1:2], subsystem_levels)
embedded_operator = EmbeddedOperator(gate, subspace_indices, subsystem_levels)

# _Show the full operator._
embedded_operator.operator .|> real |> sparse

# _Get the original operator back._
unembed(embedded_operator) .|> real |> sparse

#=
Embedded operators for composite systems are also supported. 

```@docs; canonical = false
EmbeddedOperator(
    subspace_operator::AbstractMatrix{<:Number},
    subsystem_indices::AbstractVector{Int},
    subspaces::AbstractVector{<:AbstractVector{Int}},
    subsystem_levels::AbstractVector{Int}
)
```
    
This is a two step process.
1. The provided subspace operator is [`lift_operator`](@ref)-ed  from the `subsystem_indices` where
    it is defined into the space spanned by the composite system's `subspaces`.
2. The lifted operator is embedded into the full Hilbert space spanned by the 
    `subsystem_levels`.
=#

# _Embed a CZ gate with control qubit 1 and target qubit 3 into a composite system made up 
# of three 3-level systems. An identity is performed on qubit 2._
subsystem_levels = [3, 3, 3]
subspaces = [1:2, 1:2, 1:2]
embedded_operator = EmbeddedOperator(GATES[:CZ], [1, 3], subspaces, subsystem_levels)
unembed(embedded_operator) .|> real |> sparse

#=
## Subspace and leakage indices

### The `get_subspace_indices` function
The [`get_subspace_indices`](@ref) function is a convenient way to get the indices of a
subspace in a larger quantum system. 
```@docs; canonical = false
get_subspace_indices
```
Its dual function is [`get_leakage_indices`](@ref). 

=#

get_subspace_indices(1:2, 5) |> collect, get_leakage_indices(1:2, 5) |> collect

# _Composite systems are supported. Get the indices of the two-qubit
# subspace within two 3-level systems._
get_subspace_indices([1:2, 1:2], [3, 3])

# _Qubits are assumed if the indices are not provided._
get_subspace_indices([3, 3])

#
get_leakage_indices([3, 3])

#=
### Excitation number restrictions
Sometimes we want to cap the number of excitations we allow across a composite system. 
For example, if we want to restrict ourselves to the ground and single excitation states 
of two 3-level systems:
=#
get_enr_subspace_indices(1, [3, 3])

#=
### The `get_iso_vec_subspace_indices` function
For isomorphic operators, the [`get_iso_vec_subspace_indices`](@ref) function can be used 
to find the appropriate vector indices of the equivalent operator subspace. See also,
[Isomorphisms#Quantum-operator-isomorphisms](isomorphisms.md#Quantum-operator-isomorphisms).
```@docs; canonical = false
get_iso_vec_subspace_indices
```

Its dual function is [`get_iso_vec_leakage_indices`](@ref), which by default only returns
the leakage indices of the blocks:
```math
\mathcal{H}_{\text{subspace}} \otimes \mathcal{H}_{\text{subspace}},\quad
\mathcal{H}_{\text{subspace}} \otimes \mathcal{H}_{\text{leakage}},\quad
\mathcal{H}_{\text{leakage}} \otimes \mathcal{H}_{\text{subspace}}
```
allowing for leakage-suppressing code to disregard the uncoupled pure-leakage space.
=#

get_iso_vec_subspace_indices(1:2, 3)

#
without_pure_leakage = get_iso_vec_leakage_indices(1:2, 3)

# _Show the pure-leakage indices._
with_pure_leakage = get_iso_vec_leakage_indices(1:2, 3, ignore_pure_leakage=false)
setdiff(with_pure_leakage, without_pure_leakage)

# _The pure-leakage indices can grow quickly!_
without_pure_leakage = get_iso_vec_leakage_indices([1:2, 1:2], [4, 4])
with_pure_leakage = get_iso_vec_leakage_indices([1:2, 1:2], [4, 4], ignore_pure_leakage=false)
setdiff(with_pure_leakage, without_pure_leakage) |> length
