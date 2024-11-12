```@meta
EditURL = "../../../literate/examples/multilevel_transmon.jl"
```

# Multilevel Transmon

In this example we will look at a multilevel transmon qubit with a Hamiltonian given by

```math
\hat{H}(t) = -\frac{\delta}{2} \hat{n}(\hat{n} - 1) + u_1(t) (\hat{a} + \hat{a}^\dagger) + u_2(t) i (\hat{a} - \hat{a}^\dagger)
```
where $\hat{n} = \hat{a}^\dagger \hat{a}$ is the number operator, $\hat{a}$ is the annihilation operator, $\delta$ is the anharmonicity, and $u_1(t)$ and $u_2(t)$ are control fields.

We will use the following parameter values:

```math
\begin{aligned}
\delta &= 0.2 \text{ GHz}\\
\abs{u_i(t)} &\leq 0.2 \text{ GHz}\\
T_0 &= 10 \text{ ns}\\
\end{aligned}
```

For convenience, we have defined the `TransmonSystem` function in the `QuantumSystemTemplates` module, which returns a `QuantumSystem` object for a transmon qubit. We will use this function to define the system.

## Setting up the problem

To begin, let's load the necessary packages, define the system parameters, and create a a `QuantumSystem` object using the `TransmonSystem` function.

````@example multilevel_transmon
using QuantumCollocation
using NamedTrajectories
using LinearAlgebra
using SparseArrays
using Random; Random.seed!(123)

# define the time parameters

T₀ = 10     # total time in ns
T = 50      # number of time steps
Δt = T₀ / T # time step

# define the system parameters
levels = 5
δ = 0.2

# add a bound to the controls
a_bound = 0.2

# create the system
sys = TransmonSystem(levels=levels, δ=δ)

# let's look at the parameters of the system
sys.params
````

Since this is a multilevel transmon and we want to implement an, let's say, $X$ gate on the qubit subspace, i.e., the first two levels we can utilize the `EmbeddedOperator` type to define the target operator.

````@example multilevel_transmon
# define the target operator
op = EmbeddedOperator(:X, sys)

# show the full operator
op.operator |> sparse
````

In this formulation, we also use a subspace identity as the initial state, which looks like

````@example multilevel_transmon
get_subspace_identity(op) |> sparse
````

We can then pass this embedded operator to the `UnitarySmoothPulseProblem` template to create

````@example multilevel_transmon
# create the problem
prob = UnitarySmoothPulseProblem(sys, op, T, Δt; a_bound=a_bound)

# solve the problem
solve!(prob; max_iter=50)
````

Let's look at the fidelity in the subspace

````@example multilevel_transmon
println("Fidelity: ", unitary_rollout_fidelity(prob; subspace=op.subspace_indices))
````

and plot the result using the `plot_unitary_populations` function.

````@example multilevel_transmon
plot_unitary_populations(prob; fig_size=(900, 700))
````

## Leakage suppresion
As can be seen from the above plot, there is a substantial amount of leakage into the higher levels during the evolution. To mitigate this, we have implemented the ability to add a cost to populating the leakage levels, in particular this is an $L_1$ norm cost, which is implemented via slack variables and should ideally drive those leakage populations down to zero.
To implement this, pass `leakage_suppresion=true` and `R_leakage={value}` to the `UnitarySmoothPulseProblem` template.

````@example multilevel_transmon
# create the a leakage suppression problem, initializing with the previous solution

prob_leakage = UnitarySmoothPulseProblem(sys, op, T, Δt;
    a_bound=a_bound,
    leakage_suppression=true,
    R_leakage=1e-1,
    a_guess=prob.trajectory.a
)

# solve the problem

solve!(prob_leakage; max_iter=50)
````

Let's look at the fidelity in the subspace

````@example multilevel_transmon
println("Fidelity: ", unitary_rollout_fidelity(prob_leakage; subspace=op.subspace_indices))
````

and plot the result using the `plot_unitary_populations` function.

````@example multilevel_transmon
plot_unitary_populations(prob_leakage; fig_size=(900, 700))
````

Here we can see that the leakage populations have been driven substantially down.

---

*This page was generated using [Literate.jl](https://github.com/fredrikekre/Literate.jl).*

