# # Multilevel Transmon

# In this example we will look at a multilevel transmon qubit with a Hamiltonian given by
#
# ```math
# \hat{H}(t) = -\frac{\delta}{2} \hat{n}(\hat{n} - 1) + u_1(t) (\hat{a} + \hat{a}^\dagger) + u_2(t) i (\hat{a} - \hat{a}^\dagger)
# ```
# where $\hat{n} = \hat{a}^\dagger \hat{a}$ is the number operator, $\hat{a}$ is the annihilation operator, $\delta$ is the anharmonicity, and $u_1(t)$ and $u_2(t)$ are control fields.
#
# We will use the following parameter values:
#
# ```math
# \begin{aligned}
# \delta &= 0.2 \text{ GHz}\\
# \abs{u_i(t)} &\leq 0.2 \text{ GHz}\\
# T_0 &= 10 \text{ ns}\\
# \end{aligned}
# ```
#
# For convenience, we have defined the `TransmonSystem` function in the `QuantumSystemTemplates` module, which returns a `QuantumSystem` object for a transmon qubit. We will use this function to define the system.

# ## Setting up the problem

# To begin, let's load the necessary packages, define the system parameters, and create a a `QuantumSystem` object using the `TransmonSystem` function.

using QuantumCollocation
using PiccoloQuantumObjects
using NamedTrajectories
using LinearAlgebra
using SparseArrays
using Random; Random.seed!(123)

using PiccoloPlots
using CairoMakie

## define the time parameters

N = 50      # number of time steps
T₀ = 10.0     # total time in ns

## define the system parameters
levels = 5
δ = 0.2

## add a bound to the controls
u_bound = 0.2

## create the system
sys = TransmonSystem(levels=levels, δ=δ, T_max=T₀, drive_bounds=fill(u_bound, 2))


# Since this is a multilevel transmon and we want to implement an, let's say, $X$ gate on the qubit subspace, i.e., the first two levels we can utilize the `EmbeddedOperator` type to define the target operator.

## define the target operator
op = EmbeddedOperator(:X, sys)

## show the full operator
op.operator |> sparse

# In this formulation, we also use a subspace identity as the initial state, which looks like

function get_subspace_identity(op::EmbeddedOperator)
    return embed(
        Matrix{ComplexF64}(I(length(op.subspace))),
        op.subspace,
        size(op)[1]
    )
end
get_subspace_identity(op) |> sparse

# We can then pass this embedded operator to the `UnitarySmoothPulseProblem` template to create the problem

## create the problem
prob = UnitarySmoothPulseProblem(sys, op, N)

## solve the problem
solve!(prob; max_iter=50)

# Let's look at the fidelity in the subspace

fid = unitary_rollout_fidelity(prob.trajectory, sys; subspace=op.subspace)
println("Fidelity: ", fid)
@assert fid > 0.99

# and plot the result using the `plot_unitary_populations` function.

plot_unitary_populations(prob.trajectory; fig_size=(900, 700))


# ## Leakage suppresion
# As can be seen from the above plot, there is a substantial amount of leakage into the higher levels during the evolution. To mitigate this, we have implemented a constraint to avoid populating the leakage levels, which should ideally drive those leakage populations down to zero.
# To implement this, pass `leakage_constraint=true` and set `leakage_constraint_value={value}` and `leakage_cost={value}` to the `PiccoloOptions` instance passed to the `UnitarySmoothPulseProblem` template.

## create the a leakage suppression problem, initializing with the previous solution

prob_leakage = UnitarySmoothPulseProblem(sys, op, N;
    u_guess=prob.trajectory.u[:, :],
    piccolo_options=PiccoloOptions(
        leakage_constraint=true,
        leakage_constraint_value=1e-2,
        leakage_cost=1e-2,
    ),
)

## solve the problem

solve!(prob_leakage; max_iter=250, options=IpoptOptions(eval_hessian=false))

# Let's look at the fidelity in the subspace

fid_leakage = unitary_rollout_fidelity(prob_leakage.trajectory, sys; subspace=op.subspace)
println("Fidelity: ", fid_leakage)
@assert fid_leakage > 0.99

# and plot the result using the `plot_unitary_populations` function.

plot_unitary_populations(prob_leakage.trajectory; fig_size=(900, 700))

# Here we can see that the leakage populations have been driven substantially down.
