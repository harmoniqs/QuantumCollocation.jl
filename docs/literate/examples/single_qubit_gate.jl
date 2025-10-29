# # Single Qubit Gate

# This is the canonical example for getting started with QuantumCollocation.jl. We'll optimize control pulses to implement a Hadamard gate on a single qubit.

# ## Problem Setup

# We have a qubit with a drift Hamiltonian and two control drives (X and Y). Our goal is to find smooth control pulses that implement the Hadamard gate.

using QuantumCollocation
using PiccoloQuantumObjects
using NamedTrajectories

using PiccoloPlots
using CairoMakie

# Define the system Hamiltonian: $H(t) = H_{\text{drift}} + u_x(t) H_x + u_y(t) H_y$

## Drift (always on, not controllable)
H_drift = 0.1 * PAULIS.Z  # Small Z bias

## Control Hamiltonians (we can modulate these)
H_drives = [PAULIS.X, PAULIS.Y]

## System parameters
T_max = 10.0  # Maximum time (ns or μs, depends on your units)
drive_bounds = [(-1.0, 1.0), (-1.0, 1.0)]  # Control amplitude bounds

## Create the quantum system
sys = QuantumSystem(H_drift, H_drives, T_max, drive_bounds)

# ## Define the Target Gate

# We want to implement a Hadamard gate:

U_goal = GATES.H
println("Target gate:")
display(U_goal)

# ## Set Up the Optimization Problem

# We'll use 51 timesteps to discretize the time interval:

N = 51  # Number of timesteps

## Create a smooth pulse problem
prob = UnitarySmoothPulseProblem(sys, U_goal, N)

# Let's check the initial fidelity before optimization:

fid_initial = unitary_rollout_fidelity(prob.trajectory, sys)
println("Initial fidelity: ", fid_initial)

# ## Solve the Problem

solve!(prob; max_iter=100)

# ## Analyze the Results

# Check the final fidelity:

fid_final = unitary_rollout_fidelity(prob.trajectory, sys)
println("Final fidelity: ", fid_final)
@assert fid_final > 0.99

# Plot the optimized control pulses:

plot_unitary_populations(prob.trajectory)

# ## Understanding the Solution

# Let's look at the control pulses in more detail:

## Extract the controls
u = prob.trajectory.u  # Control amplitudes (2 × N matrix)
du = prob.trajectory.du  # First derivatives
ddu = prob.trajectory.ddu  # Second derivatives

println("Control pulse shape: ", size(u))
println("Max control amplitude: ", maximum(abs.(u)))
println("Max control derivative: ", maximum(abs.(ddu)))

# The smooth pulse problem penalizes both control amplitude and its derivatives, resulting in smooth, implementable pulses.

# ## Duration

# What's the total duration of this gate?

duration = get_duration(prob.trajectory)
println("Gate duration: ", duration)

# ## Next Steps

# Now that you've seen the basic workflow, you can:
# 
# - Try different gates (X, Y, Z, CNOT on coupled qubits)
# - Adjust the number of timesteps N
# - Modify the regularization weights (R_u, R_du, R_ddu)
# - Add constraints like leakage suppression for multilevel systems
# - Optimize for minimum time instead of smooth pulses
#
# See the other examples and the problem templates documentation for more!
