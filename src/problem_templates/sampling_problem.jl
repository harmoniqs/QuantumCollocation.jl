export SamplingProblem

# Note: SamplingTrajectory is now exported from PiccoloQuantumObjects

# ============================================================================= #
# Sampling Objective Utilities
# ============================================================================= #

"""
    extract_regularization(objective, state_sym::Symbol) -> AbstractObjective

Extract regularization terms (non-state-dependent objectives) from a composite objective.
"""
function extract_regularization(objective, state_sym::Symbol)
    objs = hasproperty(objective, :objectives) ? objective.objectives : [objective]
    
    regularizers = filter(objs) do term
        term_syms = if hasproperty(term, :syms)
            term.syms
        elseif hasproperty(term, :var_names)
            term.var_names
        else
            Symbol[]
        end
        state_sym ∉ term_syms
    end
    
    return isempty(regularizers) ? NullObjective() : reduce(+, regularizers)
end

# ============================================================================= #
# Sampling State Objective (dispatch-based)
# ============================================================================= #

"""
    sampling_state_objective(qtraj, traj, state_sym, Q)

Create the state-dependent objective for a sampling member.
Dispatches on quantum trajectory type.
"""
function sampling_state_objective(
    qtraj::UnitaryTrajectory, 
    traj::NamedTrajectory, 
    state_sym::Symbol, 
    Q::Float64
)
    return UnitaryInfidelityObjective(get_goal(qtraj), state_sym, traj; Q=Q)
end

function sampling_state_objective(
    qtraj::KetTrajectory, 
    traj::NamedTrajectory, 
    state_sym::Symbol, 
    Q::Float64
)
    ψ_goal = get_goal(qtraj)
    return KetInfidelityObjective(ψ_goal, state_sym, traj; Q=Q)
end

function sampling_state_objective(
    qtraj::DensityTrajectory, 
    traj::NamedTrajectory, 
    state_sym::Symbol, 
    Q::Float64
)
    # DensityTrajectory doesn't have a fidelity objective yet
    # Return NullObjective for now
    return NullObjective(traj)
end

# ============================================================================= #
# SamplingProblem Constructor
# ============================================================================= #

@doc raw"""
    SamplingProblem(qcp::QuantumControlProblem, systems::Vector{<:AbstractQuantumSystem}; kwargs...)

Construct a `SamplingProblem` from an existing `QuantumControlProblem` and a list of systems.

This creates a robust optimization problem where the controls are shared across all systems,
but each system evolves according to its own dynamics. The objective is the weighted sum of
fidelity objectives for each system.

# Arguments
- `qcp::QuantumControlProblem`: The base problem (defines nominal trajectory, objective, etc.)
- `systems::Vector{<:AbstractQuantumSystem}`: List of systems to optimize over

# Keyword Arguments
- `weights::Vector{Float64}=fill(1.0, length(systems))`: Weights for each system
- `Q::Float64=100.0`: Weight on infidelity objective (explicit, not extracted from base problem)
- `piccolo_options::PiccoloOptions=PiccoloOptions()`: Options for the solver

# Returns
- `QuantumControlProblem{SamplingTrajectory}`: A new problem with the sampling trajectory
"""
function SamplingProblem(
    qcp::QuantumControlProblem,
    systems::Vector{<:AbstractQuantumSystem};
    weights::Vector{Float64}=fill(1.0, length(systems)),
    Q::Float64=100.0,
    piccolo_options::PiccoloOptions=PiccoloOptions(),
)
    if piccolo_options.verbose
        println("    constructing SamplingProblem...")
        println("\tusing $(length(systems)) systems")
    end

    base_qtraj = qcp.qtraj
    state_sym = get_state_name(base_qtraj)

    # 1. Build sampling trajectory with duplicated states
    new_traj, state_names = build_sampling_trajectory(
        get_trajectory(qcp), 
        state_sym, 
        length(systems)
    )

    # 2. Create SamplingTrajectory wrapper
    sampling_qtraj = SamplingTrajectory(base_qtraj, systems, weights, state_names)

    # 3. Build objective: weighted state objectives + shared regularization
    J_state = sum(
        sampling_state_objective(base_qtraj, new_traj, name, w * Q)
        for (name, w) in zip(state_names, weights)
    )
    J_reg = extract_regularization(qcp.prob.objective, state_sym)
    J_total = J_state + J_reg

    # 4. Build integrators: shared (derivative, time) + dynamics for each system
    shared_integrators = filter(qcp.prob.integrators) do int
        int isa DerivativeIntegrator || int isa TimeIntegrator
    end
    
    # Use BilinearIntegrator dispatch on SamplingTrajectory
    dynamics_integrators = BilinearIntegrator(sampling_qtraj, new_traj)
    
    all_integrators = vcat(shared_integrators, dynamics_integrators)

    # 5. Construct problem
    prob = DirectTrajOptProblem(
        new_traj,
        J_total,
        all_integrators;
        constraints=AbstractConstraint[]
    )

    return QuantumControlProblem(sampling_qtraj, prob)
end

# ============================================================================= #
# Composability with MinimumTimeProblem
# ============================================================================= #

function _update_goal(qtraj::SamplingTrajectory, new_goal)
    new_base = _update_goal(qtraj.base_trajectory, new_goal)
    return update_base_trajectory(qtraj, new_base)
end

function _final_fidelity_constraint(
    qtraj::SamplingTrajectory,
    final_fidelity::Float64,
    traj::NamedTrajectory
)
    constraints = [
        _sampling_fidelity_constraint(qtraj.base_trajectory, name, final_fidelity, traj)
        for name in get_ensemble_state_names(qtraj)
    ]
    return constraints
end

# Dispatch on trajectory type for fidelity constraint
function _sampling_fidelity_constraint(
    qtraj::UnitaryTrajectory,
    state_sym::Symbol,
    final_fidelity::Float64,
    traj::NamedTrajectory
)
    return FinalUnitaryFidelityConstraint(get_goal(qtraj), state_sym, final_fidelity, traj)
end

function _sampling_fidelity_constraint(
    qtraj::KetTrajectory,
    state_sym::Symbol,
    final_fidelity::Float64,
    traj::NamedTrajectory
)
    return FinalKetFidelityConstraint(get_goal(qtraj), state_sym, final_fidelity, traj)
end

# Tests
@testitem "SamplingProblem Construction" begin
    using QuantumCollocation
    using PiccoloQuantumObjects
    using DirectTrajOpt

    # Define system
    sys = QuantumSystem(GATES[:Z], [GATES[:X]], 10.0, [1.0])

    # Create base problem
    qtraj = UnitaryTrajectory(sys, GATES[:H], 10)
    qcp = SmoothPulseProblem(qtraj; Q=100.0)

    # Create sampling problem with 2 systems
    systems = [sys, sys] # Identical systems for testing
    sampling_prob = SamplingProblem(qcp, systems)

    @test sampling_prob isa QuantumControlProblem
    @test sampling_prob.qtraj isa SamplingTrajectory
    @test length(sampling_prob.qtraj.systems) == 2

    # Check trajectory components (now use _sample_ suffix)
    traj = get_trajectory(sampling_prob)
    @test haskey(traj.components, :Ũ⃗_sample_1)
    @test haskey(traj.components, :Ũ⃗_sample_2)
    @test haskey(traj.components, :u)

    # Check integrators
    # Should have 2 dynamics integrators + derivative integrators
    # SmoothPulseProblem adds 2 derivative integrators.
    # So total should be 2 + 2 = 4 (plus maybe time integrator if time dependent)
    @test length(sampling_prob.prob.integrators) >= 4
end

@testitem "SamplingProblem Solving" tags = [:sampling_problem] begin
    using QuantumCollocation
    using PiccoloQuantumObjects
    using DirectTrajOpt

    # Simple robust optimization
    # System with uncertainty in drift
    sys_nominal = QuantumSystem(GATES[:Z], [GATES[:X]], 10.0, [1.0])
    sys_perturbed = QuantumSystem(1.1 * GATES[:Z], [GATES[:X]], 10.0, [1.0])

    qtraj = UnitaryTrajectory(sys_nominal, GATES[:X], 20)
    qcp = SmoothPulseProblem(qtraj; Q=100.0)

    sampling_prob = SamplingProblem(qcp, [sys_nominal, sys_perturbed])

    # Solve
    solve!(sampling_prob; max_iter=10, verbose=false)

    # Check that we have a solution
    @test sampling_prob.prob.objective(sampling_prob.trajectory) < 1e10 # Just check it didn't blow up
end

@testitem "SamplingProblem with KetTrajectory" begin
    using QuantumCollocation
    using PiccoloQuantumObjects
    using DirectTrajOpt

    # Robust state transfer over parameter uncertainty
    sys_nominal = QuantumSystem(GATES[:Z], [GATES[:X], GATES[:Y]], 1.0, [1.0, 1.0])
    sys_perturbed = QuantumSystem(1.1 * GATES[:Z], [GATES[:X], GATES[:Y]], 1.0, [1.0, 1.0])

    ψ_init = ComplexF64[1.0, 0.0]
    ψ_goal = ComplexF64[0.0, 1.0]
    qtraj = KetTrajectory(sys_nominal, ψ_init, ψ_goal, 20)

    qcp = SmoothPulseProblem(qtraj; Q=50.0, R=1e-3)

    # Create sampling problem
    sampling_prob = SamplingProblem(qcp, [sys_nominal, sys_perturbed]; Q=50.0)

    @test sampling_prob isa QuantumControlProblem
    @test sampling_prob.qtraj isa SamplingTrajectory

    # Check trajectory has sample states
    traj = get_trajectory(sampling_prob)
    @test haskey(traj.components, :ψ̃_sample_1)
    @test haskey(traj.components, :ψ̃_sample_2)

    # Solve
    solve!(sampling_prob; max_iter=10, verbose=false, print_level=1)
end

@testitem "SamplingProblem with custom weights" begin
    using QuantumCollocation
    using PiccoloQuantumObjects
    using DirectTrajOpt

    sys1 = QuantumSystem(GATES[:Z], [GATES[:X]], 10.0, [1.0])
    sys2 = QuantumSystem(1.1 * GATES[:Z], [GATES[:X]], 10.0, [1.0])
    sys3 = QuantumSystem(0.9 * GATES[:Z], [GATES[:X]], 10.0, [1.0])

    qtraj = UnitaryTrajectory(sys1, GATES[:X], 15)
    qcp = SmoothPulseProblem(qtraj; Q=100.0)

    # Non-uniform weights - emphasize nominal system
    weights = [0.6, 0.2, 0.2]
    sampling_prob = SamplingProblem(qcp, [sys1, sys2, sys3]; weights=weights, Q=100.0)

    @test sampling_prob.qtraj.weights == weights
    @test length(sampling_prob.qtraj.systems) == 3

    # Should have 3 sample states
    traj = get_trajectory(sampling_prob)
    @test haskey(traj.components, :Ũ⃗_sample_1)
    @test haskey(traj.components, :Ũ⃗_sample_2)
    @test haskey(traj.components, :Ũ⃗_sample_3)

    solve!(sampling_prob; max_iter=5, verbose=false, print_level=1)
end

@testitem "SamplingProblem + MinimumTimeProblem composition" begin
    using QuantumCollocation
    using PiccoloQuantumObjects
    using DirectTrajOpt

    # Robust minimum-time optimization
    sys_nominal = QuantumSystem(0.1 * GATES[:Z], [GATES[:X]], 1.0, [1.0])
    sys_perturbed = QuantumSystem(0.11 * GATES[:Z], [GATES[:X]], 1.0, [1.0])

    qtraj = UnitaryTrajectory(sys_nominal, GATES[:X], 30; Δt_bounds=(0.01, 0.5))
    qcp = SmoothPulseProblem(qtraj; Q=100.0, R=1e-2)

    # First create sampling problem
    sampling_prob = SamplingProblem(qcp, [sys_nominal, sys_perturbed]; Q=100.0)
    solve!(sampling_prob; max_iter=20, verbose=false, print_level=1)

    # Then convert to minimum-time
    mintime_prob = MinimumTimeProblem(sampling_prob; final_fidelity=0.90, D=50.0)

    @test mintime_prob isa QuantumControlProblem
    @test mintime_prob.qtraj isa SamplingTrajectory

    # Solve minimum-time
    solve!(mintime_prob; max_iter=20, verbose=false, print_level=1)
end
@testitem "SamplingProblem with DensityTrajectory" begin
    using QuantumCollocation
    using PiccoloQuantumObjects
    using DirectTrajOpt
    using LinearAlgebra

    # Robust open-system control over parameter uncertainty
    sys_nominal = OpenQuantumSystem(GATES[:Z], [GATES[:X], GATES[:Y]], 1.0, [1.0, 1.0])
    sys_perturbed = OpenQuantumSystem(1.1 * GATES[:Z], [GATES[:X], GATES[:Y]], 1.0, [1.0, 1.0])

    ρ_init = ComplexF64[1.0 0.0; 0.0 0.0]  # |0⟩⟨0|
    ρ_goal = ComplexF64[0.0 0.0; 0.0 1.0]  # |1⟩⟨1|
    
    qtraj = DensityTrajectory(sys_nominal, ρ_init, ρ_goal, 20)

    qcp = SmoothPulseProblem(qtraj; Q=100.0, R=1e-3)

    # Create sampling problem
    sampling_prob = SamplingProblem(qcp, [sys_nominal, sys_perturbed]; Q=100.0)

    @test sampling_prob isa QuantumControlProblem
    @test sampling_prob.qtraj isa SamplingTrajectory{DensityTrajectory}

    # Check trajectory has sample states
    traj = get_trajectory(sampling_prob)
    @test haskey(traj.components, :ρ⃗̃_sample_1)
    @test haskey(traj.components, :ρ⃗̃_sample_2)

    # Check integrators (2 dynamics + 2 derivatives)
    @test length(sampling_prob.prob.integrators) >= 4

    # Solve and verify dynamics are satisfied
    solve!(sampling_prob; max_iter=20, verbose=false, print_level=1)
    
    # Test dynamics constraints are satisfied for all integrators
    for integrator in sampling_prob.prob.integrators
        if integrator isa BilinearIntegrator
            δ = zeros(integrator.dim)
            DirectTrajOpt.evaluate!(δ, integrator, traj)
            @test norm(δ, Inf) < 1e-3
        end
    end
end