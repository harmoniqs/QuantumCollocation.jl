export SplinePulseProblem

# Helper function to determine spline order from pulse type
_get_spline_order(::LinearSplinePulse) = 1
_get_spline_order(::CubicSplinePulse) = 3

@doc raw"""
    SplinePulseProblem(qtraj::AbstractQuantumTrajectory{<:AbstractSplinePulse}, N::Int; kwargs...)

Construct a `QuantumControlProblem` for spline-based pulse optimization.

Unlike `SmoothPulseProblem` (which uses piecewise constant controls with discrete smoothing 
variables), this problem template is designed for spline-based pulses where the derivative 
variables (`du`) are the actual spline coefficients or slopes.

## Pulse Type Semantics

**LinearSplinePulse**: The `du` variable represents the slope at each knot point. The pulse 
amplitude is linearly interpolated between knots.

**CubicSplinePulse** (Hermite spline): The `du` variable is the tangent/derivative at each 
knot point, which is a true degree of freedom in Hermite interpolation.

## Mathematical Notes

For `CubicSplinePulse` (Hermite splines), the `:du` values are true degrees of freedom 
representing the tangent/derivative at each knot point. These are independent of the 
control values `:u` and are used directly by the `SplineIntegrator` for cubic Hermite 
interpolation.

Unlike `SmoothPulseProblem`, there is **no `DerivativeIntegrator` constraint** enforcing 
a finite-difference relationship between `:u` and `:du`. The optimizer is free to adjust 
both independently, subject only to regularization.

# Arguments
- `qtraj::AbstractQuantumTrajectory{<:AbstractSplinePulse}`: Quantum trajectory with spline pulse
- `N::Int`: Number of timesteps for the discretization

# Keyword Arguments
- `integrator::Union{Nothing, AbstractIntegrator, Vector{<:AbstractIntegrator}}=nothing`: Optional custom integrator(s). If not provided, uses BilinearIntegrator.
- `du_bound::Float64=Inf`: Bound on derivative (slope) magnitude
- `Q::Float64=100.0`: Weight on infidelity/objective
- `R::Float64=1e-2`: Weight on regularization terms
- `R_u::Union{Float64, Vector{Float64}}=R`: Weight on control regularization
- `R_du::Union{Float64, Vector{Float64}}=R`: Weight on derivative regularization  
- `constraints::Vector{<:AbstractConstraint}=AbstractConstraint[]`: Additional constraints
- `piccolo_options::PiccoloOptions=PiccoloOptions()`: Piccolo solver options

# Returns
- `QuantumControlProblem{<:AbstractQuantumTrajectory}`: Wrapper containing trajectory and optimization problem

# Examples
```julia
# Linear spline pulse
sys = QuantumSystem(H_drift, H_drives, drive_bounds)
pulse = LinearSplinePulse(0.1 * randn(n_drives, N), collect(range(0.0, T, length=N)))
qtraj = UnitaryTrajectory(sys, pulse, U_goal)

qcp = SplinePulseProblem(qtraj, N; Q=100.0, R=1e-2, du_bound=10.0)
solve!(qcp; max_iter=100)
```

See also: [`SmoothPulseProblem`](@ref) for piecewise constant pulses with discrete smoothing.
"""
function SplinePulseProblem(
    qtraj::AbstractQuantumTrajectory{<:AbstractSplinePulse},
    N::Int;
    integrator::Union{Nothing,AbstractIntegrator,Vector{<:AbstractIntegrator}}=nothing,
    du_bound::Float64=Inf,
    Δt_bounds::Union{Nothing,Tuple{Float64,Float64}}=nothing,
    Q::Float64=100.0,
    R::Float64=1e-2,
    R_u::Union{Float64,Vector{Float64}}=R,
    R_du::Union{Float64,Vector{Float64}}=R,
    constraints::Vector{<:AbstractConstraint}=AbstractConstraint[],
    piccolo_options::PiccoloOptions=PiccoloOptions(),
)
    sys = get_system(qtraj)
    control_sym = drive_name(qtraj)
    state_sym = state_name(qtraj)

    if piccolo_options.verbose
        pulse_type = typeof(qtraj.pulse)
        println("    constructing SplinePulseProblem with $(pulse_type)...")
    end

    # Convert quantum trajectory to NamedTrajectory
    base_traj = NamedTrajectory(qtraj, N; Δt_bounds=Δt_bounds)

    # Add control derivatives to trajectory (1 derivative for splines)
    # For CubicSplinePulse, :du is already included in the base trajectory
    # For LinearSplinePulse, we need to add :du
    du_sym = Symbol(:d, control_sym)
    
    traj = if haskey(base_traj.components, du_sym)
        # CubicSplinePulse already has derivative DOFs, but bounds default to (-Inf, Inf)
        # We need to update them if du_bound is specified
        if isfinite(du_bound)
            # Use update_bound! to set the bounds properly
            update_bound!(base_traj, du_sym, (-du_bound, du_bound))
            if piccolo_options.verbose
                println("    set du bounds to ±$du_bound for CubicSplinePulse")
            end
        end
        base_traj
    else
        # LinearSplinePulse needs derivatives added
        du_bounds_vec = isfinite(du_bound) ? fill(du_bound, sys.n_drives) : Float64[]
        if !isempty(du_bounds_vec)
            add_control_derivatives(
                base_traj,
                1;  # Only 1 derivative for spline pulses
                control_name=control_sym,
                derivative_bounds=(du_bounds_vec,)
            )
        else
            add_control_derivatives(
                base_traj,
                1;
                control_name=control_sym
            )
        end
    end

    # Initialize dynamics integrators
    if isnothing(integrator)
        # Default to BilinearIntegrator
        default_int = BilinearIntegrator(qtraj, N)
        
        if default_int isa AbstractVector
            dynamics_integrators = AbstractIntegrator[default_int...]
        else
            dynamics_integrators = AbstractIntegrator[default_int]
        end
    elseif integrator isa AbstractIntegrator
        dynamics_integrators = AbstractIntegrator[integrator]
    else
        dynamics_integrators = AbstractIntegrator[integrator...]
    end

    # Get control names
    du_sym = Symbol(:d, control_sym)

    # Build objective: type-specific infidelity + regularization
    J = _state_objective(qtraj, traj, state_sym, Q)

    # Add regularization for control and derivative
    J += QuadraticRegularizer(control_sym, traj, R_u)
    J += QuadraticRegularizer(du_sym, traj, R_du)

    # Apply piccolo options
    J += _apply_piccolo_options(qtraj, piccolo_options, constraints, traj, state_sym)

    # Start with dynamics integrators
    integrators = copy(dynamics_integrators)

    # Note: No DerivativeIntegrator for CubicSplinePulse - the :du values are Hermite tangents
    # (instantaneous derivatives at knot points), not finite differences between knot values.
    # The SplineIntegrator uses these tangents directly for cubic Hermite interpolation.

    prob = DirectTrajOptProblem(
        traj,
        J,
        integrators;
        constraints=constraints
    )

    return QuantumControlProblem(qtraj, prob)
end

# ============================================================================= #
# MultiKetTrajectory Method
# ============================================================================= #

"""
    SplinePulseProblem(qtraj::MultiKetTrajectory{<:AbstractSplinePulse}, N; kwargs...)

Create a spline-based trajectory optimization problem for ensemble ket state transfers.

Uses coherent fidelity objective (phases must align) for gate implementation.

# Arguments  
- `qtraj::MultiKetTrajectory{<:AbstractSplinePulse}`: Ensemble trajectory with spline pulse
- `N::Int`: Number of timesteps

# Keyword Arguments
Same as the base `SplinePulseProblem` method.
"""
function SplinePulseProblem(
    qtraj::MultiKetTrajectory{<:AbstractSplinePulse},
    N::Int;
    integrator::Union{Nothing,AbstractIntegrator,Vector{<:AbstractIntegrator}}=nothing,
    integrator_type::Symbol=:spline,  # :spline or :ensemble
    parallel_backend::Symbol=:manual,  # :manual (default), :threads, :gpu
    du_bound::Float64=Inf,
    Δt_bounds::Union{Nothing,Tuple{Float64,Float64}}=nothing,
    Q::Float64=100.0,
    R::Float64=1e-2,
    R_u::Union{Float64,Vector{Float64}}=R,
    R_du::Union{Float64,Vector{Float64}}=R,
    constraints::Vector{<:AbstractConstraint}=AbstractConstraint[],
    piccolo_options::PiccoloOptions=PiccoloOptions(),
)
    sys = get_system(qtraj)
    control_sym = drive_name(qtraj)
    snames = state_names(qtraj)
    weights = qtraj.weights
    goals = qtraj.goals

    if piccolo_options.verbose
        pulse_type = typeof(qtraj.pulse)
        println("    constructing SplinePulseProblem for MultiKetTrajectory with $(pulse_type)...")
        println("\twith $(length(qtraj.initials)) state transfers")
    end

    # Convert quantum trajectory to NamedTrajectory
    base_traj = NamedTrajectory(qtraj, N; Δt_bounds=Δt_bounds)

    # Add control derivatives to trajectory (1 derivative for splines)
    du_sym = Symbol(:d, control_sym)
    
    traj = if haskey(base_traj.components, du_sym)
        # CubicSplinePulse already has derivative DOFs, but bounds default to (-Inf, Inf)
        # We need to update them if du_bound is specified
        if isfinite(du_bound)
            # Use update_bound! to set the bounds properly
            update_bound!(base_traj, du_sym, (-du_bound, du_bound))
            if piccolo_options.verbose
                println("    set du bounds to ±$du_bound for CubicSplinePulse")
            end
        end
        base_traj
    else
        # LinearSplinePulse needs derivatives added
        du_bounds_vec = isfinite(du_bound) ? fill(du_bound, sys.n_drives) : Float64[]
        if !isempty(du_bounds_vec)
            add_control_derivatives(
                base_traj,
                1;  # Only 1 derivative for spline pulses
                control_name=control_sym,
                derivative_bounds=(du_bounds_vec,)
            )
        else
            add_control_derivatives(
                base_traj,
                1;
                control_name=control_sym
            )
        end
    end

    # Initialize dynamics integrators
    if isnothing(integrator)
        # Choose integrator type based on integrator_type parameter
        if integrator_type == :ensemble
            dynamics_integrators = EnsembleSplineIntegrator(
                qtraj, N;
                spline_order=_get_spline_order(qtraj.pulse),
                parallel_backend=parallel_backend
            )
        else
            dynamics_integrators = BilinearIntegrator(qtraj, N)
        end
        
        if !(dynamics_integrators isa AbstractVector)
            dynamics_integrators = AbstractIntegrator[dynamics_integrators]
        else
            dynamics_integrators = AbstractIntegrator[dynamics_integrators...]
        end
    elseif integrator isa AbstractIntegrator
        dynamics_integrators = AbstractIntegrator[integrator]
    else
        dynamics_integrators = AbstractIntegrator[integrator...]
    end

    # Get control names
    du_sym = Symbol(:d, control_sym)

    # Build objective: coherent fidelity for ensemble
    J = _ensemble_ket_objective(qtraj, traj, snames, weights, goals, Q)

    # Add regularization for control and derivative
    J += QuadraticRegularizer(control_sym, traj, R_u)
    J += QuadraticRegularizer(du_sym, traj, R_du)

    # Apply piccolo options for each state
    J += _apply_piccolo_options(qtraj, piccolo_options, constraints, traj, snames)

    # Start with dynamics integrators
    integrators = copy(dynamics_integrators)

    # Note: No DerivativeIntegrator for CubicSplinePulse - the :du values are Hermite tangents
    # (instantaneous derivatives at knot points), not finite differences between knot values.

    prob = DirectTrajOptProblem(
        traj,
        J,
        integrators;
        constraints=constraints
    )

    return QuantumControlProblem(qtraj, prob)
end

# ============================================================================= #
# Fallback Error Method
# ============================================================================= #

"""
    SplinePulseProblem(qtraj::AbstractQuantumTrajectory, N::Int; kwargs...)

Fallback method that provides helpful error for non-spline pulse types.
"""
function SplinePulseProblem(
    qtraj::AbstractQuantumTrajectory{P},
    N::Int;
    kwargs...
) where P <: AbstractPulse
    pulse_type = P
    error("""
    SplinePulseProblem is only for spline-based pulses (LinearSplinePulse, CubicSplinePulse).
    
    You provided a trajectory with pulse type: $(pulse_type)
    
    For piecewise constant pulses (ZeroOrderPulse), use SmoothPulseProblem instead:
        qcp = SmoothPulseProblem(qtraj, N; ...)
    """)
end

# ============================================================================= #
# TestItems
# ============================================================================= #

@testitem "SplinePulseProblem with LinearSplinePulse" begin
    using QuantumCollocation
    using PiccoloQuantumObjects
    using NamedTrajectories
    using DirectTrajOpt
    using LinearAlgebra

    # Simple 2-level system
    σx = ComplexF64[0 1; 1 0]
    σz = ComplexF64[1 0; 0 -1]
    
    H_drift = 0.01 * σz
    H_drives = [σx]
    T = 10.0
    N = 51
    n_drives = 1
    
    # Create system and pulse
    sys = QuantumSystem(H_drift, H_drives, [1.0])
    
    times = collect(range(0.0, T, length=N))
    amps = 0.1 * randn(n_drives, N)
    pulse = LinearSplinePulse(amps, times)
    
    # Goal: X gate
    U_goal = ComplexF64[0 1; 1 0]
    
    # Create trajectory and problem
    qtraj = UnitaryTrajectory(sys, pulse, U_goal)
    qcp = SplinePulseProblem(qtraj, N; Q=100.0, R=1e-2)
    
    @test qcp isa QuantumControlProblem
    @test get_trajectory(qcp) isa NamedTrajectory
    
    # Check that we only have 1 derivative level (du, not ddu)
    traj = get_trajectory(qcp)
    @test haskey(traj.components, :u) || haskey(traj.components, :θ)
    @test !haskey(traj.components, :ddu)  # No second derivative for splines
end

@testitem "SplinePulseProblem with CubicSplinePulse" begin
    using QuantumCollocation
    using PiccoloQuantumObjects
    using NamedTrajectories
    using DirectTrajOpt
    using LinearAlgebra

    # Simple 2-level system  
    σx = ComplexF64[0 1; 1 0]
    σz = ComplexF64[1 0; 0 -1]
    
    H_drift = 0.01 * σz
    H_drives = [σx]
    T = 10.0
    N = 51
    n_drives = 1
    
    # Create system and pulse
    sys = QuantumSystem(H_drift, H_drives, [1.0])
    
    times = collect(range(0.0, T, length=N))
    amps = 0.1 * randn(n_drives, N)
    derivs = zeros(n_drives, N)  # Hermite spline with derivative DOFs
    pulse = CubicSplinePulse(amps, derivs, times)
    
    # Goal: X gate
    U_goal = ComplexF64[0 1; 1 0]
    
    # Create trajectory and problem
    qtraj = UnitaryTrajectory(sys, pulse, U_goal)
    qcp = SplinePulseProblem(qtraj, N; Q=100.0, R=1e-2)
    
    @test qcp isa QuantumControlProblem
    @test get_trajectory(qcp) isa NamedTrajectory
end

@testitem "SplinePulseProblem du_bound enforcement for CubicSplinePulse" begin
    using QuantumCollocation
    using PiccoloQuantumObjects
    using NamedTrajectories
    using DirectTrajOpt
    using LinearAlgebra

    # Simple 2-level system  
    σx = ComplexF64[0 1; 1 0]
    σz = ComplexF64[1 0; 0 -1]
    
    H_drift = 0.01 * σz
    H_drives = [σx]
    T = 10.0
    N = 51
    n_drives = 1
    
    # Create system and pulse
    sys = QuantumSystem(H_drift, H_drives, [1.0])
    
    times = collect(range(0.0, T, length=N))
    amps = 0.1 * randn(n_drives, N)
    derivs = zeros(n_drives, N)
    pulse = CubicSplinePulse(amps, derivs, times)
    
    U_goal = ComplexF64[0 1; 1 0]
    qtraj = UnitaryTrajectory(sys, pulse, U_goal)
    
    # Test with du_bound specified
    du_bound = 5.0
    qcp = SplinePulseProblem(qtraj, N; Q=100.0, R=1e-2, du_bound=du_bound)
    
    traj = get_trajectory(qcp)
    
    # Verify du bounds are set correctly
    @test haskey(traj.bounds, :du)
    du_bounds = traj.bounds[:du]
    
    # Bounds are stored as (lower_vector, upper_vector) tuple
    @test length(du_bounds) == 2  # (lower, upper) tuple
    lower_bounds, upper_bounds = du_bounds
    @test length(lower_bounds) == n_drives
    @test length(upper_bounds) == n_drives
    @test all(lower_bounds .≈ -du_bound)
    @test all(upper_bounds .≈ du_bound)
    
    # Test without du_bound (should default to Inf)
    qcp_unbounded = SplinePulseProblem(qtraj, N; Q=100.0, R=1e-2)
    traj_unbounded = get_trajectory(qcp_unbounded)
    
    # Without explicit du_bound, bounds should still be set to Inf (not throw error)
    @test haskey(traj_unbounded.bounds, :du)
end

@testitem "SplinePulseProblem rejects ZeroOrderPulse" begin
    using QuantumCollocation
    using PiccoloQuantumObjects
    using LinearAlgebra

    σx = ComplexF64[0 1; 1 0]
    σz = ComplexF64[1 0; 0 -1]
    
    H_drift = 0.01 * σz
    H_drives = [σx]
    T = 10.0
    N = 51
    
    sys = QuantumSystem(H_drift, H_drives, [1.0])
    
    times = collect(range(0.0, T, length=N))
    pulse = ZeroOrderPulse(0.1 * randn(1, N), times)
    
    U_goal = ComplexF64[0 1; 1 0]
    qtraj = UnitaryTrajectory(sys, pulse, U_goal)
    
    # Should error with helpful message
    @test_throws ErrorException SplinePulseProblem(qtraj, N)
end

@testitem "SplinePulseProblem with KetTrajectory" begin
    using QuantumCollocation
    using PiccoloQuantumObjects
    using NamedTrajectories
    using DirectTrajOpt
    using LinearAlgebra

    # Simple 2-level system
    σx = ComplexF64[0 1; 1 0]
    σz = ComplexF64[1 0; 0 -1]
    
    H_drift = 0.01 * σz
    H_drives = [σx]
    T = 10.0
    N = 51
    n_drives = 1
    
    # Create system and pulse
    sys = QuantumSystem(H_drift, H_drives, [1.0])
    
    times = collect(range(0.0, T, length=N))
    amps = 0.1 * randn(n_drives, N)
    pulse = LinearSplinePulse(amps, times)
    
    # State transfer: |0⟩ → |1⟩
    ψ_init = ComplexF64[1.0, 0.0]
    ψ_goal = ComplexF64[0.0, 1.0]
    
    # Create trajectory and problem
    qtraj = KetTrajectory(sys, pulse, ψ_init, ψ_goal)
    qcp = SplinePulseProblem(qtraj, N; Q=100.0, R=1e-2)
    
    @test qcp isa QuantumControlProblem
    @test qcp.qtraj isa KetTrajectory
    @test get_trajectory(qcp) isa NamedTrajectory
    
    # Check trajectory has proper components
    traj = get_trajectory(qcp)
    @test haskey(traj.components, :ψ̃)
    @test !haskey(traj.components, :ddu)  # No second derivative for splines
end

@testitem "SplinePulseProblem with MultiKetTrajectory" begin
    using QuantumCollocation
    using PiccoloQuantumObjects
    using NamedTrajectories
    using DirectTrajOpt
    using LinearAlgebra

    # Simple 2-level system
    σx = ComplexF64[0 1; 1 0]
    σz = ComplexF64[1 0; 0 -1]
    
    H_drift = 0.01 * σz
    H_drives = [σx]
    T = 10.0
    N = 51
    n_drives = 1
    
    # Create system and pulse
    sys = QuantumSystem(H_drift, H_drives, [1.0])
    
    times = collect(range(0.0, T, length=N))
    amps = 0.1 * randn(n_drives, N)
    pulse = LinearSplinePulse(amps, times)
    
    # Create ensemble: |0⟩ → |1⟩ and |1⟩ → |0⟩
    ψ0 = ComplexF64[1.0, 0.0]
    ψ1 = ComplexF64[0.0, 1.0]
    
    # Create trajectory and problem
    qtraj = MultiKetTrajectory(sys, pulse, [ψ0, ψ1], [ψ1, ψ0])
    qcp = SplinePulseProblem(qtraj, N; Q=100.0, R=1e-2)
    
    @test qcp isa QuantumControlProblem
    @test qcp.qtraj isa MultiKetTrajectory
    @test get_trajectory(qcp) isa NamedTrajectory
    
    # Check trajectory has proper components for both ensemble states
    traj = get_trajectory(qcp)
    @test haskey(traj.components, :ψ̃1)
    @test haskey(traj.components, :ψ̃2)
    @test !haskey(traj.components, :ddu)  # No second derivative for splines
    
    # Should have 2 dynamics integrators (one per state)
    dynamics_integrators = filter(i -> i isa BilinearIntegrator, qcp.prob.integrators)
    @test length(dynamics_integrators) == 2
end

@testitem "SplinePulseProblem with SamplingTrajectory" begin
    using QuantumCollocation
    using PiccoloQuantumObjects
    using NamedTrajectories
    using DirectTrajOpt
    using LinearAlgebra

    # Simple 2-level system with parameter variation
    σx = ComplexF64[0 1; 1 0]
    σz = ComplexF64[1 0; 0 -1]
    
    H_drift = 0.01 * σz
    H_drives = [σx]
    T = 10.0
    N = 51
    n_drives = 1
    
    # Create nominal and perturbed systems
    sys_nominal = QuantumSystem(H_drift, H_drives, [1.0])
    sys_perturbed = QuantumSystem(1.1 * H_drift, H_drives, [1.0])
    
    times = collect(range(0.0, T, length=N))
    amps = 0.1 * randn(n_drives, N)
    pulse = LinearSplinePulse(amps, times)
    
    # Goal: X gate
    U_goal = ComplexF64[0 1; 1 0]
    
    # Create base trajectory and sampling trajectory
    base_qtraj = UnitaryTrajectory(sys_nominal, pulse, U_goal)
    
    # First create a SplinePulseProblem with base trajectory
    base_qcp = SplinePulseProblem(base_qtraj, N; Q=100.0, R=1e-2)
    
    # Then create SamplingProblem
    sampling_qcp = SamplingProblem(base_qcp, [sys_nominal, sys_perturbed]; Q=100.0)
    
    @test sampling_qcp isa QuantumControlProblem
    @test sampling_qcp.qtraj isa SamplingTrajectory{<:AbstractPulse, <:UnitaryTrajectory}
    
    # Check trajectory has sample states
    traj = get_trajectory(sampling_qcp)
    @test haskey(traj.components, :Ũ⃗1)
    @test haskey(traj.components, :Ũ⃗2)
    @test !haskey(traj.components, :ddu)  # No second derivative for splines
end
