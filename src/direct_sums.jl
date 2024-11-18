module DirectSums

export add_suffix
export get_suffix
export get_suffix_label
export direct_sum

using ..Integrators
using ..Problems
using ..QuantumSystems
using ..Isomorphisms
using ..Objectives

using SparseArrays
using TestItemRunner

using NamedTrajectories


"""
    direct_sum(A::AbstractMatrix, B::AbstractMatrix)

Returns the direct sum of two matrices.
"""
function direct_sum(A::AbstractMatrix, B::AbstractMatrix)
    return [A spzeros((size(A, 1), size(B, 2))); spzeros((size(B, 1), size(A, 2))) B]
end

"""
    direct_sum(A::SparseMatrixCSC, B::SparseMatrixCSC)

Returns the direct sum of two sparse matrices.
"""
function direct_sum(A::SparseMatrixCSC, B::SparseMatrixCSC)
    return blockdiag(A, B)
end

"""
    direct_sum(Ã⃗::AbstractVector, B̃⃗::AbstractVector)

Returns the direct sum of two iso_vec operators.
"""
function direct_sum(Ã⃗::AbstractVector, B̃⃗::AbstractVector)
    return operator_to_iso_vec(direct_sum(iso_vec_to_operator(Ã⃗), iso_vec_to_operator(B̃⃗)))
end

"""
    direct_sum(sys1::QuantumSystem, sys2::QuantumSystem)

Returns the direct sum of two `QuantumSystem` objects.
"""
function direct_sum(sys1::QuantumSystem, sys2::QuantumSystem)
    H_drift = direct_sum(sys1.H_drift, sys2.H_drift)
    H1_zero = spzeros(size(sys1.H_drift))
    H2_zero = spzeros(size(sys2.H_drift))
    H_drives = [
        [direct_sum(H, H2_zero) for H ∈ sys1.H_drives]...,
        [direct_sum(H1_zero, H) for H ∈ sys2.H_drives]...
    ]
    return QuantumSystem(
        H_drift,
        H_drives,
        params=merge(sys1.params, sys2.params)
    )
end

direct_sum(systems::AbstractVector{<:QuantumSystem}) = reduce(direct_sum, systems)

# Add suffix utilities
# -----------------------
Base.startswith(symb::Symbol, prefix::AbstractString) = startswith(String(symb), prefix)
Base.startswith(symb::Symbol, prefix::Symbol) = startswith(String(symb), String(prefix))

add_suffix(symb::Symbol, suffix::String) = Symbol(string(symb, suffix))
add_suffix(symbs::Tuple, suffix::String; exclude::AbstractVector{<:Symbol}=Symbol[]) =
    Tuple(s ∈ exclude ? s : add_suffix(s, suffix) for s ∈ symbs)
add_suffix(symbs::AbstractVector, suffix::String; exclude::AbstractVector{<:Symbol}=Symbol[]) =
    [s ∈ exclude ? s : add_suffix(s, suffix) for s ∈ symbs]
add_suffix(d::Dict{Symbol, Any}, suffix::String; exclude::AbstractVector{<:Symbol}=Symbol[]) =
    typeof(d)(k ∈ exclude ? k : add_suffix(k, suffix) => v for (k, v) ∈ d)

function add_suffix(nt::NamedTuple, suffix::String; exclude::AbstractVector{<:Symbol}=Symbol[])
    symbs = Tuple(k ∈ exclude ? k : add_suffix(k, suffix) for k ∈ keys(nt))
    return NamedTuple{symbs}(values(nt))
end

function add_suffix(components::Union{Tuple, AbstractVector}, traj::NamedTrajectory, suffix::String)
    return add_suffix(get_components(components, traj), suffix)
end

function add_suffix(traj::NamedTrajectory, suffix::String)
    # TODO: Inplace
    # Timesteps are appended because of bounds and initial/final constraints.
    component_names = vcat(traj.state_names..., traj.control_names...)
    components = add_suffix(component_names, traj, suffix)
    controls = add_suffix(traj.control_names, suffix)
    return NamedTrajectory(
        components;
        controls=controls,
        timestep=traj.timestep isa Symbol ? add_suffix(traj.timestep, suffix) : traj.timestep,
        bounds=add_suffix(traj.bounds, suffix),
        initial=add_suffix(traj.initial, suffix),
        final=add_suffix(traj.final, suffix),
        goal=add_suffix(traj.goal, suffix)
    )
end

function add_suffix(sys::QuantumSystem, suffix::String)
    return QuantumSystem(
        sys.H_drift,
        sys.H_drives,
        params=add_suffix(sys.params, suffix)
    )
end

# get suffix label utilities
# --------------------

function get_suffix_label(s::String, pre::String)::String
    if startswith(s, pre)
        return chop(s, head=length(pre), tail=0)
    else
        error("Prefix '$pre' not found at the start of '$s'")
    end
end

get_suffix_label(symb::Symbol, pre::Symbol) = get_suffix_label(String(symb), String(pre))


# remove suffix utilities
# -----------------------

function remove_suffix(s::String, suffix::String)
    if endswith(s, suffix)
        return chop(s, tail=length(suffix))
    else
        error("Suffix '$suffix' not found at the end of '$s'")
    end
end

remove_suffix(symb::Symbol, suffix::String) = Symbol(remove_suffix(String(symb), suffix))
remove_suffix(symbs::Tuple, suffix::String; exclude::AbstractVector{<:Symbol}=Symbol[]) =
    Tuple(s ∈ exclude ? s : remove_suffix(s, suffix) for s ∈ symbs)
remove_suffix(symbs::AbstractVector, suffix::String; exclude::AbstractVector{<:Symbol}=Symbol[]) =
    [s ∈ exclude ? s : remove_suffix(s, suffix) for s ∈ symbs]
remove_suffix(d::Dict{Symbol, Any}, suffix::String; exclude::AbstractVector{<:Symbol}=Symbol[]) =
    typeof(d)(k ∈ exclude ? k : remove_suffix(k, suffix) => v for (k, v) ∈ d)

function remove_suffix(nt::NamedTuple, suffix::String; exclude::AbstractVector{<:Symbol}=Symbol[])
    symbs = Tuple(k ∈ exclude ? k : remove_suffix(k, suffix) for k ∈ keys(nt))
    return NamedTuple{symbs}(values(nt))
end

# Special integrator routines
# ---------------------------

function modify_integrator_suffix(
    modifier::Function,
    integrator::AbstractIntegrator,
    sys::AbstractQuantumSystem,
    traj::NamedTrajectory,
    mod_traj::NamedTrajectory,
    suffix::String
)
    if integrator isa UnitaryExponentialIntegrator
        unitary_name = get_component_names(traj, integrator.unitary_components)
        drive_name = get_component_names(traj, integrator.drive_components)
        return integrator(
            sys,
            mod_traj,
            unitary_name=modifier(unitary_name, suffix),
            drive_name=modifier(drive_name, suffix)
        )
    elseif integrator isa QuantumStateExponentialIntegrator
        state_name = get_component_names(traj, integrator.state_components)
        drive_name = get_component_names(traj, integrator.drive_components)
        return integrator(
            sys,
            mod_traj,
            state_name=modifier(state_name, suffix),
            drive_name=modifier(drive_name, suffix)
        )
    elseif integrator isa UnitaryPadeIntegrator
        unitary_name = get_component_names(traj, integrator.unitary_components)
        drive_name = get_component_names(traj, integrator.drive_components)
        return integrator(
            sys,
            mod_traj,
            unitary_name=modifier(unitary_name, suffix),
            drive_name=modifier(drive_name, suffix)
        )
    elseif integrator isa QuantumStatePadeIntegrator
        state_name = get_component_names(traj, integrator.state_components)
        drive_name = get_component_names(traj, integrator.drive_components)
        return integrator(
            sys,
            mod_traj,
            state_name=modifier(state_name, suffix),
            drive_name=modifier(drive_name, suffix)
        )
    elseif integrator isa DerivativeIntegrator
        variable = get_component_names(traj, integrator.variable_components)
        derivative = get_component_names(traj, integrator.derivative_components)
        return integrator(
            mod_traj,
            variable=modifier(variable, suffix),
            derivative=modifier(derivative, suffix)
        )
    else
        error("Integrator type not recognized")
    end
end

function add_suffix(
    integrator::AbstractIntegrator,
    sys::AbstractQuantumSystem,
    traj::NamedTrajectory,
    mod_traj::NamedTrajectory,
    suffix::String
)
    return modify_integrator_suffix(add_suffix, integrator, sys, traj, mod_traj, suffix)
end

function add_suffix(
    integrators::AbstractVector{<:AbstractIntegrator},
    sys::AbstractQuantumSystem,
    traj::NamedTrajectory,
    mod_traj::NamedTrajectory,
    suffix::String
)
    return [add_suffix(intg, sys, traj, mod_traj, suffix) for intg in integrators]
end

function remove_suffix(
    integrator::AbstractIntegrator,
    sys::AbstractQuantumSystem,
    traj::NamedTrajectory,
    mod_traj::NamedTrajectory,
    suffix::String
)
    return modify_integrator_suffix(remove_suffix, integrator, sys, traj, mod_traj, suffix)
end

function remove_suffix(
    integrators::AbstractVector{<:AbstractIntegrator},
    sys::AbstractQuantumSystem,
    traj::NamedTrajectory,
    mod_traj::NamedTrajectory,
    suffix::String
)
    return [remove_suffix(intg, sys, traj, mod_traj, suffix) for intg in integrators]
end


# Get suffix utilities
# --------------------

Base.endswith(symb::Symbol, suffix::AbstractString) = endswith(String(symb), suffix)
Base.endswith(integrator::UnitaryPadeIntegrator, suffix::String) = endswith(integrator.unitary_symb, suffix)
Base.endswith(integrator::DerivativeIntegrator, suffix::String) = endswith(integrator.variable, suffix)

function Base.endswith(integrator::AbstractIntegrator, traj::NamedTrajectory, suffix::String)
    if integrator isa UnitaryExponentialIntegrator
        name = get_component_names(traj, integrator.unitary_components)
    elseif integrator isa QuantumStateExponentialIntegrator
        name = get_component_names(traj, integrator.state_components)
    elseif integrator isa UnitaryPadeIntegrator
        name = get_component_names(traj, integrator.unitary_components)
    elseif integrator isa QuantumStatePadeIntegrator
        name = get_component_names(traj, integrator.state_components)
    elseif integrator isa DerivativeIntegrator
        name = get_component_names(traj, integrator.variable_components)
    else
        error("Integrator type not recognized")
    end
    return endswith(name, suffix)
end

function get_suffix(nt::NamedTuple, suffix::String; remove::Bool=false)
    names = Tuple(remove ? remove_suffix(k, suffix) : k for (k, v) ∈ pairs(nt) if endswith(k, suffix))
    values = [v for (k, v) ∈ pairs(nt) if endswith(k, suffix)]
    return NamedTuple{names}(values)
end

function get_suffix(d::Dict{<:Symbol, <:Any}, suffix::String; remove::Bool=false)
    return Dict(remove ? remove_suffix(k, suffix) : k => v for (k, v) ∈ d if endswith(k, suffix))
end

function get_suffix(traj::NamedTrajectory, suffix::String; remove::Bool=false)
    state_names = Tuple(s for s ∈ traj.state_names if endswith(s, suffix))

    # control names
    if traj.timestep isa Symbol
        if endswith(traj.timestep, suffix)
            control_names = Tuple(s for s ∈ traj.control_names if endswith(s, suffix))
            timestep = remove ? remove_suffix(traj.timestep, suffix) : traj.timestep
            exclude = Symbol[]
        else
            # extract the shared timestep
            control_names = Tuple(s for s ∈ traj.control_names if endswith(s, suffix) || s == traj.timestep)
            timestep = traj.timestep
            exclude = [timestep]
        end
    else
        control_names = Tuple(s for s ∈ traj.control_names if endswith(s, suffix))
        timestep = traj.timestep
        exclude = Symbol[]
    end

    component_names = Tuple(vcat(state_names..., control_names...))
    components = get_components(component_names, traj)
    if remove
        components = remove_suffix(components, suffix; exclude=exclude)
    end

    return NamedTrajectory(
        components,
        controls=remove ? remove_suffix(control_names, suffix; exclude=exclude) : control_names,
        timestep=timestep,
        bounds=get_suffix(traj.bounds, suffix, remove=remove),
        initial=get_suffix(traj.initial, suffix, remove=remove),
        final=get_suffix(traj.final, suffix, remove=remove),
        goal=get_suffix(traj.goal, suffix, remove=remove)
    )
end

function get_suffix(
    integrators::AbstractVector{<:AbstractIntegrator},
    sys::AbstractQuantumSystem,
    traj::NamedTrajectory,
    mod_traj::NamedTrajectory,
    suffix::String
)
    found = AbstractIntegrator[]
    for integrator ∈ integrators
        if endswith(integrator, traj, suffix)
            push!(found, remove_suffix(integrator, sys, traj, mod_traj, suffix))
        end
    end
    return found
end

function get_suffix(
    prob::QuantumControlProblem,
    subproblem_traj::NamedTrajectory,
    suffix::String;
    unitary_prefix::Symbol=:Ũ⃗,
    remove::Bool=false,
)
    # Extract the trajectory
    traj = get_suffix(prob.trajectory, suffix, remove=remove)

    # Extract the integrators
    integrators = get_suffix(prob.integrators, prob.system, prob.trajectory, subproblem_traj, suffix)

    # Get direct sum indices
    # TODO: Should have separate utility function
    # TODO: doesn't exclude more than one match
    i₀ = 0
    indices = Int[]
    for (k, d) ∈ pairs(traj.dims)
        if startswith(k, unitary_prefix)
            if endswith(k, suffix)
                # isovec: undo real/imag, undo vectorization
                append!(indices, i₀+1:i₀+isqrt(d ÷ 2))
            else
                i₀ += isqrt(d ÷ 2)
            end
        end
    end

    # Extract the system
    system = QuantumSystem(
        copy(prob.system.H_drift[indices, indices]),
        [copy(H[indices, indices]) for H in prob.system.H_drives if !iszero(H[indices, indices])],
        # params=get_suffix(prob.system.params, suffix)
    )

    # Null objective function
    # TODO: Should we extract past objectives?
    J = NullObjective()

    return QuantumControlProblem(
        system,
        traj,
        J,
        integrators
    )
end

# =========================================================================== #

@testitem "Apply suffix to trajectories" begin
    include("../test/test_utils.jl")

    traj = named_trajectory_type_1(free_time=false)
    suffix = "_new"
    new_traj = add_suffix(traj, suffix)

    @test new_traj.state_names == add_suffix(traj.state_names, suffix)
    @test new_traj.control_names == add_suffix(traj.control_names, suffix)

    same_traj = add_suffix(traj, "")
    @test traj == same_traj
end

@testitem "Merge systems" begin
    H_drift = 0.01 * GATES[:Z]
    H_drives = [GATES[:X], GATES[:Y]]
    T = 50
    sys = QuantumSystem(H_drift, H_drives, params=Dict(:T=>T))

    # apply suffix and sum
    sys2 = direct_sum(
        add_suffix(sys, "_1"),
        add_suffix(sys, "_2")
    )

    @test length(sys2.H_drives) == 4
    @test sys2.params[:T_1] == T
    @test sys2.params[:T_2] == T

    # add another system
    sys = QuantumSystem(H_drift, H_drives, params=Dict(:T=>T, :S=>2T))
    sys3 = direct_sum(sys2, add_suffix(sys, "_3"))
    @test length(sys3.H_drives) == 6
    @test sys3.params[:T_3] == T
    @test sys3.params[:S_3] == 2T
end

# TODO: fix broken test
@testitem "Get suffix" begin
    @test_broken false

    # using NamedTrajectories

    # sys = QuantumSystem(0.01 * GATES[:Z], [GATES[:X], GATES[:Y]])
    # T = 50
    # Δt = 0.2
    # ip_ops = IpoptOptions(print_level=1)
    # pi_ops = PiccoloOptions(verbose=false, free_time=false)
    # prob1 = UnitarySmoothPulseProblem(sys, GATES[:X], T, Δt, piccolo_options=pi_ops, ipopt_options=ip_ops)
    # prob2 = UnitarySmoothPulseProblem(sys, GATES[:Y], T, Δt, piccolo_options=pi_ops, ipopt_options=ip_ops)

    # # Direct sum problem with suffix extraction
    # # Note: Turn off control reset
    # direct_sum_prob = UnitaryDirectSumProblem([prob1, prob2], 0.99, drive_reset_ratio=0.0, ipopt_options=ip_ops)
    # # TODO: BROKEN HERE
    # prob1_got = get_suffix(direct_sum_prob, "1")
    # @test prob1_got.trajectory == add_suffix(prob1.trajectory, "1")

    # # Mutate the direct sum problem
    # update!(prob1_got.trajectory, :a1, ones(size(prob1_got.trajectory[:a1])))
    # @test prob1_got.trajectory != add_suffix(prob1.trajectory, "1")

    # # Remove suffix during extraction
    # prob1_got_without = get_suffix(direct_sum_prob, "1", remove=true)
    # @test prob1_got_without.trajectory == prob1.trajectory
end

# TODO: fix broken test
@testitem "Append to default integrators" begin
    @test_broken false
    # sys = QuantumSystem(0.01 * GATES[:Z], [GATES[:Y]])
    # T = 50
    # Δt = 0.2
    # ip_ops = IpoptOptions(print_level=1)
    # pi_false_ops = PiccoloOptions(verbose=false, free_time=false)
    # pi_true_ops = PiccoloOptions(verbose=false, free_time=true)
    # prob1 = UnitarySmoothPulseProblem(sys, GATES[:Y], T, Δt, piccolo_options=pi_false_ops, ipopt_options=ip_ops)
    # prob2 = UnitarySmoothPulseProblem(sys, GATES[:Y], T, Δt, piccolo_options=pi_true_ops, ipopt_options=ip_ops)

    # suffix = "_new"
    # # UnitaryPadeIntegrator
    # # TODO: BROKEN HERE
    # prob1_new = add_suffix(prob1.integrators, suffix)
    # @test prob1_new[1].unitary_symb == add_suffix(prob1.integrators[1].unitary_symb, suffix)
    # @test prob1_new[1].drive_symb == add_suffix(prob1.integrators[1].drive_symb, suffix)

    # # DerivativeIntegrator
    # @test prob1_new[2].variable == add_suffix(prob1.integrators[2].variable, suffix)

    # # UnitaryPadeIntegrator with free time
    # prob2_new = add_suffix(prob2.integrators, suffix)
    # @test prob2_new[1].unitary_symb == add_suffix(prob2.integrators[1].unitary_symb, suffix)
    # @test prob2_new[1].drive_symb == add_suffix(prob2.integrators[1].drive_symb, suffix)

    # # DerivativeIntegrator
    # @test prob2_new[2].variable == add_suffix(prob2.integrators[2].variable, suffix)
end

# TODO: fix broken test
@testitem "Free time get suffix" begin
    # using NamedTrajectories

    # sys = QuantumSystem(0.01 * GATES[:Z], [GATES[:Y]])
    # T = 50
    # Δt = 0.2
    # ops = IpoptOptions(print_level=1)
    # pi_false_ops = PiccoloOptions(verbose=false, free_time=false)
    # pi_true_ops = PiccoloOptions(verbose=false, free_time=true)
    # suffix = "_new"
    # timestep_name = :Δt

    # prob1 = UnitarySmoothPulseProblem(sys, GATES[:Y], T, Δt, piccolo_options=pi_false_ops, ipopt_options=ops)
    # traj1 = direct_sum(prob1.trajectory, add_suffix(prob1.trajectory, suffix), free_time=true)

    # # Direct sum (shared timestep name)
    # @test get_suffix(traj1, suffix).timestep == timestep_name
    # @test get_suffix(traj1, suffix, remove=true).timestep == timestep_name

    # prob2 = UnitarySmoothPulseProblem(sys, GATES[:Y], T, Δt, ipopt_options=ops, piccolo_options=pi_true_ops)
    # traj2 = add_suffix(prob2.trajectory, suffix)

    # # Trajectory (unique timestep name)
    # @test get_suffix(traj2, suffix).timestep == add_suffix(timestep_name, suffix)
    # @test get_suffix(traj2, suffix, remove=true).timestep == timestep_name
end

end # module
