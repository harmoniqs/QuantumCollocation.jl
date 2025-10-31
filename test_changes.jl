using QuantumCollocation
using PiccoloQuantumObjects

println("Testing basic system creation...")
try
    sys = TransmonSystem(10.0, [0.2, 0.2]; ω=4.0, δ=0.2, levels=3)
    println("✓ TransmonSystem created successfully")
    println("  n_drives: ", sys.n_drives)
    println("  levels: ", sys.levels)
catch e
    println("✗ Error creating TransmonSystem:")
    println(e)
end

println("\nTesting QuantumStateSmoothPulseProblem...")
try
    N = 51
    Δt = 0.2
    sys = QuantumSystem(0.1 * GATES[:Z], [GATES[:X], GATES[:Y]], 10.0, [1.0, 1.0])
    ψ_init = Vector{ComplexF64}([1.0, 0.0])
    ψ_target = Vector{ComplexF64}([0.0, 1.0])
    
    prob = QuantumStateSmoothPulseProblem(
        sys, ψ_init, ψ_target, N, Δt;
        piccolo_options=PiccoloOptions(verbose=false)
    )
    println("✓ QuantumStateSmoothPulseProblem created successfully")
    println("  trajectory N: ", prob.trajectory.N)
catch e
    println("✗ Error creating problem:")
    println(e)
    for (exc, bt) in Base.catch_stack()
        showerror(stdout, exc, bt)
        println()
    end
end

println("\nTesting UnitarySmoothPulseProblem...")
try
    N = 51
    sys = TransmonSystem(10.0, [0.2, 0.2]; ω=4.0, δ=0.2, levels=2)
    U_goal = GATES[:X]
    
    prob = UnitarySmoothPulseProblem(
        sys, U_goal, N;
        piccolo_options=PiccoloOptions(verbose=false)
    )
    println("✓ UnitarySmoothPulseProblem created successfully")
    println("  trajectory N: ", prob.trajectory.N)
catch e
    println("✗ Error creating problem:")
    println(e)
    for (exc, bt) in Base.catch_stack()
        showerror(stdout, exc, bt)
        println()
    end
end

println("\nAll basic tests completed!")
