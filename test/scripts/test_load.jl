#!/usr/bin/env julia
using Pkg
Pkg.activate(".")
try
    @info "Attempting to load QuantumCollocation..."
    using QuantumCollocation
    println("✓ QuantumCollocation loaded successfully!")
    println("✓ QuantumControlProblem available: ", isdefined(QuantumCollocation, :QuantumControlProblem))
    println("✓ SmoothPulseProblem available: ", isdefined(QuantumCollocation, :SmoothPulseProblem))
    println("✓ MinimumTimeProblem available: ", isdefined(QuantumCollocation, :MinimumTimeProblem))
catch e
    println("✗ Error loading QuantumCollocation:")
    showerror(stdout, e, catch_backtrace())
    println()
    exit(1)
end
