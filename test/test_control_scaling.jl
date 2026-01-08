using Test
using QuantumCollocation
using PiccoloQuantumObjects
using Piccolissimo
using DirectTrajOpt
using LinearAlgebra

@testset "Control Scaling Integration Tests" begin
    
    @testset "SmoothPulseProblem with control scaling (KetTrajectory)" begin
        # Create system with asymmetric bounds
        levels = 2
        H_drift = GATES[:Z]
        H_drives = [GATES[:X]]
        
        sys = QuantumSystem(
            H_drift, 
            H_drives, 
            10.0,
            [1.0];
            drive_bounds=[(-2.0, 5.0)]  # Asymmetric bounds
        )
        
        ψ_init = ComplexF64[1.0, 0.0]
        ψ_goal = ComplexF64[0.0, 1.0]
        N = 15
        T = 5.0
        
        qtraj = KetTrajectory(sys, ψ_init, ψ_goal, N, T)
        
        # Create SplineIntegrator with control scaling
        integrator = SplineIntegrator(qtraj; use_control_scaling=true)
        
        # Create SmoothPulseProblem with the scaled integrator
        qcp = SmoothPulseProblem(qtraj; integrator=integrator, Q=50.0, R=1e-2)
        
        @test qcp isa QuantumControlProblem
        
        # Check that prob.trajectory has normalized controls :θ instead of :u
        @test haskey(qcp.prob.trajectory.components, :θ)
        @test !haskey(qcp.prob.trajectory.components, :u)
        
        # Check normalized bounds
        @test qcp.prob.trajectory.bounds[:θ] == [(-1.0, 1.0)]
        
        # Check that derivative variables were also renamed
        @test haskey(qcp.prob.trajectory.components, :dθ)
        @test haskey(qcp.prob.trajectory.components, :ddθ)
        
        # qtraj.trajectory should still have physical :u
        @test haskey(qtraj.trajectory.components, :u)
        @test !haskey(qtraj.trajectory.components, :θ)
        @test qtraj.trajectory.bounds[:u] == [(-2.0, 5.0)]
        
        # Solve the problem
        solve!(qcp; max_iter=50, verbose=false, print_level=1)
        
        # After solving with sync=true (default), qtraj.trajectory should have physical controls
        @test haskey(qtraj.trajectory.components, :u)
        @test !haskey(qtraj.trajectory.components, :θ)
        
        # Check that controls are within original physical bounds
        u_values = qtraj.trajectory.u
        @test all(u_values .>= -2.0)
        @test all(u_values .<= 5.0)
        
        # Test fidelity
        traj = qtraj.trajectory
        ψ̃_final = traj[end][:ψ̃]
        ψ_final = iso_to_ket(ψ̃_final)
        fid = fidelity(ψ_final, ψ_goal)
        @test fid > 0.5  # Modest fidelity requirement for this test
        
        # Test that dynamics constraints are satisfied in the optimized trajectory
        dynamics_integrator = qcp.prob.integrators[1]
        δ = zeros(dynamics_integrator.dim)
        DirectTrajOpt.evaluate!(δ, dynamics_integrator, qcp.prob.trajectory)
        @test norm(δ, Inf) < 1e-2
    end
    
    @testset "SmoothPulseProblem with control scaling (UnitaryTrajectory)" begin
        # Create system with asymmetric bounds
        H_drift = GATES[:Z]
        H_drives = [GATES[:X], GATES[:Y]]
        
        sys = QuantumSystem(
            H_drift, 
            H_drives, 
            10.0,
            [1.0, 1.0];
            drive_bounds=[(-3.0, 2.0), (-1.0, 4.0)]  # Different asymmetric bounds
        )
        
        U_goal = GATES[:X]
        N = 15
        T = 5.0
        
        qtraj = UnitaryTrajectory(sys, U_goal, N, T)
        
        # Create SplineIntegrator with control scaling
        integrator = SplineIntegrator(qtraj; use_control_scaling=true)
        
        # Verify scaling info
        @test integrator.scaling_info.scale_factors_lower == [3.0, 1.0]
        @test integrator.scaling_info.scale_factors_upper == [2.0, 4.0]
        
        # Create problem
        qcp = SmoothPulseProblem(qtraj; integrator=integrator, Q=100.0, R=1e-2)
        
        # Verify normalized trajectory structure
        @test haskey(qcp.prob.trajectory.components, :θ)
        @test qcp.prob.trajectory.bounds[:θ] == [(-1.0, 1.0), (-1.0, 1.0)]
        
        # Solve
        solve!(qcp; max_iter=30, verbose=false, print_level=1)
        
        # After solve, qtraj should have physical controls within original bounds
        u_values = qtraj.trajectory.u
        @test all(u_values[1, :] .>= -3.0)
        @test all(u_values[1, :] .<= 2.0)
        @test all(u_values[2, :] .>= -1.0)
        @test all(u_values[2, :] .<= 4.0)
        
        # Test basic fidelity
        Ũ⃗_final = qtraj.trajectory[end][:Ũ⃗]
        U_final = iso_vec_to_operator(Ũ⃗_final)
        fid = unitary_fidelity(U_final, U_goal)
        @test fid > 0.5
    end
    
end
