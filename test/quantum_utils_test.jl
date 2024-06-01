"""
Tests: QuantumUtils submodule
"""

@testitem "GATES" begin
    using QuantumUtils
    @test get_gate(:X) * get_gate(:X) ==  get_gate(:I)
    @test get_gate(:Y) * get_gate(:Y) ==  get_gate(:I)
    @test -get_gate(:Z) * get_gate(:Z) ==  get_gate(:I)
    @test -get_gate(:X) * get_gate(:Y) == get_gate(:Z)
    @test Int.(round.(real.(GATES[:H]*GATES[:H]))) == GATES[:I]
    @test GATES[:X] ^ 2 == GATES[:I]
    @test GATES[:Y] ^ 2 == GATES[:I]
    @test GATES[:Z] ^ 2 == -GATES[:I]
    @test Int.(round.(real.(GATES[:X]*GATES[:Y] + GATES[:Y]*GATES[:X]))) == zeros(2, 2)
    @test Int.(round.(real.(GATES[:Y]*GATES[:Z] + GATES[:Z]*GATES[:Y]))) == zeros(2, 2)
    @test Int.(round.(real.(GATES[:Z]*GATES[:X] + GATES[:X]*GATES[:Z]))) == zeros(2, 2)
end

@testitem "⊗" begin
    using QuantumUtilis
    @test Int.(round.(real.(GATES[:X] ⊗ GATES[:X]))) == [0  0  0  1;
											 0  0  1  0;
											 0  1  0  0;
 											 1  0  0  0]
    @test Int.(round.(real.(GATES[:X] ⊗ GATES[:I]))) == [0  0  1  0;
											 0  0  0  1;
											 1  0  0  0;
 										 	 0  1  0  0]
    @test complex.(GATES[:Z] ⊗ GATES[:I]) == [-im	0	0 	0;
								      0	-im	0	0;
									 0   0	im   0;
 									 0   0    0	im]
    #Associativity: (X ⊗ Y) ⊗ Z  == X ⊗ (Y  ⊗ Z)
    @test(GATES[:X] ⊗ GATES[:Y]) ⊗ GATES[:Z] == GATES[:X] ⊗ (GATES[:Y] ⊗ GATES[:Z])
    #Distributivity: A ⊗( B + C ) = A ⊗ B + A ⊗ C    
    @test GATES[:X] ⊗ (GATES[:Y] + GATES[:Z]) == GATES[:X] ⊗ GATES[:Y] + GATES[:X] ⊗ GATES[:Z]
end 

@testitem "Test apply function with Pauli gates" begin
    using QuantumUtils
    # Define the initial state
    ψ₀ = [1.0 + 0.0im, 0.0 + 0.0im]  # Example initial state for a single qubit
    ψ₁ = [0.0 + 0.0im, 1.0 + 0.0im]

    # Apply the X gate
    @test apply(:X, ψ₀) == [0.0 + 0.0im, 1.0 + 0.0im] # Applying X gate should flip |0⟩ to |1⟩
    @test apply(:X, ψ₁) == [1.0 + 0.0im, 0.0 + 0.0im] # Applying X gate should flip |1⟩ to |0⟩

    # Define the Y gate matrix manually
    Y = [0.0 + 0.0im 0.0 - 1.0im;
         0.0 + 1.0im 0.0 + 0.0im]
    @test apply(:Y, ψ₀) == GATES[:Y] * ψ₀  # Apply the Y gate to the state |0⟩
    @test apply(:Y, ψ₁) == GATES[:Y] * ψ₁  # Apply the Y gate to the state |1⟩

end

# Define a test set for qubit_system_state function
@testitem "Test qubit_system_state function" begin
    using QuantumUtils
    @test qubit_system_state("0") == [1, 0]
    @test qubit_system_state("1") == [0, 1]
    @test qubit_system_state("00") == [1, 0, 0, 0]
    @test qubit_system_state("01") == [0, 1, 0, 0]
    @test qubit_system_state("10") == [0, 0, 1, 0]
    @test qubit_system_state("11") == [0, 0, 0, 1]
end

# Define a test set for lift function
@testitem "Test lift function" begin
    using QuantumUtils
    #Lift 2x2 identity matrix U to the first qubit in a 2-qubit system
    U1 = [1 0; 0 1]  # Identity matrix
    @test size(lift(U1, 1, 2)) == (4, 4) 

    #Lift 2x2 identity matrix U to the second qubit in a 2-qubit system
    @test size(lift(U1, 2, 2)) == (4, 4) 

    #Lift 2x2 identity matrix U to the first qubit in a 4-qubit system
    @test size(lift(U1, 1, 3)) == (16, 16)
end

# Define test cases for isomorphism utilities
@testitem "Test isomorphism utilities" begin
    using QuantumUtils
    # Test vector for conversion
    iso_vec = [1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0]
    
    # Test vec⁻¹ function: Converts a vector into a square matrix.
    @test vec⁻¹([1.0, 2.0, 3.0, 4.0]) == [1.0 3.0; 2.0 4.0]

    # Test ket_to_iso function: Converts a ket vector into a complex vector with real and imaginary parts.
    @test ket_to_iso([1.0, 2.0]) == [1.0, 2.0, 0.0, 0.0]

    # Test iso_to_ket function: Converts a complex vector with real and imaginary parts into a ket vector.
    @test iso_to_ket([1.0, 2.0, 0.0, 0.0]) == [1.0, 2.0]

    # Test iso_vec_to_operator function: Converts a complex vector into a complex matrix representing an operator.
    @test iso_vec_to_operator(iso_vec) == [1.0 0.0; 0.0 1.0]

    # Test iso_vec_to_iso_operator function: Converts a complex vector into a real matrix representing an isomorphism operator.
    @test iso_vec_to_iso_operator(iso_vec) == [1.0 0.0 -0.0 -0.0; 0.0 1.0 -0.0 -0.0; 0.0 0.0 1.0 0.0; 0.0 0.0 0.0 1.0]

    # Test operator_to_iso_vec function: Converts a complex matrix representing an operator into a complex vector.
    @test operator_to_iso_vec(Complex[1.0 0.0; 0.0 1.0]) == iso_vec
    
    # Test iso_operator_to_iso_vec function: Converts a real matrix representing an isomorphism operator into a complex vector.
    @test iso_operator_to_iso_vec(iso_vec_to_iso_operator(iso_vec)) == iso_vec
end
