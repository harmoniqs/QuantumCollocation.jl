export CatSystem
export coherent_ket
export get_cat_controls

function coherent_ket(α::Union{Real, Complex}, levels::Int)::Vector{ComplexF64}
    return [exp(-0.5 * abs2(α)) * α^n / sqrt(factorial(n)) for n in 0:levels-1]
end

"""
    CatSystem(;
        g2::Real=0.36,
        χ_aa::Real=-7e-3,
        χ_bb::Real=-32,
        χ_ab::Real=0.79,
        κa::Real=53e-3,
        κb::Real=13,
        cat_levels::Int=13,
        buffer_levels::Int=3,
        prefactor::Real=1,
    )::OpenQuantumSystem

Returns an `OpenQuantumSystem` for a quantum cat.
"""
function CatSystem(;
    g2::Real=0.36,
    χ_aa::Real=-7e-3,
    χ_bb::Real=-32,
    χ_ab::Real=0.79,
    κa::Real=53e-3,
    κb::Real=13,
    cat_levels::Int=13,
    buffer_levels::Int=3,
    prefactor::Real=1,
)

    # Cat ⊗ Buffer
    a = annihilate(cat_levels) ⊗ Matrix(1.0I, buffer_levels, buffer_levels)
    b = Matrix(1.0I, cat_levels, cat_levels) ⊗ annihilate(buffer_levels)

    H_drift = -χ_aa/2 * a'a'a*a - χ_bb/2 * b'b'b*b - χ_ab * a'a*b'b + g2 * a'a'b + conj(g2) * a*a*b'

    # buffer drive, kerr-correction drive
    H_drives = [b + b', a'a]

    L_dissipators = [√κa * a, √κb * b]

    H_drift *= 2π
    H_drives .*= 2π
    L_dissipators .*= sqrt(2π)

    return OpenQuantumSystem(
        H_drift,
        H_drives,
        L_dissipators;
    )
end
