using QuantumCollocation
using PiccoloQuantumObjects
using Documenter
using Literate

push!(LOAD_PATH, joinpath(@__DIR__, "..", "src"))

@info "Building Documenter site for QuantumCollocation.jl"

pages = [
    "Home" => "index.md",
    "Quantum Systems" => [
        "Quantum Systems" => "generated/tmp/pqo/quantum_systems.md",
    ],
    "Quantum System Operations" => [
        "Isomorphisms" => "generated/tmp/pqo/isomorphisms.md",
        "Rollout" => "generated/tmp/pqo/rollouts.md",
    ],
    
    # "Examples" => [
    #     "Two Qubit Gates" => "generated/examples/two_qubit_gates.md",
    #     "Multilevel Transmon" => "generated/examples/multilevel_transmon.md",
    # ],
    # "Library" => [
    #     "Ket Problem Templates" => "generated/man/ket_problem_templates.md",
    #     "Unitary Problem Templates" => "generated/man/unitary_problem_templates.md",
    # ],
]

format = Documenter.HTML(;
    prettyurls=get(ENV, "CI", "false") == "true",
    canonical="https://docs.harmoniqs.co/QuantumCollocation.jl",
    edit_link="main",
    assets=String[],
    mathengine = MathJax3(Dict(
        :loader => Dict("load" => ["[tex]/physics"]),
        :tex => Dict(
            "inlineMath" => [["\$","\$"], ["\\(","\\)"]],
            "tags" => "ams",
            "packages" => [
                "base",
                "ams",
                "autoload",
                "physics"
            ],
        ),
    )),
    # size_threshold=4_000_000,
)

src = joinpath(@__DIR__, "src")
lit = joinpath(@__DIR__, "literate")

lit_output = joinpath(src, "generated")

for (root, _, files) ∈ walkdir(lit), file ∈ files
    splitext(file)[2] == ".jl" || continue
    ipath = joinpath(root, file)
    opath = splitdir(replace(ipath, lit=>lit_output))[1]
    Literate.markdown(ipath, opath)
end

makedocs(;
    # modules=[QuantumCollocation],
    modules=[QuantumCollocation, PiccoloQuantumObjects],
    authors="Aaron Trowbridge <aaron.j.trowbridge@gmail.com> and contributors",
    sitename="QuantumCollocation.jl",
    format=format,
    pages=pages,
    pagesonly=true,
    warnonly=true,
)

deploydocs(;
    repo="github.com/harmoniqs/QuantumCollocation.jl.git",
    devbranch="main",
)
