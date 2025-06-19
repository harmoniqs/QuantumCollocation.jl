using QuantumCollocation
using Documenter
using Literate

push!(LOAD_PATH, joinpath(@__DIR__, "..", "src"))

@info "Building Documenter site for QuantumCollocation.jl"

pages = [
    "Home" => "index.md",
    "Examples" => [
        "Two Qubit Gates" => "generated/examples/two_qubit_gates.md",
    ],
    "Library" => [
        "Ket Problem Templates" => "generated/man/ket_problem_templates.md",
        "Unitary Problem Templates" => "generated/man/unitary_problem_templates.md",
    ],
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
    size_threshold=4_000_000,
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
    modules=[QuantumCollocation],
    authors="Aaron Trowbridge <aaron.j.trowbridge@gmail.com> and contributors",
    sitename="QuantumCollocation.jl",
    format=format,
    pages=pages,
    warnonly=true,
)

deploydocs(;
    repo="github.com/harmoniqs/QuantumCollocation.jl.git",
    devbranch="main",
)
