using QuantumCollocation
using PiccoloDocsTemplate

pages = [
    "Home" => "index.md",
    "Manual" => [
        "Ket Problem Templates" => "generated/man/ket_problem_templates.md",
        "Unitary Problem Templates" => "generated/man/unitary_problem_templates.md",
    ],
    "Examples" => [
        "Two Qubit Gates" => "generated/examples/two_qubit_gates.md",
        "Multilevel Transmon" => "generated/examples/multilevel_transmon.md",
    ],
    "Library" => "lib.md",
]

generate_docs(
    @__DIR__,
    "QuantumCollocation",
    [QuantumCollocation],
    pages;
    make_index = false,
    make_assets = false,
    format_kwargs = (canonical = "https://docs.harmoniqs.co/QuantumCollocation.jl",),
)