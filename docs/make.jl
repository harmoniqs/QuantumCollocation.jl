using QuantumCollocation
using PiccoloDocsTemplate

pages = [
    "Home" => "index.md",
    "Manual" => [
        "Problem Templates Overview" => "generated/man/problem_templates_overview.md",
        "Ket Problem Templates" => "generated/man/ket_problem_templates.md",
        "Unitary Problem Templates" => "generated/man/unitary_problem_templates.md",
        "Working with Solutions" => "generated/man/working_with_solutions.md",
        "PiccoloOptions Reference" => "generated/man/piccolo_options.md",
    ],
    "Examples" => [
        "Single Qubit Gate" => "generated/examples/single_qubit_gate.md",
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