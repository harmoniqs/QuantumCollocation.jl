using QuantumCollocation
using PiccoloQuantumObjects

pages = [
    "Home" => "index.md",
    "Quantum Systems" => [
        "Quantum Systems" => "generated/tmp/pqo/quantum_systems.md",
    ],
    "Quantum System Operations" => [
        "Isomorphisms" => "generated/tmp/pqo/isomorphisms.md",
        "Rollout" => "generated/tmp/pqo/rollouts.md",
    ],

    "Examples" => [
        "Two Qubit Gates" => "generated/examples/two_qubit_gates.md",
        "Multilevel Transmon" => "generated/examples/multilevel_transmon.md",
    ],
    "Library" => [
        "Ket Problem Templates" => "generated/man/ket_problem_templates.md",
        "Unitary Problem Templates" => "generated/man/unitary_problem_templates.md",
    ],
]

# Check if utils.jl exists and warn if not found
utils_path = joinpath(@__DIR__, "utils.jl")
if !isfile(utils_path)
    error("docs/utils.jl is required but not found. Please run get_docs_utils.sh")
end

include("utils.jl")

generate_docs(
    @__DIR__,
    "QuantumCollocation",
    [QuantumCollocation, PiccoloQuantumObjects],
    pages;
    make_index = false,
    make_assets = false,
    makedocs_kwargs = (draft = true,),
)