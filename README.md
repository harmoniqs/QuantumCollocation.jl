<!--```@raw html-->
<div align="center">
  <a href="https://github.com/harmoniqs/Piccolo.jl">
    <img src="assets/logo.svg" alt="Piccolo.jl" width="25%"/>
  </a>
</div>

<div align="center">
  <table>
    <tr>
      <td align="center">
        <b>Documentation</b>
        <br>
        <a href="https://docs.harmoniqs.co/QuantumCollocation/dev/">
          <img src="https://img.shields.io/badge/docs-stable-blue.svg" alt="Stable"/>
        </a>
        <a href="https://docs.harmoniqs.co/QuantumCollocation/dev/">
          <img src="https://img.shields.io/badge/docs-dev-blue.svg" alt="Dev"/>
        </a>
        <a href="https://arxiv.org/abs/2305.03261">
          <img src="https://img.shields.io/badge/arXiv-2305.03261-b31b1b.svg" alt="arXiv"/>
        </a>
      </td>
      <td align="center">
        <b>Build Status</b>
        <br>
        <a href="https://github.com/harmoniqs/QuantumCollocation.jl/actions/workflows/CI.yml?query=branch%3Amain">
          <img src="https://github.com/harmoniqs/QuantumCollocation.jl/actions/workflows/CI.yml/badge.svg?branch=main" alt="Build Status"/>
        </a>
        <a href="https://codecov.io/gh/harmoniqs/QuantumCollocation.jl">
          <img src="https://codecov.io/gh/harmoniqs/QuantumCollocation.jl/branch/main/graph/badge.svg" alt="Coverage"/>
        </a>
      </td>
      <td align="center">
        <b>License</b>
        <br>
        <a href="https://opensource.org/licenses/MIT">
          <img src="https://img.shields.io/badge/License-MIT-yellow.svg" alt="MIT License"/>
        </a>
      </td>
      <td align="center">
        <b>Support</b>
        <br>
        <a href="https://unitary.fund">
          <img src="https://img.shields.io/badge/Supported%20By-Unitary%20Fund-FFFF00.svg" alt="Unitary Fund"/>
        </a>
      </td>
    </tr>
  </table>
</div>

<div align="center">
  <i> Quickly set up and solve problem templates for quantum optimal control</i>
  <br>
</div>
<!--```-->

# QuantumCollocation.jl

**QuantumCollocation.jl** sets up and solves *quantum control problems* as nonlinear programs (NLPs). In this context, a generic quantum control problem looks like
```math
\begin{aligned}
    \arg \min_{\mathbf{Z}}\quad & J(\mathbf{Z}) \\
    \nonumber \text{s.t.}\qquad & \mathbf{f}(\mathbf{Z}) = 0 \\
    \nonumber & \mathbf{g}(\mathbf{Z}) \le 0  
\end{aligned}
```
where $\mathbf{Z}$ is a trajectory  containing states and controls, from [NamedTrajectories.jl](https://github.com/harmoniqs/NamedTrajectories.jl).

For details of our implementation please see our IEEE QCE 2023 paper, [Direct Collocation for Quantum Optimal Control](https://arxiv.org/abs/2305.03261). If you use QuantumCollocation.jl in your work, please cite :raised_hands:!

## Installation

This package is registered! To install, enter the Julia REPL, type `]` to enter pkg mode, and then run:
```julia
pkg> add QuantumCollocation
```

## Example

### Single Qubit Hadamard Gate
```Julia
using QuantumCollocation

T = 50
Δt = 0.2
system = QuantumSystem([PAULIS[:X], PAULIS[:Y]])
U_goal = GATES.H

# Hadamard Gate
prob = UnitarySmoothPulseProblem(system, U_goal, T, Δt)
solve!(prob, max_iter=100)
```


### Building Documentation
This package uses a Documenter config that is shared with many of our other repositories. To build the docs, you will need to run the docs setup script to clone and pull down the utility. 
```
# first time only
./docs/get_docs_utils.sh   # or ./get_docs_utils.sh if cwd is in ./docs/
```

To build the docs pages:
```
julia --project=docs docs/make.jl
```

or editing the docs live:
```
julia --project=docs
> using LiveServer, QuantumCollocation, Revise
> servedocs(literate_dir="docs/literate", skip_dirs=["docs/src/generated"])
```

> **Note:** `servedocs` needs to watch a subset of the files in the `docs/` folder. If it watches files that are generated on a docs build/re-build, `servedocs` will continuously try to re-serve the pages.
> 
> To prevent this, ensure all generated files are included in the skip dirs or skip files args for `servedocs`.

For example, if we forget docs/src/generated like so:
```
julia --project=docs
> using LiveServer, Piccolo, Revise
> servedocs(literate_dir="docs/literate")
```
it will not build and serve.