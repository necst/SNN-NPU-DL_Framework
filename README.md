# Speak, Friend, and Enter — NPU Backend for Neuromorphic Computing

## Project Overview

This project implements a backend for neuromorphic (spiking) workloads targeting the AMD Ryzen™ AI NPU (and optionally GPU) using MLIR / AIE tools and the SNNtorch framework.

### Goals

* Understand the MLIR flow for the AIE/NPU.
* Define and implement basic neuromorphic primitives (spiking neurons, synaptic updates, encoders/decoders).
* Integrate **SNNtorch** with a backend that compiles to Ryzen AI NPU (and supports heterogeneous GPU+NPU runs).

---

## Toolbox

* **SNNtorch** — PyTorch-based SNN framework. [https://snntorch.readthedocs.io/](https://snntorch.readthedocs.io/)
* **AIE MLIR IRON API** — Python wrapper to describe and compile AIE arrays.
* **Mini-PC** with **AMD Ryzen™ 9 7949HS** (example target hardware).

---

## Quick start / Prerequisites

1. A Linux machine with NPU support (BIOS updated to enable the NPU).
2. Python 3.10+ (or compatible with SNNtorch and your MLIR tools).
3. `git`, `make`, and a working C++ toolchain (g++/clang) for the testbench.
4. Follow the MLIR-AIE repo instructions (below) to install toolchains and dependencies.

---

## Installation

> See: Getting Started for AMD Ryzen™ AI on Linux: [https://github.com/Xilinx/mlir-aie#getting-started-for-amd-ryzen-ai-on-linux](https://github.com/Xilinx/mlir-aie#getting-started-for-amd-ryzen-ai-on-linux)

Follow the guide and once the env is ready execute the following lines to enter inside the folder where the source code is located.

```bash
# move the example source into the expected folder layout
cd OpenHW_deliver/
```

---

## Building and running (command line)

From the example folder (the one containing the Makefile):

```bash
# compile the design
make

# run the C++ testbench
make run
```

---

## Jupyter Notebook (alternative run)

A documented Jupyter notebook is included in the repository. To run it:

```bash
# start jupyter in the repo root
jupyter lab
# or
jupyter notebook
```

Open the provided notebook and follow the cells to build, compile and run examples using the MLIR-AIE flow.

---

## Project layout (suggested)

```
/ (repo root)
├─ notebooks/                # examples & experiments (Jupyter)
├─ snn_primitives/           # Python implementations of neuron, synapse, encoders
├─ mlir_aie/                 # IRON wrapper code & MLIR generation scripts
├─ examples/                 # end-to-end SNN models and compilation examples
└─ README.md
```

---

## Example usage (Python sketch)

This short sketch shows how to import SNNtorch models and call a hypothetical MLIR-AIE compiler API. Adapt names to the actual API in your repo.

```python
from snntorch import surrogate
import torch
# import your wrapper that lowers to AIE/MLIR
from mlir_aie import aie_compiler

# build or load an snn model (PyTorch + SNNtorch)
model = ...  # SNNtorch model

# trace or lower model to MLIR and compile to AIE binary
aie_module = aie_compiler.lower_model(model, input_shape=(1, 3, 32, 32))
aie_binary = aie_compiler.compile(aie_module)

# deploy / run on target (NPU) — pseudocode
aie_compiler.deploy(aie_binary, target='ryzen_ai')
result = aie_compiler.run_on_target(aie_binary, input_data)
```

---

## Tips & troubleshooting

* Ensure the BIOS and kernel drivers expose the NPU on your platform.
* Match PyTorch / SNNtorch versions with your Python version.
* If `make` fails, inspect the Makefile and required paths in `mlir-aie` — dependencies or environment variables may be missing.

---

## Contributing

* Open issues for bugs or feature requests.
* Make PRs against `main` with a clear description and tests when possible.

---

## License

Add your preferred license here (e.g., MIT, Apache-2.0).

---

## Contacts

* Student: Palladino Vittorio — [vittorio.palladino@mail.polimi.it](mailto:vittorio.palladino@mail.polimi.it)
* Supervisors: Conficconi Davide — [davide.conficconi@polimi.it](mailto:davide.conficconi@polimi.it)
  Sorrentino Giuseppe — [giuseppe.sorrentino@polimi.it](mailto:giuseppe.sorrentino@polimi.it)

---
