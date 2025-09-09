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

---

## Installation

> See: Getting Started for AMD Ryzen™ AI on Linux: [https://github.com/Xilinx/mlir-aie#getting-started-for-amd-ryzen-ai-on-linux](https://github.com/Xilinx/mlir-aie#getting-started-for-amd-ryzen-ai-on-linux)

Here i leave the command needed to install the env, taken from the previous link to the mlir repo.


## Initial Setup

  > Be sure you have the latest BIOS on your laptop or mini-PC that enables the NPU. See [here](#update-bios).

If starting from `Ubuntu 24.04` you may need to update the Linux kernel to 6.11+ by installing the Hardware Enablement (HWE) stack:

  ```bash
  sudo apt update
  sudo apt install --install-recommends linux-generic-hwe-24.04
  sudo reboot
  ```

## Prerequisites

### BIOS Settings:

Turn off SecureBoot (Allows for unsigned drivers to be installed):
   ```BIOS → Security → Secure boot → Disable```

### Build and install the XDNA™ Driver and XRT

1. Execute the scripted build process:

    > This script will install package dependencies, build the xdna-driver and xrt packages, and install them. *These steps require `sudo` access.*

    ```bash
    bash ./utils/build_drivers.sh
    ```

1. Reboot as directed after the script exits.

    ```bash
    sudo reboot
    ```

1. Check that the NPU is working if the device appears with xrt-smi:

   ```bash
   source /opt/xilinx/xrt/setup.sh
   xrt-smi examine
   ```

   > At the bottom of the output you should see:
   >  ```
   >  Devices present
   >  BDF             :  Name
   > ------------------------------------
   >  [0000:66:00.1]  :  NPU Strix
   >  ```

### Install IRON and MLIR-AIE Prerequisites

1. Install the following packages needed for MLIR-AIE:

    ```bash
    # Python versions 3.10, 3.12 and 3.13 are currently supported by our wheels
    sudo apt install \
    build-essential clang clang-14 lld lld-14 cmake ninja-build python3-venv python3-pip
    ```

## Install IRON for AMD Ryzen™ AI AIE Application Development

1. Clone [the mlir-aie repository](https://github.com/Xilinx/mlir-aie.git):
   ```bash
   git clone https://github.com/Xilinx/mlir-aie.git
   cd mlir-aie
   ```

1. Setup a virtual environment:
   ```bash
   python3 -m venv ironenv
   source ironenv/bin/activate
   python3 -m pip install --upgrade pip
   ```

1. Install IRON library, mlir-aie and llvm-aie compilers from wheels and dependencies:

   For release v1.0:
   ```bash
   # Install IRON library and mlir-aie from a wheel
   python3 -m pip install mlir_aie -f https://github.com/Xilinx/mlir-aie/releases/expanded_assets/v1.0

   # Install Peano from a llvm-aie wheel
   python3 -m pip install https://github.com/Xilinx/llvm-aie/releases/download/nightly/llvm_aie-19.0.0.2025041501+b2a279c1-py3-none-manylinux_2_27_x86_64.manylinux_2_28_x86_64.whl

   # Install basic Python requirements (still needed for release v1.0, but is no longer needed for latest wheels)
   python3 -m pip install -r python/requirements.txt

   # Install MLIR Python Extras
   HOST_MLIR_PYTHON_PACKAGE_PREFIX=aie python3 -m pip install -r python/requirements_extras.txt
   ```

   For daily latest:
   ```bash
   # Install IRON library and mlir-aie from a wheel
   python3 -m pip install mlir_aie -f https://github.com/Xilinx/mlir-aie/releases/expanded_assets/latest-wheels-2

   # Install Peano from llvm-aie wheel
   python3 -m pip install llvm-aie -f https://github.com/Xilinx/llvm-aie/releases/expanded_assets/nightly

   # Install MLIR Python Extras
   HOST_MLIR_PYTHON_PACKAGE_PREFIX=aie python3 -m pip install -r python/requirements_extras.txt
   ```


1. Setup environment
   ```bash
   source utils/env_setup.sh
   ```


1. Go inside the delivery folder of the openHW
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

A documented Jupyter notebook is included in the repository. This will exploit the wrapper written in python to directly call the npu kernel from the notebook, using a small library.

Open the provided notebook and follow the cells to build, compile and run examples using the MLIR-AIE flow.

---

## Project layout explanation

```
/ (repo root)
├─ OpenHWDelivery
    └─ Makefile
    └─ denselayer.py #design for the feedforward network
    └─ singlecore.py #design for the single aie core 
    └─ multicore.py #design for the multi aie core
    └─ lif_kernel_denselayer.cc #kernel implementaion of the feedforward neural network layer
    └─ lif_kernel_singlecore.cc #implementation of the singlecore kernel (vectorized and scalar)
    └─ lif_kernel_multicore.cc #implemenation of the multicore kernel (vector and scalar)
    └─ test.cpp
    └─ ... The following files have been taken from the repo mlir and has the only role to set up correctly all the utils and library of the AIE cores.
    └─ cxxopts.hpp #file to 
    └─ test_utils.cpp
    └─ test_utils.h
    └─ makefile-common
    └─ ...
└─ README.md
```

---

## Tips & troubleshooting

* Ensure the BIOS and kernel drivers expose the NPU on your platform.
* Match PyTorch / SNNtorch versions with your Python version.
* If `make` fails, inspect the Makefile and required paths in `mlir-aie` — dependencies or environment variables may be missing.

---

## License

Add your preferred license here (e.g., MIT, Apache-2.0).

---

## Contacts

* Student: Palladino Vittorio — [vittorio.palladino@mail.polimi.it](mailto:vittorio.palladino@mail.polimi.it)
* Supervisors: Conficconi Davide — [davide.conficconi@polimi.it](mailto:davide.conficconi@polimi.it)
  Sorrentino Giuseppe — [giuseppe.sorrentino@polimi.it](mailto:giuseppe.sorrentino@polimi.it)

---
