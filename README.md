# NPU Backend for Neuromorphic Computing

## Project Description

**Speak, Friend, and Enter: A NPU Backend for Neuromorphic Computing**

**Abstract:**  
Neuromorphic computing is at the frontier of novel breakthrough paradigms for data-driven approaches with low-power and online learning capabilities. The NPU of Ryzen AI technology is an appealing backend for this computation. This project aims to:  
1. Understand the MLIR flow for the NPU.  
2. Explore neuromorphic computing basic computations and develop a set of primitives.  
3. Integrate the [SNNtorch](https://snntorch.readthedocs.io/) framework with a backend on Ryzen AI NPU (and/or GPU) heterogeneous systems.

## Toolbox of the project
- SNNTorch (framework to implement Spiking Neural Network using pytorch based framework)
- AIE MLIR IRON (wrapper to implement the AIE array structure using python)
- [Mini-PC with an AMD Ryzen 9 7949HS](https://store.minisforum.com/products/minisforum-um790-pro)

---

## Team

### Student
- Palladino Vittorio vittorio.palladino@mail.polimi.it

### Supervisors
- Conficconi Davide davide.conficconi@polimi.it
- Sorrentino Giuseppe giuseppe.sorrentino@polimi.it

---

## Installation Guide

### Setting Up Iron API for MLIR AIE on LINUX

Follow these steps to set up the Iron API for MLIR AIE:

1. **Prerequisites**:
   - Ensure you have the latest BIOS on your Mini that enables the NPU usage.

2. **Follow the instruction on the official github page**:
   - [Getting Started for AMD Ryzen™ AI on Linux](https://github.com/Xilinx/mlir-aie#getting-started-for-amd-ryzen-ai-on-linux)
