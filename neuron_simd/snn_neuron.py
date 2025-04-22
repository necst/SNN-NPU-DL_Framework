# snn_neuron/snn_neuron.py -*- Python -*-
#
# This file is licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# (c) Copyright 2024 Advanced Micro Devices, Inc. or its affiliates
import numpy as np
import sys

from aie.iron import ObjectFifo, Program, Runtime, Worker, Kernel
from aie.iron.placers import SequentialPlacer
from aie.iron.device import NPU1Col1, NPU2
from aie.iron.controlflow import range_

#Number of elements (32 bit integer)
PROBLEM_SIZE = 1024
AIE_TILE_WIDTH = 128
MEMBRANE_SIZE = 16

if len(sys.argv) > 2:
    if sys.argv[1] == "npu":
        dev = NPU1Col1()
    elif sys.argv[1] == "npu2":
        dev = NPU2Col1()
    else:
        raise ValueError("[ERROR] Device name {} is unknown".format(sys.argv[1]))


def snn_neuron(dev):
    # Define tensor types
    aie_tile_ty = np.ndarray[(AIE_TILE_WIDTH,), np.dtype[np.int32]]
    all_data_ty = np.ndarray[(PROBLEM_SIZE,), np.dtype[np.int32]]
    membrane_ty = np.ndarray[(MEMBRANE_SIZE,), np.dtype[np.int32]]

    number_of_cycle = PROBLEM_SIZE // AIE_TILE_WIDTH


    # TODO check wheter the input size need to be all_data_ty
    # Object fifo for the input spikes
    # Use the mem tiles to forward the data and make an implicit copy, instead of passing directly from the shim tiles
    # To be clear: object fifo between shim tiles (L3) and compute tiles(L1)
    of_in_spikes_0 = ObjectFifo(aie_tile_ty, name="in_spikes")

    # object fifo between compute tiles and shit tiles
    of_out_spikes_0 = ObjectFifo(aie_tile_ty, name="out_spikes")

    of_in_membrane = ObjectFifo(membrane_ty, name="input_membrane", default_depth = 2)

    # Define the kernel function to call
    lif_neuron_sisd = Kernel("snnNeuronLineInteger","scale.o", [aie_tile_ty, aie_tile_ty, np.int32],)
    
    lif_neuron_simd = Kernel("snnNeuronLineSimd", "scale.o", [aie_tile_ty, aie_tile_ty, membrane_ty, membrane_ty, np.int32],)
    
    # Define a compute task to perform
    def core_body(of_in_spikes_0, of_out_spikes_0, of_in_membrane, of_out_membrane, lif_neuron):
        init_mem = of_out_membrane.acquire(1)
        for i in range_(MEMBRANE_SIZE):
            init_mem[i] = 0
        of_out_membrane.release(1)
        for _ in range_(number_of_cycle):
            elem_in_spikes = of_in_spikes_0.acquire(1)
            elem_out = of_out_spikes_0.acquire(1)
            elem_in_membrane = of_in_membrane.acquire(1)
            elem_out_membrane = of_out_membrane.acquire(1)
            lif_neuron(elem_in_spikes, elem_out, elem_in_membrane, elem_out_membrane, AIE_TILE_WIDTH)
            of_in_spikes_0.release(1)
            of_out_spikes_0.release(1)
            of_in_membrane.release(1)
            of_out_membrane.release(1)

    # Create a worker to run the task
    worker = Worker(core_body, fn_args=[of_in_spikes_0.cons(), of_out_spikes_0.prod(), of_in_membrane.cons(), of_in_membrane.prod(), lif_neuron_simd])

    # Runtime operations to move data to/from the AIE-array
    rt = Runtime()
    with rt.sequence(all_data_ty, all_data_ty) as (inTensor, outTensor):
        rt.start(worker)
        rt.fill(of_in_spikes_0.prod(), inTensor)
        rt.drain(of_out_spikes_0.cons(), outTensor, wait=True)

    # Place program components (assign them resources on the device) and generate an MLIR module
    return Program(dev, rt).resolve_program(SequentialPlacer())

dev = NPU1Col1()
module = snn_neuron(dev)
res = module.operation.verify()
if res == True:
    print(module)
else:
    print(res)