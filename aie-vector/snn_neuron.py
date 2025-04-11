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

PROBLEM_SIZE = 1024
#MEM_TILE_WIDTH = 64
AIE_TILE_WIDTH = 32
THRESHOLD_SIZE = 1

if len(sys.argv) > 1:
    if sys.argv[1] == "npu":
        dev = NPU1Col1()
    elif sys.argv[1] == "npu2":
        dev = NPU2()
    else:
        raise ValueError("[ERROR] Device name {} is unknown".format(sys.argv[1]))


def snn_neuron():
    # Define tensor types
    aie_tile_ty = np.ndarray[(AIE_TILE_WIDTH,), np.dtype[np.int32]]
    all_data_ty = np.ndarray[(PROBLEM_SIZE,), np.dtype[np.int32]]
    threshold_ty = np.ndarray[(THRESHOLD_SIZE,), np.dtype[np.int32]]


    # TODO check wheter the input size need to be all_data_ty
    # Object fifo for the input spikes
    # Use the mem tiles to forward the data and make an implicit copy, instead of passing directly from the shim tiles
    # To be clear: object fifo between shim tiles (L3) and compute tiles(L1)
    of_in_spikes_0 = ObjectFifo(aie_tile_ty, name="in_spikes")

    # Object fifo for the input parameter (in this first implementation only the threshold is passed) No need of the ping pong buffer
    # depth could be one in this case, no ping pong needed
    of_in_threshold = ObjectFifo(threshold_ty, name="in_threshold")

    # object fifo between compute tiles and mem tiles
    of_out_spikes_0 = ObjectFifo(aie_tile_ty, name="out_threshold")


    # Define the kernel function to call
    lif_neuron = Kernel(
        "snnNeuronLineInteger",
        "lif_kernel_single32",
        [aie_tile_ty, threshold_ty, aie_tile_ty, np.int32],
    )
    
    # Define a compute task to perform
    def core_body(of_in_spikes_0, of_in_threshold, of_out_spikes_0):
        # TODO check wheter it works without a loop of all data / aie tile
        elem_in_spikes = of_in_spikes_0.acquire(1)
        elem_in_threshold = of_in_threshold.acquire(1)
        elem_out = of_out_spikes_0.acquire(1)
        lif_neuron(elem_in_spikes, elem_in_threshold, elem_out)
        of_in_spikes_0.release(1)
        of_in_threshold.release(1)
        of_out_spikes_0.release(1)

    # Create a worker to run the task
    worker = Worker(core_body, fn_args=[of_in_spikes_0.cons(), of_out_spikes_0.prod()])

    # Runtime operations to move data to/from the AIE-array
    rt = Runtime()
    with rt.sequence(all_data_ty, threshold_ty, all_data_ty) as (inTensor, inThreshold, outTensor):
        rt.start(worker)
        rt.fill(of_input_spikes_0.prod(), inTensor)
        rt.fill(of_input_threshold, inThreshold)
        rt.drain(of_output_spikes_0.cons(), outTensor, wait=True)

    # Place program components (assign them resources on the device) and generate an MLIR module
    return Program(dev, rt).resolve_program(SequentialPlacer())


module = snn_neuron()
res = module.operation.verify()
if res == True:
    print(module)
else:
    print(res)