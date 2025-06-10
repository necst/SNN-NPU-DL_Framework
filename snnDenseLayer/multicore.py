# snn_neuron/snn_neuron.py -*- Python -*-
#
# This file is licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# (c) Copyright 2024 Advanced Micro Devices, Inc. or its affiliates
import numpy as np
import argparse
import sys

from aie.iron import ObjectFifo, Program, Runtime, Worker, Kernel
from aie.iron.placers import SequentialPlacer
from aie.iron.device import NPU1Col1, NPU2
from aie.iron.controlflow import range_

def snn_neuron(dev, in1_size, out_size, threshold, decay_factor, reset, hard_reset, trace_size):

    input_spike = np.float32
    out_spike = np.float32
    tile_size = 256
    mem_size = 512
    # Depending on the size of the aie register
    membrane_size = 16
    problem_size = in1_size // input_spike(0).nbytes
    n_cores = 2

    # Assertion
    assert in1_size == out_size, "Input and output size must be the same"
    assert membrane_size == 16, "Membrane buffer must have the same size as the AIE register till now"
    assert reset == -1 or reset > 0, "Reset must be -1 if hard reset is required"
    #assert decay_factor <= 1 and decay_factor > 0, "Decay factor must be between 0 and 1"
    assert n_cores % 2 == 0, "Num of cores must be a multiple of 2"

    
    # Define tensor types
    aie_tile_ty = np.ndarray[(tile_size,), np.dtype[np.int32]]
    all_data_ty = np.ndarray[(problem_size,), np.dtype[np.int32]]
    mem_tile_ty = np.ndarray[(mem_size,), np.dtype[np.int32]]
    membrane_ty = np.ndarray[(membrane_size,), np.dtype[np.float32]]
    
    # Number of sub vector to iterate the worker on
    number_sub_vectors = problem_size // mem_size

    # Object fifo declarations
    
    # L3 -> L2
    of_in_spikes_L3_L2 = ObjectFifo(mem_tile_ty, name="in_spikes")

    # L2 -> L3
    of_out_spikes_L2_L3 = ObjectFifo(mem_tile_ty, name="out_spikes")

    # L1 -> L1 membrane accumulation
    of_in_membrane = [ObjectFifo(membrane_ty, name=f"in_out_membrane_{i}", default_depth=2) for i in range(n_cores)]

    # Offset to access the data coming from the memtile to the compute tiles
    of_offsets = [(mem_size // n_cores) * i for i in range(n_cores)]
    
    # Produce a list of two buffer objects
    # L2 -> L1
    of_in_spikes_L2_L1 = of_in_spikes_L3_L2.cons().split(
        of_offsets,
        obj_types=[aie_tile_ty] * n_cores,
        names=[f"obj_L2_L1_neuron_{i}" for i in range(n_cores)],
    )

    # L1 -> L2
    of_out_spikes_L1_L2 = of_out_spikes_L2_L3.prod().join(
        of_offsets,
        obj_types=[aie_tile_ty] * n_cores,
        names=[f"obj_L1_L2_neuron_{i}" for i in range(n_cores)],
    )
    
    vectorized = False
    if(vectorized):
        unit = "Simd"
    else:
        unit = "Scalar"

    lif_neuron_simd = Kernel(f"snnNeuronLine{unit}", "scale.o", [aie_tile_ty, aie_tile_ty, membrane_ty, membrane_ty, np.float32, np.float32, np.float32, np.int32, np.int32],)
    
    def core_body(of_in_spikes, of_out_spikes, of_in_membrane, of_out_membrane, lif_neuron):
        init_mem = of_out_membrane.acquire(1)
        for i in range_(membrane_size):
            init_mem[i] = 0
        of_out_membrane.release(1)
        for _ in range_(number_sub_vectors):
            elem_in_spikes = of_in_spikes.acquire(1)
            elem_out = of_out_spikes.acquire(1)
            elem_in_membrane = of_in_membrane.acquire(1)
            elem_out_membrane = of_out_membrane.acquire(1)
            lif_neuron(elem_in_spikes, elem_out, elem_in_membrane, elem_out_membrane, threshold, decay_factor, reset, hard_reset, tile_size)
            of_in_spikes.release(1)
            of_out_spikes.release(1)
            of_in_membrane.release(1)
            of_out_membrane.release(1)

    # Create a list of workers
    workers = []

    for i in range(n_cores):
        # Append a worker to the list to run the task
        workers.append(Worker(core_body, fn_args=[of_in_spikes_L2_L1[i].cons(), of_out_spikes_L1_L2[i].prod(), of_in_membrane[i].cons(), of_in_membrane[i].prod(), lif_neuron_simd]))

    # Runtime operations to move data to/from the AIE-array
    rt = Runtime()
    with rt.sequence(all_data_ty, all_data_ty) as (inTensor, outTensor):
        #rt.enable_trace(trace_size, workers=[worker])
        rt.start(*workers)
        rt.fill(of_in_spikes_L3_L2.prod(), inTensor)
        rt.drain(of_out_spikes_L2_L3.cons(), outTensor, wait=True)

    # Place program components (assign them resources on the device) and generate an MLIR module
    return Program(dev, rt).resolve_program(SequentialPlacer())

if len(sys.argv) < 3:
    raise ValueError(
        "[ERROR] Need at least 3 arguments (dev, in1_size, out_size, threshold, decay factor, reset factor)"
    )

# Create an argument parser
p = argparse.ArgumentParser()
p.add_argument("-d", "--dev", required=True, dest="device", help="AIE Device")
p.add_argument("-i1s", "--in1_size", required=True, dest="in1_size", help="Input 1 size of the spike")
p.add_argument("-os", "--out_size", required=True, dest="out_size", help="Output size of the spike")
p.add_argument("-th", "--threshold", required=True, dest="threshold", help="Threshold of the neurons")
p.add_argument("-df", "--decay_factor", required=True, dest="decay_factor", help="Decay factor of the neurons")
p.add_argument("-rs", "--reset", required=True, dest="reset_factor", help="Reset factor")
p.add_argument("-hr", "--hard_reset", required=True, dest="hard_reset", help="Equal to one for hard reset")
p.add_argument("-t", "--trace_size", required=False, dest="trace_size", default=0, help="Trace buffer size")

opts = p.parse_args(sys.argv[1:])

in1_size = int(opts.in1_size)
if in1_size % 128:
    print("Input size of the spike must be a multiple of 128 (so lenght is a multiple of 64")

if opts.device == "npu":
    dev = NPU1Col1()
elif opts.device == "npu2":
    dev = NPU2Col1()
else:
    raise ValueError("[ERROR] Device name {} is unknown".format(opts.device))

out_size = int(opts.out_size)
threshold = float(opts.threshold)
decay_factor = float(opts.decay_factor)
reset_factor = float(opts.reset_factor)
hard_reset = int(opts.hard_reset)
trace_size = int(opts.trace_size)

module = snn_neuron(dev, in1_size, out_size, threshold, decay_factor, reset_factor, hard_reset, trace_size)
res = module.operation.verify()
if res == True:
    print(module)
else:
    print(res)
