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
    tile_size = 128
    problem_size = in1_size // input_spike(0).nbytes

    # Assertion
    assert in1_size == out_size, "Input and output size must be the same"
    assert reset == -1 or reset > 0, "Reset must be -1 if hard reset is required"
    assert decay_factor <= 1 and decay_factor > 0, "Decay factor must be between 0 and 1"

    
    # Define tensor types
    aie_tile_ty = np.ndarray[(tile_size,), np.dtype[np.int32]]
    all_data_ty = np.ndarray[(problem_size,), np.dtype[np.int32]]

    # Number of sub vector to iterate the worker on
    number_sub_vectors = problem_size // tile_size

    # Object fifo declarations
    of_in_spikes_0 = ObjectFifo(aie_tile_ty, name="in_spikes")

    of_out_spikes_0 = ObjectFifo(aie_tile_ty, name="out_spikes")

    lif_neuron_sisd = Kernel("snnNeuronLineInteger", "scale.o", [aie_tile_ty, aie_tile_ty, np.float32, np.float32, np.float32, np.int32, np.int32],)

    def core_body(of_in_spikes_0, of_out_spikes_0, lif_neuron):
        for _ in range_(number_sub_vectors):
            elem_in_spikes = of_in_spikes_0.acquire(1)
            elem_out = of_out_spikes_0.acquire(1)
            lif_neuron(elem_in_spikes, elem_out, threshold, decay_factor, reset, hard_reset, tile_size)
            of_in_spikes_0.release(1)
            of_out_spikes_0.release(1)

    # Create a worker to run the task
    worker = Worker(core_body, fn_args=[of_in_spikes_0.cons(), of_out_spikes_0.prod(), lif_neuron_sisd])

    # Runtime operations to move data to/from the AIE-array
    rt = Runtime()
    with rt.sequence(all_data_ty, all_data_ty) as (inTensor, outTensor):
        #rt.enable_trace(trace_size, workers=[worker])
        rt.start(worker)
        rt.fill(of_in_spikes_0.prod(), inTensor)
        rt.drain(of_out_spikes_0.cons(), outTensor, wait=True)

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