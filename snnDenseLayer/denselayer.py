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
    input_layer = 4
    hidden_layer_1 = 16
    output_layer = 4
    membrane_size = 16
    input_all = in1_size // input_spike(0).nbytes
    output_all = (input_all // input_layer)*(output_layer + hidden_layer_1)
    weights_all = (input_layer * hidden_layer_1 + hidden_layer_1 * output_layer)
    n_layer = 3 # including input layer

    # Assertion
    assert membrane_size == 16, "Membrane buffer must have the same size as the AIE register till now"
    assert reset == -1 or reset > 0, "Reset must be -1 if hard reset is required"
    assert decay_factor <= 1 and decay_factor > 0, "Decay factor must be between 0 and 1"
    assert input_all % input_layer == 0, "Input size must be a multiple of the input layer"

    
    # Define tensor types
    input_tile_ty = np.ndarray[(input_layer,), np.dtype[np.int32]]
    input_all_data_ty = np.ndarray[(input_all,), np.dtype[np.int32]]
    output_all_data_ty = np.ndarray[(output_all,), np.dtype[np.int32]]
    mem_tile_ty = np.ndarray[(mem_size,), np.dtype[np.int32]]
    
    # For quantization reduce the precision of the weight
    weight_tile_all_ty = np.ndarray[(weights_all,), np.dtype[np.float32]]
    membrane_hidden_ty = np.ndarray[(16,), np.dtype[np.float32]]
    membrane_output_ty = np.ndarray[(4,), np.dtype[np.float32]]
    hidden_1_output_ty = np.ndarray[(hidden_layer_1,), np.dtype[np.int32]]
    output_layer_ty = np.ndarray[(output_layer,), np.dtype[np.int32]]
    weight_input_hidden_ty = np.ndarray[(input_layer * hidden_layer_1,), np.dtype[np.float32]]
    weight_hidden_output_ty = np.ndarray[(hidden_layer_1 * output_layer,), np.dtype[np.float32]]  # hidden->output weights
    
    # Number of sub vector to iterate the worker on
    number_sub_vectors = input_all // input_layer

    # Object fifo declarations
    
    # L3 -> L2
    of_in_spikes_L3_L2 = ObjectFifo(mem_tile_ty, name="in_spikes")

    # L2 -> L3
    of_out_spikes_L2_L3 = ObjectFifo(mem_tile_ty, name="out_spikes")

    # L3 -> L2
    of_in_weights_all_L3_L2 = ObjectFifo(weight_tile_all_ty, name="input_weights_all")

    # L1 -> L1 membrane accumulation
    of_in_membrane_hidden = ObjectFifo(membrane_hidden_ty, name="in_membrane", default_depth=2)
    of_in_membrane_output = ObjectFifo(membrane_output_ty, name="in_membrane_output", default_depth=2)
    
    # L1 hidden_layer_1 -> L1 output_layer
    of_spikes_hidden_1_output = ObjectFifo(hidden_1_output_ty, name="hidden_1_output")
    
    # L2 -> L1
    of_in_weight_L2_L1 = of_in_weights_all_L3_L2.cons().split(
        offsets=[0, input_layer * hidden_layer_1],
        depths=[1, 1],
        obj_types=[
            weight_input_hidden_ty,  # input->hidden weights
            weight_hidden_output_ty  # hidden->output weights
        ],
        names=["input_hidden_1_weights", "hidden_1_output_weights"]
    )
    
    # L2 -> L1
    of_in_spikes_L2_L1 = of_in_spikes_L3_L2.cons().forward(
        obj_type=input_tile_ty,
        name=f"obj_L2_L1_layer",
    )

    # L1 -> L2
    of_out_spikes_L1_L2 = of_out_spikes_L2_L3.prod().join(
        offsets=[0, hidden_layer_1],
        obj_types=[
            hidden_1_output_ty,
            output_layer_ty
        ],
        names=[f"obj_L1_L2_layer_{i}" for i in range(n_layer - 1)],
    )
    
    def core_body(of_in_spikes, of_out_spikes, of_in_membrane, of_out_membrane, of_in_weights_L2_L1, input_layer_size, output_layer_size, lif_neuron):
        init_mem = of_out_membrane.acquire(1)
        weights = of_in_weights_L2_L1.acquire(1)
        for i in range_(membrane_size):
            init_mem[i] = 0
        of_out_membrane.release(1)
        
        for _ in range_(number_sub_vectors):
            elem_in_spikes = of_in_spikes.acquire(1)
            elem_out = of_out_spikes.acquire(1)
            elem_in_membrane = of_in_membrane.acquire(1)
            elem_out_membrane = of_out_membrane.acquire(1)
            lif_neuron(elem_in_spikes, elem_out, elem_in_membrane, elem_out_membrane, weights, input_layer_size, output_layer_size, threshold, decay_factor, reset, hard_reset, tile_size)
            of_in_spikes.release(1)
            of_out_spikes.release(1)
            of_in_membrane.release(1)
            of_out_membrane.release(1)
            
        of_in_weights_L2_L1.release(1)

    # Create a list of workers
    workers = []

    # Define the kernels outside the worker append, or pass them as variables
    snn_kernel_input_hidden = Kernel("snnNeuronLineSimdInputHidden", "scale.o", [input_tile_ty, hidden_1_output_ty, membrane_hidden_ty, membrane_hidden_ty, weight_input_hidden_ty, np.int32, np.int32, np.float32, np.float32, np.float32, np.int32, np.int32],)
    snn_kernel_hidden_output = Kernel("snnNeuronLineSimdHiddenOutput", "scale.o", [hidden_1_output_ty, output_layer_ty, membrane_output_ty, membrane_output_ty, weight_hidden_output_ty, np.int32, np.int32, np.float32, np.float32, np.float32, np.int32, np.int32],)


    # Input -> Hidden_1
    workers.append(Worker(core_body, fn_args=[of_in_spikes_L2_L1.cons(), of_out_spikes_L1_L2[0].prod(), of_in_membrane_hidden.cons(), of_in_membrane_hidden.prod(), of_in_weight_L2_L1[0].cons(), input_layer, hidden_layer_1, snn_kernel_input_hidden]))

    # Hidden_1 -> Output
    workers.append(Worker(core_body, fn_args=[of_out_spikes_L1_L2[0].cons(), of_out_spikes_L1_L2[1].prod(), of_in_membrane_output.cons(), of_in_membrane_output.prod(), of_in_weight_L2_L1[1].cons(), hidden_layer_1, output_layer, snn_kernel_hidden_output]))

    # Runtime operations to move data to/from the AIE-array
    rt = Runtime()
    with rt.sequence(input_all_data_ty, weight_tile_all_ty, output_all_data_ty) as (inTensor, weightTensor, outTensor):
        #rt.enable_trace(trace_size, workers=[worker])
        rt.start(*workers)
        rt.fill(of_in_spikes_L3_L2.prod(), inTensor)
        rt.fill(of_in_weights_all_L3_L2.prod(), weightTensor)
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
