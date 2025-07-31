#
# This file is licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# Copyright (C) 2024, Advanced Micro Devices, Inc.

import torch
import torch.nn as nn
import sys
import math
from aie.utils.ml import DataShaper
import time
import os
import numpy as np
from aie.utils.xrt import setup_aie, extract_trace, write_out_trace, execute
import aie.utils.test as test_utils


import snntorch as snn
from snntorch import surrogate
from snntorch import backprop
from snntorch import functional as SF
from snntorch import utils
from snntorch import spikeplot as splt
from snntorch import spikegen

torch.use_deterministic_algorithms(True)
torch.manual_seed(0)


def main(opts):
    design = "conv2d"
    xclbin_path = opts.xclbin
    insts_path = opts.instr

    log_folder = "log/"
    if not os.path.exists(log_folder):
        os.makedirs(log_folder)

    width = int(opts.width)
    height = int(opts.height)
    ci = int(opts.in_channels)
    co = int(opts.out_channels)

    ci8 = ci // 8
    co8 = co // 8

    num_iter = 1
    npu_time_total = 0
    npu_time_min = 9999999
    npu_time_max = 0
    trace_size = opts.trace_size
    enable_trace = False if not trace_size else True
    trace_file = "log/trace_" + design + ".txt"
    # ------------------------------------------------------
    # Configure this to match your design's buffer size
    # ------------------------------------------------------
    dtype_in = np.dtype("int8")
    dtype_wts = np.dtype("int8")
    dtype_out = np.dtype("int8")

    ci_co = ci * co
    shape_total_wts = (ci_co, 1)
    shape_in_act = (height, ci8, width, 8)  #'YCXC8' , 'CYX'
    shape_in_wts1 = (co8, ci8, 1, 1, 8, 8)  # out,in,ky,kx,in8,out8
    shape_out = (height, co8, width, 8)

    # ------------------------------------------------------
    # Initialize activation, weights, scaling factor for int8 model
    # ------------------------------------------------------
    T = 1000  # Number of timestamps
    # This could be imagined as a static single frame of an image
    # Generate more spikes by increasing input values or using different encoding
    static_img = torch.randint(20, 100, (1, ci, height, width)).type(torch.FloatTensor)
    spike_train = spikegen.rate(static_img, num_steps=T, gain=0.2)  # Adjust gain

    int_weight = torch.randint(50, 80, (co, ci, 1, 1)).type(torch.FloatTensor)
    # s value
    conv_scale = 7.6294e-06  # scale to convert int8 output to floating point
    # z value
    int8_scale = 0.0078  # scale to convert int8 output to floating point
    min_val = -128
    max_val = 127
    # ------------------------------------------------------
    # Get device, load the xclbin & kernel and register them
    # ------------------------------------------------------
    app = setup_aie(
        xclbin_path,
        insts_path,
        shape_in_act,
        dtype_in,
        shape_total_wts,
        dtype_wts,
        shape_out,
        dtype_out,
        enable_trace=enable_trace,
        trace_size=trace_size,
        trace_after_output=True,
    )

    # ------------------------------------------------------
    # Define your golden reference
    # ------------------------------------------------------
    class conv2d_int_model(nn.Module):
        def __init__(self, in_planes=ci, out_planes=co):
            super(conv2d_int_model, self).__init__()
            self.conv = nn.Conv2d(ci, co, kernel_size=1, bias=False)
            self.lif1 = snn.Leaky(beta=1.0, threshold=10000)  # Modified parameters
    
        def forward(self, x_sequence):
            mem1 = self.lif1.init_leaky()
            output_sequence = []
    
            for t in range(x_sequence.size(0)):
                x = x_sequence[t]
                
                out_int_1 = self.conv(x)
                print(f"Timestep {t}, Conv output mean: {out_int_1.mean().item()}")  # Debug
                
                out_int_2, mem1 = self.lif1(out_int_1, mem1)
                print(f"Timestep {t}, LIF output spikes: {out_int_2.sum().item()}")  # Debug

                #No need to dequantize already discrete value
                #out_quant = out_int_2 * conv_scale
                #out_float = int8_scale * torch.clamp(
                #    torch.round(out_quant / int8_scale), min_val, max_val
                #)
                #output_sequence.append(out_float)
                output_sequence.append(out_int_2)
            
            return torch.stack(output_sequence, dim=0)

    # ------------------------------------------------------
    # Pytorch baseline
    # ------------------------------------------------------
    model = conv2d_int_model()
    model.eval()
    model.conv.weight.data.copy_(int_weight)

    golden_output = model(spike_train)

    print(golden_output)

    # ------------------------------------------------------
    # Reorder input data-layout for AIE
    # ------------------------------------------------------
    ds = DataShaper()
    
    # Take the first timestamp from the spike_train for AIE input
    # Shape of spike_train: (T, 1, C, H, W)
    # We want spike_train[0] which is (1, C, H, W)
    aie_input_tensor = spike_train[0] 
    
    # Squeeze the batch dimension if it's not needed for AIE reordering
    # Shape becomes (C, H, W) if original was (1, C, H, W)
    before_input = aie_input_tensor.squeeze(0).data.numpy().astype(dtype_in)
    
    before_input.tofile(
        log_folder + "/before_ifm_mem_fmt_1x1.txt", sep=",", format="%d"
    )
    ifm_mem_fmt = ds.reorder_mat(before_input, "YCXC8", "CYX")
    ifm_mem_fmt.tofile(log_folder + "/after_ifm_mem_fmt_1x1.txt", sep=",", format="%d")

    wts1 = ds.reorder_mat(int_weight.data.numpy().astype(dtype_wts), "OIYXI8O8", "OIYX")
    total_wts = np.concatenate((wts1), axis=None)
    total_wts.tofile(log_folder + "/weights_mem_fmt_final.txt", sep=",", format="%d")

    # ------------------------------------------------------
    # Main run loop
    # ------------------------------------------------------
    start = time.time_ns()
    # Pass the single volume input to the AIE
    entire_buffer = execute(app, ifm_mem_fmt, total_wts)
    stop = time.time_ns()

    if enable_trace:
        data_buffer, trace_buffer = extract_trace(
            entire_buffer, shape_out, dtype_out, trace_size
        )
        data_buffer = data_buffer * int8_scale
        write_out_trace(trace_buffer, trace_file)
    else:
        data_buffer = entire_buffer * int8_scale
        trace_buffer = None

    npu_time = stop - start
    npu_time_total = npu_time_total + npu_time

    # ------------------------------------------------------
    # Reorder output data-layout from AIE
    # ------------------------------------------------------
    temp_out = data_buffer.reshape(height, co8, width, 8)
    temp_out = ds.reorder_mat(temp_out, "CDYX", "YCXD")
    ofm_mem_fmt = temp_out.reshape(co, height, width)
    if enable_trace:
        ofm_log_filename = "/after_ofm_mem_fmt_final_trace.txt"
    else:
        ofm_log_filename = "/after_ofm_mem_fmt_final.txt"
    ofm_mem_fmt.tofile(
        log_folder + "/after_ofm_mem_fmt_final.txt", sep=",", format="%d"
    )
    
    # The AIE output is a single volume, so we reshape it to match the 
    # expected golden output's single timestamp shape for comparison.
    # Golden output first timestamp shape: (1, C, H, W) after golden_output[0]
    ofm_mem_fmt_out = torch.from_numpy(ofm_mem_fmt).unsqueeze(0) # Adds batch dimension (1, C, H, W)

    # ------------------------------------------------------
    # Compare the AIE output and the golden reference
    # ------------------------------------------------------

    print("\nAvg NPU time: {}us.".format(int((npu_time_total / num_iter) / 1000)))

    # Compare the AIE output (single volume) with the first timestamp of the golden reference
    if np.allclose(
        ofm_mem_fmt_out.detach().numpy(),
        golden_output[0].detach().numpy(), # Compare with the first timestamp of golden_output
        rtol=0,
        atol=2 * int8_scale,
    ):
        print("\nPASS!\n")
        exit(0)
    else:
        print("\nFailed.\n")
        print("\nGolden output (first timestamp) shape:", golden_output[0].shape)
        print("AIE output shape:", ofm_mem_fmt_out.shape)
        
        # Optionally print differences for debugging
        # diff = np.abs(ofm_mem_fmt_out.detach().numpy() - golden_output[0].detach().numpy())
        # print("\nMaximum absolute difference:", np.max(diff))
        # print("Difference matrix (first timestamp):\n", diff)
        
        exit(-1)


if __name__ == "__main__":
    p = test_utils.create_default_argparser()
    p.add_argument(
        "-wd",
        "--width",
        dest="width",
        default=32,
        help="Width of convolution tile",
    )
    p.add_argument(
        "-ht",
        "--height",
        dest="height",
        default=32,
        help="Height of convolution tile",
    )
    p.add_argument(
        "-ic",
        "--in_channels",
        dest="in_channels",
        default=64,
        help="Number of input channels for convolution tile",
    )
    p.add_argument(
        "-oc",
        "--out_channels",
        dest="out_channels",
        default=64,
        help="Number of output channels for convolution tile",
    )
    opts = p.parse_args(sys.argv[1:])
    main(opts)
