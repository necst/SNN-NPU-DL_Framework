//===- snn_neuron.cc -------------------------------------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Copyright (C) 2024, Advanced Micro Devices, Inc.
//
//===----------------------------------------------------------------------===//

#define NOCPP

#include <stdint.h>
#include <stdlib.h>

#include <aie_api/aie.hpp>

__attribute__((noinline)) void snnNeuron_aie_integer(int32_t *restrict in, 
                                            int32_t *restrict out,
                                            const int32_t threshold,
                                            const int32_t width) {
  event0();

  // Since i cannot pipelined multiple sum over the same neuron maybe it is beneficial to implement multiple neurons on the same kernel.
  // Using 16 elements per vector since we're working with 32-bit integers
  constexpr int VECTOR_SIZE = 16;

  // Initialize membrane potential
  int32_t membrane_potential = 0;

  v16int32 *restrict outPtr = (v16int32 *)out;
  v16int32 *restrict inPtr = (v16int32 *)in;

  for (int j = 0; j < (width); j += VECTOR_SIZE) {
    chess_prepare_for_pipelining chess_loop_range(6, ) {
      // Load input spikes
      v16int32 input_spikes = *inPtr++;
      v16int32 output_spikes = undef_v16int32();
      
      // Process each element in the vector
      // To have a comparison with what has been implemented in snnTorch use a vector of 16 spikes each of them encoded as 32 bit integer.
      for (int i = 0; i < VECTOR_SIZE; i++) {
        //Take one element out of the vector
        int32_t spike = ext_elem(input_spikes, i);  // One 32-bit spike value
        membrane_potential += spike;

        //Initiliaze the output to zero and verify if there is a spike or not
        int32_t output = 0;
        if (membrane_potential >= threshold) {
          output = 1;
          membrane_potential = 0;
        }
      
        output_spikes = upd_elem(output_spikes, i, output);
      }
      
      
      // Store output vector
      *outPtr++ = output_spikes;
    }
  }

    //Compute the time taken by the main kernel
  event1();
}

extern "C" {

void snnNeuronLineInteger(int32_t *in, int32_t *out, int32_t threshold, int32_t lineWidth) {
  snnNeuron_aie_integer(in, out, threshold, lineWidth);
}

} // extern "C"
