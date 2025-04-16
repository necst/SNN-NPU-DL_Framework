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

//To mantain the same membrane accross the multiple subtiles managed
int32_t membrane_potential = 0;


__attribute__((noinline)) 
void snn_neuron_aie_simd_(int32_t *restrict in, 
                          int32_t *restrict out,
                          const int32_t width) {
  constexpr int VECTOR_SIZE = 16;

  // Vector registers for membrane potentials
  
  const aie::vector<int32, VECTOR_SIZE> v_reset = aie::broadcast<int32, VECTOR_SIZE>(0);
  const aie::vector<int32, VECTOR_SIZE> v_threshold = aie::broadcast<int32, VECTOR_SIZE>(10);
  const aie::vector<int32, VECTOR_SIZE> v_one = aie::broadcast<int32, VECTOR_SIZE>(1);
  //aie::vector<int32, 16> v_membrane = aie::zeros<int32, 16>();

  static aie::vector<int32, VECTOR_SIZE> g_membrane_potential = aie::zeros<int32, VECTOR_SIZE>();
    
  int32_t* inPtr = in;
  int32_t* outPtr = out;

  for (int j = 0; j < width; j += VECTOR_SIZE) {
    chess_prepare_for_pipelining
    chess_loop_range(8, ) {

      // Load input spikes for 16 neurons
      aie::vector<int32, VECTOR_SIZE> v_spikes = aie::load_v<VECTOR_SIZE>(inPtr);
      inPtr += VECTOR_SIZE;

      // 1. Update membrane potentials
      g_membrane_potential = aie::add(g_membrane_potential, v_spikes);

      // 2. Generate fire mask
      auto v_fire_mask = aie::ge(g_membrane_potential, v_threshold);

      // 3. Reset membrane where spike occurred
      g_membrane_potential = aie::select(v_reset, g_membrane_potential, v_fire_mask);

      // 4. Output spikes as 1s and 0s
      aie::vector<int32, VECTOR_SIZE> v_output = aie::select(aie::zeros<int32, VECTOR_SIZE>(), v_one, v_fire_mask);


      // Store output
      aie::store_v(outPtr, v_output);
      outPtr += VECTOR_SIZE;
    }
  }

  event1();  // Optional profiling/event marker
}



__attribute__((noinline)) void snnNeuron_aie_integer_(int32_t *restrict in, int32_t *restrict out, const int32_t width){
    int32_t threshold = 10;
    for(int i = 0; i < width; i++)
    {
        membrane_potential = membrane_potential + in[i];
        if(membrane_potential >= threshold)
        {
            out[i] = 1;
            membrane_potential = 0;
        }
        else
        {
            out[i] = 0;
        }
    }
}


extern "C" {

void snnNeuronLineInteger(int32_t *in, int32_t *out, int32_t lineWidth) {
  snnNeuron_aie_integer_(in, out, lineWidth);
}

void snnNeuronLineSimd(int32_t *in, int32_t *out, int32_t lineWidth){
  snn_neuron_aie_simd_(in, out, lineWidth);
}

} // extern "C"
