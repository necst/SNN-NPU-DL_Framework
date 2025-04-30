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

constexpr int VECTOR_SIZE = 16;

//To mantain the same membrane accross the multiple subtiles managed
int32_t membrane_potential = 0;


__attribute__((noinline)) 
void snn_neuron_aie_simd_(int32_t *restrict in, 
                          int32_t *restrict out,
                          int32_t *restrict in_membrane,
                          int32_t *restrict out_membrane,
                          const int32_t threshold,
                          const int32_t decay_factor,
                          const int32_t reset,
                          const int32_t width) {
  //constexpr int VECTOR_SIZE = 16;

  // Vector registers for membrane potentials

  aie::vector<int32, VECTOR_SIZE> v_reset;
    
  if(reset == -1)
  {
    v_reset = aie::broadcast<int32, VECTOR_SIZE>(0);
  }
  else
  {
    v_reset = aie::broadcast<int32, VECTOR_SIZE>(reset); 
  }
      
  const aie::vector<int32, VECTOR_SIZE> v_threshold = aie::broadcast<int32, VECTOR_SIZE>(threshold);
  const aie::vector<int32, VECTOR_SIZE> v_one = aie::broadcast<int32, VECTOR_SIZE>(1);
  const aie::vector<int32, VECTOR_SIZE> v_decay_factor = aie::broadcast<int32, VECTOR_SIZE>(decay_factor);
  aie::vector<int32, VECTOR_SIZE> g_membrane_potential = aie::zeros<int32, VECTOR_SIZE>();

    
  int32_t* inPtr = in;
  int32_t* outPtr = out;
  int32_t* inMembrane = in_membrane;
  int32_t* outMembrane = out_membrane;

  g_membrane_potential = aie::load_v<VECTOR_SIZE>(inMembrane);
    
  for (int j = 0; j < width; j += VECTOR_SIZE) {
    chess_prepare_for_pipelining
    chess_loop_range(8, ) {

      // Load input spikes for 16 neurons
      aie::vector<int32, VECTOR_SIZE> v_spikes = aie::load_v<VECTOR_SIZE>(inPtr);
      
      inPtr += VECTOR_SIZE;

      // 1. Multiply the membrane by the decay factor

      auto acc = aie::mul(g_membrane_potential, v_decay_factor);
      // Mul return an accumulator vector
      g_membrane_potential = aie::to_vector<int32>(acc);


      // 2. Update membrane potentials
      g_membrane_potential = aie::add(g_membrane_potential, v_spikes);

      // 3. Generate fire mask
      auto v_fire_mask = aie::ge(g_membrane_potential, v_threshold);

      // 4. Reset membrane where spike occurred (if reset = -1 hard reset is        activated and membrane set to 0
      if(reset == -1)
      {
         g_membrane_potential = aie::select(g_membrane_potential, v_reset, v_fire_mask);
      }else
      {
         auto v_subtracted = aie::sub(g_membrane_potential, v_reset);
g_membrane_potential = aie::select(g_membrane_potential, v_subtracted, v_fire_mask);

      }
      
      // 5. Output spikes as 1s and 0s
      aie::vector<int32, VECTOR_SIZE> v_output = aie::select(aie::zeros<int32, VECTOR_SIZE>(), v_one, v_fire_mask);

      // 6. Store output
      aie::store_v(outPtr, v_output);
      outPtr += VECTOR_SIZE;
    }
    
  }
    aie::store_v(outMembrane, g_membrane_potential);

  event1();  // Optional profiling/event marker
}



__attribute__((noinline)) void snnNeuron_aie_integer_(int32_t *restrict in, int32_t *restrict out, const int32_t threshold, int32_t decay_factor, int32_t reset, const int32_t width){
    for(int i = 0; i < width; i++)
    {
        membrane_potential = membrane_potential * decay_factor + in[i];
        if(membrane_potential >= threshold)
        {
            out[i] = 1;
            if(reset == -1)
            {
                membrane_potential = 0;
            }else
            {
                membrane_potential -= reset;
            }
        }
        else
        {
            out[i] = 0;
        }
    }
}


extern "C" {

void snnNeuronLineInteger(int32_t *in, int32_t *out, int32_t threshold, int32_t decay_factor, int32_t reset, int32_t lineWidth) {
  snnNeuron_aie_integer_(in, out, threshold, decay_factor, reset, lineWidth);
}

void snnNeuronLineSimd(int32_t *in, int32_t *out, int32_t *inMem, int32_t *outMem, int32_t threshold, int32_t decay_factor, int32_t reset, int32_t lineWidth){
  snn_neuron_aie_simd_(in, out, inMem, outMem, threshold, decay_factor, reset, lineWidth);
}

} // extern "C"
