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

#define input_type int32_t // Input type of the stream
#define required_type float // Type for the computation

constexpr int VECTOR_SIZE = 16;

template <typename T, int N>
__attribute__((noinline)) 
void snn_neuron_aie_simd_(int32_t *restrict in, 
                          int32_t *restrict out,
                          float *restrict in_membrane,
                          float *restrict out_membrane,
                          const float threshold,
                          const float decay_factor,
                          const float reset,
                          const int32_t hard_reset,
                          const int32_t width) {
    aie::vector<float, VECTOR_SIZE> v_reset;

    if (reset == -1)
        v_reset = aie::broadcast<float, VECTOR_SIZE>(0);
    else
        v_reset = aie::broadcast<float, VECTOR_SIZE>(reset);

    const aie::vector<float, VECTOR_SIZE> v_threshold = aie::broadcast<float, VECTOR_SIZE>(threshold);
    const aie::vector<float, VECTOR_SIZE> v_one_float = aie::broadcast<float, VECTOR_SIZE>(1.0f);
    const aie::vector<float, VECTOR_SIZE> v_decay_factor = aie::broadcast<float, VECTOR_SIZE>(decay_factor);

    aie::vector<float, VECTOR_SIZE> g_membrane_potential = aie::zeros<float, VECTOR_SIZE>();

    int32_t* inPtr = in;
    int32_t* outPtr = out;
    float* inMembrane = in_membrane;
    float* outMembrane = out_membrane;

    g_membrane_potential = aie::load_v<VECTOR_SIZE>(inMembrane);

    for (int j = 0; j < width; j += VECTOR_SIZE) {
        chess_prepare_for_pipelining
        chess_loop_range(8, ) {

        // Load input spikes for 16 neurons, INT32
        aie::vector<int32_t, VECTOR_SIZE> v_spikes_int = aie::load_v<VECTOR_SIZE>(inPtr);
        inPtr += VECTOR_SIZE;

        // Convert input spikes from int32_t -> float
        aie::vector<float, VECTOR_SIZE> v_spikes = aie::to_float(v_spikes_int);

        // 1. Decay membrane
        auto acc = aie::mul(g_membrane_potential, v_decay_factor);
        g_membrane_potential = aie::to_vector<float>(acc);

        // 2. Add spikes
        g_membrane_potential = aie::add(g_membrane_potential, v_spikes);

        // 3. Fire mask
        auto v_fire_mask = aie::ge(g_membrane_potential, v_threshold);

        // 4. Reset membrane if fired. hard_reset == 1 to set the membrane to 0
        if(hard_reset == 1)
        {
            g_membrane_potential = aie::select(g_membrane_potential, v_reset, v_fire_mask);
        }
        else
        {
            auto v_subtracted = aie::sub(g_membrane_potential, v_reset);
            g_membrane_potential = aie::select(g_membrane_potential, v_subtracted, v_fire_mask);
        }


        // 5. Generate output spikes
        aie::vector<float, VECTOR_SIZE> v_output_float = aie::select(aie::zeros<float, VECTOR_SIZE>(), v_one_float, v_fire_mask);

        // 6. Convert float spikes back to int32_t
        aie::vector<int32_t, VECTOR_SIZE> v_output =  aie::to_fixed<int32_t>(v_output_float);
    //convert_to_output_type(v_one_float);
    

        // 7. Store output
        aie::store_v(outPtr, v_output);
        outPtr += VECTOR_SIZE;
        }
    }

    aie::store_v(outMembrane, g_membrane_potential);

    event1();  // Optional profiling/event marker
}


extern "C" {


void snnNeuronLineSimd(int32_t *in, int32_t *out, float *inMem, float *outMem, float threshold, float decay_factor, float reset, int32_t hard_reset, int32_t lineWidth){
  snn_neuron_aie_simd_<int32_t, 16>(in, out, inMem, outMem, threshold, decay_factor, reset, hard_reset, lineWidth);
}

} // extern "C"
