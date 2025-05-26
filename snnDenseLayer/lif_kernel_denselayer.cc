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

template <typename T, int INPUT_LAYER, int OUTPUT_LAYER>
__attribute__((noinline)) 
void snn_neuron_aie_simd_(int32_t *restrict in, 
                          int32_t *restrict out,
                          float *restrict in_membrane,
                          float *restrict out_membrane,
                          float *restrict in_weights,
                          const float threshold,
                          const float decay_factor,
                          const float reset,
                          const int32_t hard_reset,
                          const int32_t width) {
    constexpr int OUTPUT_SIZE = OUTPUT_LAYER;
    constexpr int INPUT_SIZE = INPUT_LAYER;


    aie::vector<float, OUTPUT_SIZE> v_reset;
    // Initialize vectors
    if (reset == -1)
        v_reset = aie::broadcast<float, OUTPUT_SIZE>(0);
    else
        v_reset = aie::broadcast<float, OUTPUT_SIZE>(reset);

    const aie::vector<float, OUTPUT_SIZE> v_threshold = 
        aie::broadcast<float, OUTPUT_SIZE>(threshold);
    const aie::vector<float, OUTPUT_SIZE> v_one_float = 
        aie::broadcast<float, OUTPUT_SIZE>(1.0f);
    const aie::vector<float, OUTPUT_SIZE> v_decay_factor = 
        aie::broadcast<float, OUTPUT_SIZE>(decay_factor);

    aie::vector<float, OUTPUT_SIZE> g_membrane_potential = aie::zeros<float, OUTPUT_SIZE>();

    int32_t* inPtr = in;
    int32_t* outPtr = out;
    float* inMembrane = in_membrane;
    float* outMembrane = out_membrane;

    g_membrane_potential = aie::load_v<OUTPUT_SIZE>(inMembrane);

    for (int j = 0; j < width; j += INPUT_SIZE) {
        chess_prepare_for_pipelining
        chess_loop_range(8, ) {
            // Load and convert input spikes
            aie::vector<int32_t, INPUT_SIZE> v_spikes_int = 
                aie::load_v<INPUT_SIZE>(inPtr);
            inPtr += INPUT_LAYER;
            
            // Convert to float (suppress warnings with explicit template)
            auto v_spikes = aie::to_float<float>(v_spikes_int);

            // 1. Decay membrane
            auto acc = aie::mul(g_membrane_potential, v_decay_factor);
            g_membrane_potential = aie::to_vector<float>(acc);
        
            // 2. Process weights and spikes
            for (int i = 0; i < INPUT_LAYER; ++i) {
                aie::vector<float, OUTPUT_LAYER> weight_column = 
                    aie::load_v<OUTPUT_LAYER>(in_weights + i * OUTPUT_LAYER);
                
                // Multiply and add
                auto weighted_input_accum = aie::mul(weight_column, v_spikes[i]);
                weight_column = aie::to_vector<float>(weighted_input_accum);
                g_membrane_potential = aie::add(g_membrane_potential, weight_column);
            }

            // 3. Fire mask
            auto v_fire_mask = aie::ge(g_membrane_potential, v_threshold);

            // 4. Reset membrane
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
            // Generate output spikes (corrected version)
            aie::vector<float, OUTPUT_SIZE> v_output_float = aie::select(aie::zeros<float, OUTPUT_SIZE>(), v_one_float, v_fire_mask);

            // 6. Convert to fixed-point
            aie::vector<int32_t, OUTPUT_SIZE> v_output = 
                aie::to_fixed<int32_t>(v_output_float);

            // 7. Store output
            aie::store_v(outPtr, v_output);
            outPtr += OUTPUT_SIZE;
        }
    }

    // Store final membrane potential
    aie::store_v(out_membrane, g_membrane_potential);
}


extern "C" {

void snnNeuronLineSimdInputHidden(int32_t *in, int32_t *out, float *inMem, float *outMem, float *inWeights, int input_layer_size, int output_layer_size, float threshold, float decay_factor, float reset, int32_t hard_reset, int32_t width){
  snn_neuron_aie_simd_<int32_t, 16, 16>(in, out, inMem, outMem, inWeights, threshold, decay_factor, reset, hard_reset, width);
}

void snnNeuronLineSimdHiddenOutput(int32_t *in, int32_t *out, float *inMem, float *outMem, float *inWeights, int input_layer_size, int output_layer_size, float threshold, float decay_factor, float reset, int32_t hard_reset, int32_t width){
  snn_neuron_aie_simd_<int32_t, 16, 16>(in, out, inMem, outMem, inWeights, threshold, decay_factor, reset, hard_reset, width);
}

} // extern "C"
