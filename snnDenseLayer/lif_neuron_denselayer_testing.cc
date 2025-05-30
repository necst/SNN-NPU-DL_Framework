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

// Used since iteration in the for does not work correctly
float hardcoded_retrieve(aie::vector<float, 16> vector, int32_t index) {
    float spike_val = 0.0f;
    switch(index) {
        case 0:  spike_val = extract_elem(vector, 0);  break;
        case 1:  spike_val = extract_elem(vector, 1);  break;
        case 2:  spike_val = extract_elem(vector, 2);  break;
        case 3:  spike_val = extract_elem(vector, 3);  break;
        case 4:  spike_val = extract_elem(vector, 4);  break;
        case 5:  spike_val = extract_elem(vector, 5);  break;
        case 6:  spike_val = extract_elem(vector, 6);  break;
        case 7:  spike_val = extract_elem(vector, 7);  break;
        case 8:  spike_val = extract_elem(vector, 8);  break;
        case 9:  spike_val = extract_elem(vector, 9);  break;
        case 10: spike_val = extract_elem(vector, 10); break;
        case 11: spike_val = extract_elem(vector, 11); break;
        case 12: spike_val = extract_elem(vector, 12); break;
        case 13: spike_val = extract_elem(vector, 13); break;
        case 14: spike_val = extract_elem(vector, 14); break;
        case 15: spike_val = extract_elem(vector, 15); break;
        default: spike_val = 0.0f; break;
    }
    return spike_val;
}

__attribute__((noinline)) 
void snn_neuron_aie_simd_(int32_t *restrict in, 
                          int32_t *restrict out,
                          float *restrict in_membrane,
                          float *restrict out_membrane,
                          float *restrict in_weights,
                          int32_t input_size,
                          int32_t output_size,
                          const float threshold,
                          const float decay_factor,
                          const float reset,
                          const int32_t hard_reset,
                          const int32_t width) {
    const int OUTPUT_SIZE = input_size;
    const int INPUT_SIZE = output_size;
    const int DEFAULT_SIZE = 16;

    aie::vector<float, DEFAULT_SIZE> v_reset;
    // Initialize vectors
    if (reset == -1)
        v_reset = aie::broadcast<float, DEFAULT_SIZE>(0);
    else
        v_reset = aie::broadcast<float, DEFAULT_SIZE>(reset);

    const aie::vector<float, DEFAULT_SIZE> v_threshold = 
        aie::broadcast<float, DEFAULT_SIZE>(threshold);
    const aie::vector<float, DEFAULT_SIZE> v_one_float = 
        aie::broadcast<float, DEFAULT_SIZE>(1.0f);
    const aie::vector<float, DEFAULT_SIZE> v_decay_factor = 
        aie::broadcast<float, DEFAULT_SIZE>(decay_factor);

    aie::vector<float, DEFAULT_SIZE> g_membrane_potential = aie::zeros<float, DEFAULT_SIZE>();

    int32_t* inPtr = in;
    int32_t* outPtr = out;
    float* inWeightsPtr = in_weights; 
    float* inMembrane = in_membrane;
    float* outMembrane = out_membrane;

    g_membrane_potential = aie::load_v<DEFAULT_SIZE>(inMembrane);

    for (int j = 0; j < width; j += DEFAULT_SIZE) {
        chess_prepare_for_pipelining
        chess_loop_range(8,) {
            // Load and convert input spikes
            aie::vector<int32_t, DEFAULT_SIZE> v_spikes_int = 
                aie::load_v<DEFAULT_SIZE>(inPtr);
            inPtr += DEFAULT_SIZE;
            
            // Convert to float (suppress warnings with explicit template)
            auto v_spikes = aie::to_float<float>(v_spikes_int);

            // 1. Decay membrane
            auto acc = aie::mul(g_membrane_potential, v_decay_factor);
            g_membrane_potential = aie::to_vector<float>(acc);

            //int32_t index = 0;
         // 2. Process weights row-wise (one output neuron at a time)
            for (int32_t index = 0; index < INPUT_SIZE; ++index) {
                aie::vector<float, DEFAULT_SIZE> weights_column = 
                    aie::load_v<DEFAULT_SIZE>(inWeightsPtr + index * DEFAULT_SIZE);

                float spike_val = extract_elem(v_spikes, index);
                aie::vector<float, DEFAULT_SIZE> v_input = aie::broadcast<float, DEFAULT_SIZE>(spike_val);
            
                auto weights_input_acc = aie::mul(weights_column, v_input);
                auto weights_input_vec = aie::to_vector<float>(weights_input_acc);
                g_membrane_potential = aie::add(g_membrane_potential, weights_input_vec);
            }
   /*     
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
            aie::vector<float, DEFAULT_SIZE> v_output_float = aie::select(aie::zeros<float, DEFAULT_SIZE>(), v_one_float, v_fire_mask);
*/
            // 6. Convert to fixed-point
            aie::vector<int32_t, DEFAULT_SIZE> v_output = 
            aie::to_fixed<int32_t>(g_membrane_potential);

            // 7. Store output
            aie::store_v(outPtr, v_output);
            outPtr += DEFAULT_SIZE;
        }
    
   }

    // Store final membrane potential when the function has finished
    aie::store_v(out_membrane, g_membrane_potential);
}

extern "C" {

void snnNeuronLineSimdInputHidden(int32_t *in, int32_t *out, float *inMem, float *outMem, float *inWeights, int32_t input_layer_size, int32_t output_layer_size, float threshold, float decay_factor, float reset, int32_t hard_reset, int32_t width){
  snn_neuron_aie_simd_(in, out, inMem, outMem, inWeights, input_layer_size, output_layer_size, threshold, decay_factor, reset, hard_reset, width);
}

void snnNeuronLineSimdHiddenOutput(int32_t *in, int32_t *out, float *inMem, float *outMem, float *inWeights, int input_layer_size, int output_layer_size, float threshold, float decay_factor, float reset, int32_t hard_reset, int32_t width){
  snn_neuron_aie_simd_(in, out, inMem, outMem, inWeights, input_layer_size, output_layer_size, threshold, decay_factor, reset, hard_reset, width);
}

} // extern "C"
