//===- conv2dk1_i8.cc -------------------------------------------------*- C++
//-*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Copyright (C) 2022-2025, Advanced Micro Devices, Inc.
//
//===----------------------------------------------------------------------===//

#define NOCPP

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

#include "../aie_kernel_utils.h"
#include <aie_api/aie.hpp>

#define REL_WRITE 0
#define REL_READ 1

// Only include Vector implementation
#ifndef SCALAR

#ifdef INT8_ACT

//*****************************************************************************
// conv2d 1x1 - vector with SNN neuron logic
// act: int8, wts: int8, out: int8 (spikes)
// membrane: int32 (internal state)
//
// Assume IC >= 16 as that gives ideal inner loop schedule
//
// TODO - Restricting input_width is mutiple of 32
// Because each VMAC works on 4 inputs at a time and we store intermediate
// results in 8 accumulators, having input_width be a multiple of 4*8=32 is
// ideal. However, we should be able to support input_width that is only a
// multiple of 4 but there is some strange scheduling happening now so for
// now, we do not.
//*****************************************************************************
void conv2dk1_i8_vector(int8_t *input, int8_t *kernels, int8_t *output_spikes, // output is now spikes (0 or 1)
                        int32_t *in_membrane_potential,  // Input membrane potential for this time step
                        int32_t *out_membrane_potential, // Output membrane potential for next time step
                        const int32_t input_width, const int32_t input_channels,
                        const int32_t output_channels,
                        const int scale, // Scaling for threshold comparison
                        const int32_t threshold,
                        const int32_t decay_factor_int, // Integer representation of decay (e.g., 2^N)
                        const int32_t reset_value,      // Value to reset membrane to if hard_reset
                        const int32_t hard_reset_flag) { // Flag for hard or soft reset
  event0();

  constexpr int NUM_ACC = 8; // Number of accumulators
  constexpr int MMUL_M = 4;  // Matrix A M size in MxK (Input width)
  constexpr int MMUL_K = 8;
  constexpr int MMUL_N = 8;
  constexpr int CHANNEL_FACTOR = MMUL_K;
  constexpr int MMUL_MK = MMUL_M * MMUL_K; // 4 * 8 = 32
  constexpr int MMUL_KN = MMUL_K * MMUL_N; // 8 * 8 = 64
  constexpr int MMUL_MN = MMUL_M * MMUL_N; // 4 * 8 = 32 (size of one output block)

  // Note: For SNN, saturation/rounding mode applies to the *spike output*,
  // not necessarily the internal membrane potential which can be int32.
  // We'll apply a custom threshold/reset/spike logic.
  // The MMUL itself still saturates/rounds intermediate results if they
  // overflow acc32 range, but int8 * int8 -> acc32 shouldn't overflow acc32.
  ::aie::set_saturation(aie::saturation_mode::saturate);
  ::aie::set_rounding(aie::rounding_mode::symmetric_inf);

  int8_t *restrict out_spikes_ptr = output_spikes;
  int32_t *restrict in_mem_ptr = in_membrane_potential;
  int32_t *restrict out_mem_ptr = out_membrane_potential;

  MMUL8x8x8 acc_tmp[NUM_ACC]; // These will hold the current membrane potentials

  // Broadcast constants for SNN logic
  // These will be applied to the 32-bit membrane potential
  const aie::vector<int32_t, MMUL_MN> v_threshold_int = aie::broadcast<int32_t, MMUL_MN>(threshold);
  const aie::vector<int32_t, MMUL_MN> v_reset_int = aie::broadcast<int32_t, MMUL_MN>(reset_value);
  const aie::vector<int32_t, MMUL_MN> v_zero_int32 = aie::broadcast<int32_t, MMUL_MN>(0);
  const aie::vector<int8_t, MMUL_MN> v_one_int8 = aie::broadcast<int8_t, MMUL_MN>(1);
  const aie::vector<int8_t, MMUL_MN> v_zero_int8 = aie::broadcast<int8_t, MMUL_MN>(0);

  const int iw = input_width;
  const int iw_partial = (input_width / MMUL_M) / NUM_ACC;

  // Assertions for input_width alignment
  assert((input_width / MMUL_M) % NUM_ACC == 0);
  const int iw_partial_rem = 0; // TODO - See restriction

  assert((input_channels / CHANNEL_FACTOR) > 2); // Assume IC >= 16

  int8_t *input_begin_ptr = input;
  // This pointer adjustment for remainder is likely not needed if iw_partial_rem is 0
  // int8_t *input_rem_begin_ptr = input + iw_partial * MMUL_M * NUM_ACC * CHANNEL_FACTOR;


  // Pointers for current batch within the full buffer
  // Adjust these to point to the correct sections of input/output/membrane buffers
  int8_t *current_input_ptr = input;
  int32_t *current_in_mem_ptr_base = in_mem_ptr;
  int32_t *current_out_mem_ptr_base = out_mem_ptr;
  int8_t *current_out_spikes_ptr_base = out_spikes_ptr;


  // Loop over output channels (groups of 8)
  for (int oc = 0; oc < (output_channels / CHANNEL_FACTOR); oc++) {

    // Loop over input width blocks (groups of MMUL_M * NUM_ACC)
    for (int iw_block_idx = 0; iw_block_idx < iw_partial; iw_block_idx++) {

        // --- SNN Membrane Load and Pre-processing ---
        for (int x = 0; x < NUM_ACC; x++) { // For each of the 8 accumulators (representing 8*4=32 neurons)
            // Load previous membrane potential for this block of neurons
            // Assuming membrane_potential_buffer is laid out similar to output
            acc_tmp[x] = aie::load_v<acc32, MMUL_MN>(current_in_mem_ptr_base + (iw_block_idx * NUM_ACC * MMUL_MN) + (x * MMUL_MN));

            // Apply decay *before* adding new input
            // Shift right by decay_factor_int bits to simulate division by 2^decay_factor_int
            acc_tmp[x].to_vector<int32_t>() = aie::srs(acc_tmp[x].to_vector<int32_t>(), decay_factor_int);
        }

        // --- Convolutional Summation ---
        AIE_PREPARE_FOR_PIPELINING
        AIE_LOOP_MIN_ITERATION_COUNT(2) // Ensure pipeline fills
        for (int ic = 0; ic < (input_channels / CHANNEL_FACTOR); ic++) {
          aie::vector<int8, MMUL_KN> in_b = aie::load_v<MMUL_KN>(kernels); // Load weights for MMUL_K input, MMUL_N output channels
          kernels += MMUL_KN;

          for (int x = 0; x < NUM_ACC; x++) { // For each of the 8 accumulators
            aie::vector<int8, MMUL_MK> in_a = aie::load_v<MMUL_MK>(current_input_ptr); // Load input activations
            current_input_ptr += MMUL_MK; // Move input pointer
            acc_tmp[x].mac(in_a, in_b); // Accumulate weighted sum onto existing potential
          }
          // Move current_input_ptr to the next input channel block for the same input_width block
          current_input_ptr += (iw * CHANNEL_FACTOR) - (MMUL_MK * NUM_ACC);
        }

        // --- SNN Neuron Fire and Reset Logic ---
        for (int xx = 0; xx < NUM_ACC; xx++) {
          aie::vector<int32_t, MMUL_MN> current_potential = acc_tmp[xx].to_vector<int32_t>(); // Get 32-bit potential

 
          auto v_fire_mask = aie::ge(current_potential, v_threshold_int);

          // Reset membrane potential based on hard_reset_flag
          aie::vector<int32_t, MMUL_MN> next_potential;
          if (hard_reset_flag == 1) {
            next_potential = aie::select(current_potential, v_reset_int, v_fire_mask); // If fired, reset; else, keep
          } else { // Soft reset (subtract threshold)
            // Note: v_reset_int here acts as the value to subtract for soft reset.
            // If the original 'reset' parameter meant just a reset value, we might
            // need another parameter for soft reset amount. Assuming v_reset_int = threshold for soft reset.
            auto v_subtracted = aie::sub(current_potential, v_threshold_int); // Subtract threshold
            next_potential = aie::select(current_potential, v_subtracted, v_fire_mask); // If fired, subtract; else, keep
          }

          // Generate output spikes (0 or 1)
          aie::vector<int8_t, MMUL_MN> o_spikes = aie::select(v_zero_int8, v_one_int8, v_fire_mask);

          // Store the binary spikes
          aie::store_v(out_spikes_ptr, o_spikes);
          out_spikes_ptr += MMUL_MN;

          // Store the updated membrane potentials back to the buffer
          aie::store_v(out_mem_ptr, next_potential);
          out_mem_ptr += MMUL_MN;
        }

        // Reset input pointer for the next input_width block (for the next output channel)
        current_input_ptr = input_begin_ptr + (oc * input_channels * iw); // Reset for next ic block
    }

    // After processing all iw_partial blocks for the current oc:
    // Move pointers to the start of the next output channel block
    // These large jumps ensure we are correctly indexing into the full 3D buffers
    current_input_ptr = input_begin_ptr; // Reset input for next oc block
    kernels -= (input_channels / CHANNEL_FACTOR) * MMUL_KN; // Reset kernel pointer for next oc block calculation
                                                            // (this loop advances kernels, so need to rewind to start of current oc, then advance for next oc)
    kernels += (input_channels / CHANNEL_FACTOR) * MMUL_KN * (output_channels / CHANNEL_FACTOR - oc); // Adjust kernels for next oc block
      
    current_in_mem_ptr_base += (iw_partial * NUM_ACC * MMUL_MN);
    current_out_mem_ptr_base += (iw_partial * NUM_ACC * MMUL_MN);
    current_out_spikes_ptr_base += (iw_partial * NUM_ACC * MMUL_MN);
  }

  event1();
}
#endif // INT8_ACT
#endif // Vector

//*****************************************************************************
// conv2d 1x1 wrappers
//*****************************************************************************
extern "C" {

// Only include Vector wrapper as requested
#ifndef SCALAR

#ifdef INT8_ACT

void conv2dk1_i8(int8_t *input, int8_t *kernels, int8_t *output_spikes,
                 int32_t *in_membrane_potential, int32_t *out_membrane_potential,
                 const int32_t input_width, const int32_t input_channels,
                 const int32_t output_channels, const int scale,
                 const int32_t threshold, const int32_t decay_factor_int,
                 const int32_t reset_value, const int32_t hard_reset_flag) {
  conv2dk1_i8_vector(input, kernels, output_spikes,
                     in_membrane_potential, out_membrane_potential,
                     input_width, input_channels, output_channels,
                     scale, threshold, decay_factor_int,
                     reset_value, hard_reset_flag);
}
#endif // INT8_ACT
#endif // Vector
} // extern "C"
