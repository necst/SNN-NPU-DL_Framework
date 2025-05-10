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
float membrane_potential = 0;

__attribute__((noinline)) void snnNeuron_aie_integer_(int32_t *restrict in, int32_t *restrict out, const float threshold, float decay_factor, float reset, int32_t hard_reset, const int32_t width){
    for(int i = 0; i < width; i++)
    {
        membrane_potential = membrane_potential * decay_factor + in[i];
        if(membrane_potential >= threshold)
        {
            out[i] = 1;
            if(hard_reset == 1)
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

void snnNeuronLineInteger(int32_t *in, int32_t *out, float threshold, float decay_factor, float reset, int32_t hard_reset, int32_t lineWidth) {
  snnNeuron_aie_integer_(in, out, threshold, decay_factor, reset, hard_reset, lineWidth);
}

} // extern "C"
