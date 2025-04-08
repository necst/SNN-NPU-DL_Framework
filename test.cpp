//===- test.cpp -------------------------------------------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Copyright (C) 2023, Advanced Micro Devices, Inc.
//
//===----------------------------------------------------------------------===//

#include <cstdint>
#include <fstream>
#include <iostream>
#include <sstream>

#include "test_utils.h"
#include "xrt/xrt_bo.h"

#ifndef DATATYPES_USING_DEFINED
#define DATATYPES_USING_DEFINED
using DATATYPE = std::uint32_t; // Configure this to match your buffer data type
#endif

namespace po = boost::program_options;
const int threshold = 0.5;

int main(int argc, const char *argv[]) {

  // Program arguments parsing
  po::options_description desc("Allowed options");
  po::variables_map vm;
  test_utils::add_default_options(desc);

  test_utils::parse_options(argc, argv, desc, vm);
  int verbosity = vm["verbosity"].as<int>();

  // Declaring design constants
  constexpr bool VERIFY = true;
  constexpr int IN_SIZE = 1024;
  constexpr int OUT_SIZE = IN_SIZE;


  // Load instruction sequence
  std::vector<uint32_t> instr_v =
      test_utils::load_instr_binary(vm["instr"].as<std::string>());

  if (verbosity >= 1)
    std::cout << "Sequence instr count: " << instr_v.size() << "\n";

  // Start the XRT context and load the kernel
  xrt::device device;
  xrt::kernel kernel;

  test_utils::init_xrt_load_kernel(device, kernel, verbosity,
                                   vm["xclbin"].as<std::string>(),
                                   vm["kernel"].as<std::string>());


  // Setting up the object buffers

  auto bo_instr = xrt::bo(device, instr_v.size() * sizeof(int),
                          XCL_BO_FLAGS_CACHEABLE, kernel.group_id(1));
  auto bo_in_spikes = xrt::bo(device, IN_SIZE * sizeof(DATATYPE),
                        XRT_BO_FLAGS_HOST_ONLY, kernel.group_id(3));
  auto bo_threshold = xrt::bo(device, 1 * sizeof(DATATYPE),
                             XRT_BO_FLAGS_HOST_ONLY, kernel.group_id(4));
  auto bo_out_spikes = xrt::bo(device, OUT_SIZE * sizeof(DATATYPE),
                         XRT_BO_FLAGS_HOST_ONLY, kernel.group_id(5));

  if (verbosity >= 1)
    std::cout << "Writing data into buffer objects.\n";

  // Copy instruction stream to xrt buffer object
  void *bufInstr = bo_instr.map<void *>();
  memcpy(bufInstr, instr_v.data(), instr_v.size() * sizeof(int));

  // Initialize buffer bo_inA
  // Initialize with the input spikes
  DATATYPE *buf_in_spikes = bo_in_spikes.map<DATATYPE *>();
  
  //TODO find a better way to generate the input spikes
  std::mt19937 gen(42); // Fixed seed for reproducibility
  std::bernoulli_distribution dist(0.1);
  
  for (int t = 0; t < TIMESTEPS; ++t) {
    for (int n = 0; n < NUM_NEURONS; ++n) {
      input[t * NUM_NEURONS + n] = dist(gen) ? 1 : 0;
    }
  }

  //Initialise threshold in input
  DATATYPE *buf_threshold = bo_threshold.map<uint32_t*>();
  buf_threshold = threshold;

  //TODO make one buffer only for the parameters (threshold, leak, ...)

  // Zero out buffer for the output spikes
  DATATYPE *buf_out_spikes = bo_out_spikes.map<DATATYPE *>();
  memset(bufOut, 0, OUT_SIZE * sizeof(DATATYPE));

  // SYNC HOST TO DEVICE MEMORIES //

  // sync host to device memories
  bo_instr.sync(XCL_BO_SYNC_BO_TO_DEVICE);
  bo_in_spikes.sync(XCL_BO_SYNC_BO_TO_DEVICE);
  bo_threshold.sync(XCL_BO_SYNC_BO_TO_DEVICE);
  bo_out_spikes.sync(XCL_BO_SYNC_BO_TO_DEVICE);

  // Execute the kernel and wait to finish
  if (verbosity >= 1)
    std::cout << "Running Kernel.\n";
  unsigned int opcode = 3;
  auto run =
      kernel(opcode, bo_instr, instr_v.size(), bo_in_spikes, bo_threshold, bo_out_spikes);
  run.wait();

  // COMPARING RESULT //

  // Build a vector with the output spike (ref) to compare with the bufOut[i]
  int errors = 0;
  if (verbosity >= 1) {
    std::cout << "Verifying results ..." << std::endl;
  }
  for (uint32_t i = 0; i < IN_SIZE; i++) {
    int32_t ref = bufInA[i] * scaleFactor;
    int32_t test = bufOut[i];
    if (test != ref) {
      if (verbosity >= 1)
        std::cout << "Error in output " << test << " != " << ref << std::endl;
      errors++;
    } else {
      if (verbosity >= 1)
        std::cout << "Correct output " << test << " == " << ref << std::endl;
    }
  }

  // Print Pass/Fail result of our test
  if (!errors) {
    std::cout << std::endl << "PASS!" << std::endl << std::endl;
    return 0;
  } else {
    std::cout << std::endl
              << errors << " mismatches." << std::endl
              << std::endl;
    std::cout << std::endl << "fail." << std::endl << std::endl;
    return 1;
  }
}