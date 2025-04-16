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
#include <random> // Added for mt19937 and bernoulli_distribution
#include <cassert> // Added for test assertions

#include "xrt/xrt_bo.h"
#include "xrt/xrt_device.h"
#include "xrt/xrt_kernel.h"

#include "test_utils.h"


namespace po = boost::program_options;
const int threshold = 10;

int main(int argc, const char *argv[]) {

  // Program arguments parsing
  po::options_description desc("Allowed options");
  po::variables_map vm;
  test_utils::add_default_options(desc);

  test_utils::parse_options(argc, argv, desc, vm);
  int verbosity = vm["verbosity"].as<int>();
  verbosity = 1;
  // Declaring design constants
  constexpr bool VERIFY = true;
  constexpr int IN_SIZE = 1024;
  constexpr int OUT_SIZE = IN_SIZE;

  // Load instruction sequence
  std::vector<uint32_t> instr_v =
      test_utils::load_instr_binary(vm["instr"].as<std::string>());

  if (verbosity >= 1)
    std::cout << "Sequence instr count: " << instr_v.size() << "\n";

  // ------------------------------------------------------
  // Get device, load the xclbin & kernel and register them
  // ------------------------------------------------------
  // Get a device handle
  unsigned int device_index = 0;
  auto device = xrt::device(device_index);

  // Load the xclbin
  if (verbosity >= 1)
    std::cout << "Loading xclbin: " << vm["xclbin"].as<std::string>() << "\n";
  auto xclbin = xrt::xclbin(vm["xclbin"].as<std::string>());

  // Load the kernel
  if (verbosity >= 1)
    std::cout << "Kernel opcode: " << vm["kernel"].as<std::string>() << "\n";
  std::string Node = vm["kernel"].as<std::string>();

  auto xkernels = xclbin.get_kernels();
  auto xkernel = *std::find_if(xkernels.begin(), xkernels.end(),
                               [Node, verbosity](xrt::xclbin::kernel &k) {
                                 auto name = k.get_name();
                                 if (verbosity >= 1) {
                                   std::cout << "Name: " << name << std::endl;
                                 }
                                 return name.rfind(Node, 0) == 0;
                               });
  auto kernelName = xkernel.get_name();

     // Register xclbin
  if (verbosity >= 1)
    std::cout << "Registering xclbin: " << vm["xclbin"].as<std::string>()
              << "\n";
  device.register_xclbin(xclbin);

  // Get a hardware context
  if (verbosity >= 1)
    std::cout << "Getting hardware context.\n";
  xrt::hw_context context(device, xclbin.get_uuid());

  // Get a kernel handle
  if (verbosity >= 1)
    std::cout << "Getting handle to kernel:" << kernelName << "\n";
  auto kernel = xrt::kernel(context, kernelName);

  // ------ FILL BUFFER --------------

  // Setting up the object buffers
  auto bo_instr = xrt::bo(device, instr_v.size() * sizeof(int),
                          XCL_BO_FLAGS_CACHEABLE, kernel.group_id(1));
  auto bo_in_spikes = xrt::bo(device, IN_SIZE * sizeof(int32_t),
                              XRT_BO_FLAGS_HOST_ONLY, kernel.group_id(3));
  auto bo_out_spikes = xrt::bo(device, OUT_SIZE * sizeof(int32_t),
                               XRT_BO_FLAGS_HOST_ONLY, kernel.group_id(4));

  if (verbosity >= 1)
    std::cout << "Writing data into buffer objects.\n";

  // Copy instruction stream to xrt buffer object
  void *bufInstr = bo_instr.map<void *>();
  memcpy(bufInstr, instr_v.data(), instr_v.size() * sizeof(int));

  // Initialize buffer bo_in_spikes
    int32_t *buf_in_spikes = bo_in_spikes.map<int32_t *>();


    std::mt19937 gen(42); // Fixed seed for reproducibility
    std::bernoulli_distribution dist(0.1);


    std::vector<int32_t> srcVecSpikes;
    srcVecSpikes.reserve(IN_SIZE); // Pre-allocate for efficiency

    for (int i = 0; i < IN_SIZE; ++i) {
        srcVecSpikes.push_back(dist(gen) ? 1 : 0);
    }

    // Copy to the buffer
    memcpy(buf_in_spikes, srcVecSpikes.data(), IN_SIZE * sizeof(int32_t));

  uint32_t *buf_out_spikes = bo_out_spikes.map<uint32_t *>();

  // SYNC HOST TO DEVICE MEMORIES //

  // sync host to device memories
  bo_instr.sync(XCL_BO_SYNC_BO_TO_DEVICE);
  bo_in_spikes.sync(XCL_BO_SYNC_BO_TO_DEVICE);
  bo_out_spikes.sync(XCL_BO_SYNC_BO_TO_DEVICE);

  // Execute the kernel and wait to finish
  if (verbosity >= 1)
    std::cout << "Running Kernel.\n";
  unsigned int opcode = 3;
  auto run =
      kernel(opcode, bo_instr, instr_v.size(), bo_in_spikes, bo_out_spikes);
  run.wait();

  bo_out_spikes.sync(XCL_BO_SYNC_BO_FROM_DEVICE);
  

  
  // COMPARING RESULT //

  // Build a vector with the output spike (ref) to compare with the buf_out_spikes[i]
  int errors = 0;
  int32_t ref = 0;
  int32_t test = 0;
  int32_t out = 0;
  if (verbosity >= 1) {
    std::cout << "Verifying results ..." << std::endl;
  }
  for (uint32_t i = 0; i < IN_SIZE; i++) {
    ref += buf_in_spikes[i];
    test = buf_out_spikes[i];
    //std::cout << "value output:" << test << "i: " << i << std::endl;
    if (ref >= threshold) {
      out = 1;
      ref = 0;
      //printf("fire at %d\n", i);
    } else {
      out = 0;
    }

    
    if (out != test) {
      if (verbosity = 1)
        std::cout << "Error in output " << test << " != " << out << std::endl;
      errors++;
    } else {
      if (verbosity = 1)
        std::cout << "Correct output " << test << " == " << out << std::endl;
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
