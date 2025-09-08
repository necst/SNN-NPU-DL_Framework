//===- test.cpp -------------------------------------------------*- C++ -*-===//
// 
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Copyright (C) 2023, Advanced Micro Devices, Inc.
//
//===----------------------------------------------------------------------===//
#include "cxxopts.hpp"
#include <bits/stdc++.h>
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

//namespace po = boost::program_options

//--------------------------------------------------------------------------
// XRT Wrapper struct and functions (only those used)
//--------------------------------------------------------------------------

/**
 * @brief Structure to hold command-line arguments.
 *
 * This struct stores all the necessary parameters read from the command line.
 */
struct args {
    int input_size;
    int output_size;
    int threshold;
    float decay_factor;
    int hard_reset;
    int verbosity;
    int reset;
    int aie_design_type;
    int trace_size;
    std::string instr;
    std::string xclbin;
    std::string kernel;
    std::string trace_file;
};

args parse_args(int argc, const char *argv[]) {
    cxxopts::Options desc("Allowed options");
    cxxopts::ParseResult vm;
    test_utils::add_default_options(desc);

    desc.add_options()
        ("in1_size", "Input size", cxxopts::value<int>())
        ("out_size", "Output size", cxxopts::value<int>())
        ("threshold", "Neuron firing threshold", cxxopts::value<float>())
        ("decay_factor", "Decay factor for neuron potential", cxxopts::value<float>())
        ("hard_reset", "Use hard reset (1) or soft reset (0)", cxxopts::value<int>()->default_value("0"))
        ("reset", "Reset behavior", cxxopts::value<float>()->default_value("0"))
        ("aie_design", "AIE design type", cxxopts::value<int>()->default_value("0"));

    args myargs;
    test_utils::parse_options(argc, argv, desc, vm);
    myargs.input_size = vm["in1_size"].as<int>();
    myargs.output_size = vm["out_size"].as<int>();
    myargs.verbosity = vm["verbosity"].as<int>();
    myargs.threshold = vm["threshold"].as<float>();
    myargs.decay_factor = vm["decay_factor"].as<float>();
    myargs.hard_reset = vm["hard_reset"].as<int>();
    myargs.reset = vm["reset"].as<float>();
    myargs.aie_design_type = vm["aie_design"].as<int>();
    myargs.trace_size = vm["trace_sz"].as<int>();
    myargs.instr = vm["instr"].as<std::string>();
    myargs.xclbin = vm["xclbin"].as<std::string>();
    myargs.kernel = vm["kernel"].as<std::string>();
    myargs.trace_file = vm["trace_file"].as<std::string>();
    return myargs;
}

void computeMetrics(auto start_kernel, auto stop_kernel, auto start_data, auto stop_data, int IN_SIZE)
{
    // ------------------------------------------------------
    // PRINTING RESULT AND TIME SPENT
    // ------------------------------------------------------

    int n_iterations = IN_SIZE;
    float npu_time_total = 0;
    float npu_time_min = 9999999;
    float npu_time_max = 0;

    float npu_time = std::chrono::duration_cast<std::chrono::microseconds>(stop_kernel - start_kernel).count();
    float npu_time_data = std::chrono::duration_cast<std::chrono::microseconds>(stop_data - start_data).count();
    npu_time_total += npu_time;
    npu_time_min = (npu_time < npu_time_min) ? npu_time : npu_time_min;
    npu_time_max = (npu_time > npu_time_max) ? npu_time : npu_time_max;

    // Response time
    std::cout << "NPU response time (only kernel): " << npu_time << "us" << std::endl;
    std::cout << "NPU response time: " << npu_time_data << "us" << std::endl;

    // Troughput (including the data movement)
    float throughput_ops_per_sec = (float)n_iterations / (npu_time_total / 1000000.0f);
    std::cout << std::endl << "Throughput: " << throughput_ops_per_sec << " operations/second." << std::endl;
    
    float macs = 0;
    if (macs > 0)
        std::cout << "Avg NPU gflops: " << macs / (1000 * npu_time_total / n_iterations) << std::endl;
    std::cout << std::endl << "Min NPU time: " << npu_time_min << " us." << std::endl;
    if (macs > 0)
        std::cout << "Max NPU gflops: " << macs / (1000 * npu_time_min) << std::endl;
    std::cout << std::endl << "Max NPU time: " << npu_time_max << " us." << std::endl;
    if (macs > 0)
        std::cout << "Min NPU gflops: " << macs / (1000 * npu_time_max) << std::endl;

    std::cout << std::endl << "FINISHED - Cleaning up." << std::endl; 
}


void generateInput(int32_t *buf_in_spikes, int32_t IN_SIZE, args myargs){
    
    int verbosity = myargs.verbosity;
    
    std::vector<int32_t> srcVecSpikes;
    srcVecSpikes.reserve(IN_SIZE); // Pre-allocate for efficiency

    // Read input from a txt file produced by the torch wrapper
    std::ifstream infile("input_spikes.txt");
    //if( !infile.is_open()) {
    //    std::cerr << "Failed to open the input file\n";
    //    return;
    //}

    int value;
    int count = 0;
    while (count < IN_SIZE) {
        srcVecSpikes.push_back(1);
        ++count;
    }
    
    // Copy to the buffer
    memcpy(buf_in_spikes, srcVecSpikes.data(), IN_SIZE * sizeof(int32_t));
}

void generateWeights(float *buf_in_weights, int32_t WEIGHT_SIZE, args myargs){
    
    int verbosity = myargs.verbosity;
    
    std::vector<float> srcWeights;
    srcWeights.reserve(WEIGHT_SIZE); // Pre-allocate for efficiency

    int count = 0;
    while (count < WEIGHT_SIZE) {
        srcWeights.push_back(1.0f);
        ++count;
    }
    
    // Copy to the buffer
    memcpy(buf_in_weights, srcWeights.data(), WEIGHT_SIZE * sizeof(float));
}

int singlecore_testbench(int32_t* buf_in_spikes, uint32_t* buf_out_spikes, args myargs)
{
    // Generate trace
    //if (myargs.trace_size > 0)
    //    test_utils::write_out_trace(bufTrace, myargs.trace_size, myargs.trace_file);

    int verbosity = myargs.verbosity;
    int IN_SIZE = myargs.input_size / sizeof(std::int32_t);
  
    //Perfomance evaluator variables
    float npu_time_total = 0;
    float npu_time_min = 9999999;
    float npu_time_max = 0;
    int ret_val = 0;

    // Build a vector with the output spike (ref) to compare with the buf_out_spikes[i]
    int errors = 0;
    float membrane_potential = 0;
    int32_t test = 0;
    int32_t out = 0;
    float decay_factor = myargs.decay_factor;

    int vectorized = 1;
    if(vectorized == 0)
    {
        std::cout << "Verifying results ... Single neuron testbench" << std::endl;
            
        for (uint32_t i = 0; i < IN_SIZE; i++) {
            membrane_potential = buf_in_spikes[i] + membrane_potential * decay_factor;
            test = buf_out_spikes[i];
            
            if (membrane_potential >= myargs.threshold) {
                out = 1;
                membrane_potential = 0;
            } else {
                out = 0;
            }
    
        
        if (out != test) {
            if (verbosity >= 1)
                std::cout << "Error in output " << test << " != " << out << std::endl;
            errors++;
        } else {
            if (verbosity >= 1)
                std::cout << "Correct output " << test << " == " << out << std::endl;
            }
        }
    
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
    else
    {
           auto vstart = std::chrono::system_clock::now();

    const uint32_t NUM_NEURONS = 16;
    const uint32_t TIME_STEPS = IN_SIZE / NUM_NEURONS;

    for (uint32_t neuron = 0; neuron < NUM_NEURONS; ++neuron) {
        float membrane_potential = 0;

    for (uint32_t t = 0; t < TIME_STEPS; ++t) {
        uint32_t index = neuron + t * NUM_NEURONS;

        int32_t input_spike = buf_in_spikes[index];
        int32_t expected_output;
        int32_t actual_output = buf_out_spikes[index];

        membrane_potential = membrane_potential * decay_factor;
        membrane_potential += input_spike;

        if (membrane_potential >= myargs.threshold) {
            expected_output = 1;
            membrane_potential = 0;
        } else {
            expected_output = 0;
        }

    if (expected_output != actual_output) {
        if (verbosity >= 1) {
        std::cout << "Mismatch at neuron " << neuron
                  << ", time step " << t
                  << ": expected " << expected_output
                  << ", got " << actual_output << std::endl;
        }
        ++errors;
    } else {
        if (verbosity >= 2) {
            std::cout << "Correct at neuron " << neuron
                  << ", time step " << t
                  << ": output " << actual_output << std::endl;
                }
            }
        }
    }

    auto vstop = std::chrono::system_clock::now();
    }


}


int multicore_testbench(int32_t* buf_in_spikes, uint32_t* buf_out_spikes, args myargs)
{
    int verbosity = myargs.verbosity;
    int IN_SIZE = myargs.input_size / sizeof(std::int32_t);
    int OUT_SIZE = IN_SIZE;
    //Perfomance evaluator variables
    float npu_time_total = 0;
    float npu_time_min = 9999999;
    float npu_time_max = 0;
    int ret_val = 0;

    // Build a vector with the output spike (ref) to compare with the buf_out_spikes[i]
    int errors = 0;
    int32_t ref = 0;
    int32_t test = 0;
    int32_t out = 0;
    
    std::cout << "Verifying results ..." << std::endl;
        
    auto vstart = std::chrono::system_clock::now();

    const uint32_t NUM_CORES = 2; // Number of compute tiles used in the architecture
    const uint32_t NUM_NEURONS = 32; // Number of neurons in the architecture
    const uint32_t NUM_NEURONS_PER_CORE = 16;
    const uint32_t MEM_SIZE = 512; // Adjsut as needed
    const uint32_t AIE_SIZE = MEM_SIZE / NUM_CORES; // Number of neurons in the architecture
    const uint32_t TIME_STEPS = AIE_SIZE / NUM_NEURONS_PER_CORE; // Number of time steps
    const uint32_t NUM_CALLED_CORE = IN_SIZE / MEM_SIZE;
    uint32_t shift_neuron = 0;
    
    for (uint32_t set_of_neurons = 0; set_of_neurons < NUM_NEURONS / NUM_NEURONS_PER_CORE; ++set_of_neurons)
        {
    
            for (uint32_t neuron = 0; neuron < NUM_NEURONS / 2; ++neuron)
            {
                float membrane_potential = 0;
    
                for (uint32_t offset_neuron = 0; offset_neuron < NUM_CALLED_CORE; ++offset_neuron)
                {
    
                    for (uint32_t t = 0; t < TIME_STEPS; ++t)
                    {
                        uint32_t index = neuron + t * NUM_NEURONS_PER_CORE + offset_neuron * MEM_SIZE + set_of_neurons * AIE_SIZE;
                        
                        int32_t input_spike = buf_in_spikes[index];
                        int32_t expected_output;
                        int32_t actual_output = buf_out_spikes[index];
    
                        std::cout << "Neuron" << neuron << " input:" << input_spike << " output:" << actual_output << "\n";
    
                        membrane_potential = membrane_potential * myargs.decay_factor;
                        membrane_potential += (float)input_spike;
    
                        if (membrane_potential >= myargs.threshold)
                        {
                            expected_output = 1;
                            membrane_potential = 0;
                        }
                        else
                        {
                            expected_output = 0;
                        }
    
                        if (expected_output != actual_output)
                        {
                            if (verbosity >= 1)
                            {
                                std::cout << " Mismatch at neuron " << neuron
                                          << ", time step " << t
                                          << ": expected " << expected_output
                                          << ", got " << actual_output << std::endl;
                            }
                            ++errors;
                        }
                        else
                        {
                            if (verbosity >= 2)
                            {
                                std::cout << " Correct at neuron " << neuron
                                          << ", time step " << t
                                          << ": output " << actual_output << std::endl;
                            }
                        }
                    }
                }
            }
        }
    
        auto vstop = std::chrono::system_clock::now();
    
        // ------------------------------------------------------
        // WRITE OUTPUT SPIKES TO FILE
        // ------------------------------------------------------
        std::ofstream outfile("output_spikes.txt");
        if (!outfile.is_open()) {
            std::cerr << "Failed to open output file for writing.\n";
            return 1;
        }
        
        for (int i = 0; i < OUT_SIZE; ++i) {
            outfile << buf_out_spikes[i] << "\n";
        }
        
        outfile.close();
        
        if (verbosity >= 1)
            std::cout << "Output spikes written to output_spikes.txt\n";
    
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

int denselayer_testbench(int32_t* buf_in_spikes, uint32_t* buf_out_spikes, args myargs)
{
    int verbosity = myargs.verbosity;
    int IN_SIZE = myargs.input_size / sizeof(std::int32_t);
    int OUT_SIZE = IN_SIZE;
    //Perfomance evaluator variables
    float npu_time_total = 0;
    float npu_time_min = 9999999;
    float npu_time_max = 0;
    int ret_val = 0;

    // Build a vector with the output spike (ref) to compare with the buf_out_spikes[i]
    int errors = 0;
    int32_t ref = 0;
    int32_t test = 0;
    int32_t out = 0;
    
    std::cout << "Verifying results ..." << std::endl;
        
    auto vstart = std::chrono::system_clock::now();

    const uint32_t NUM_CORES = 2; // Number of compute tiles used in the architecture
    const uint32_t NUM_NEURONS = 32; // Number of neurons in the architecture
    const uint32_t NUM_NEURONS_PER_CORE = 16;
    const uint32_t MEM_SIZE = 512; // Adjsut as needed
    const uint32_t AIE_SIZE = MEM_SIZE / NUM_CORES; // Number of neurons in the architecture
    const uint32_t TIME_STEPS = AIE_SIZE / NUM_NEURONS_PER_CORE; // Number of time steps
    const uint32_t NUM_CALLED_CORE = IN_SIZE / MEM_SIZE;
    uint32_t shift_neuron = 0;
    
    for (int32_t counter = 0; counter < IN_SIZE; counter ++)
        {
        std::cout << buf_out_spikes[counter] << std::endl;
        }
    
        auto vstop = std::chrono::system_clock::now();
    
        // ------------------------------------------------------
        // WRITE OUTPUT SPIKES TO FILE
        // ------------------------------------------------------
        std::ofstream outfile("output_spikes.txt");
        if (!outfile.is_open()) {
            std::cerr << "Failed to open output file for writing.\n";
            return 1;
        }
        
        for (int i = 0; i < OUT_SIZE*2; ++i) {
            outfile << buf_out_spikes[i] << "\n";
        }
        
        outfile.close();
        
        if (verbosity >= 1)
            std::cout << "Output spikes written to output_spikes.txt\n";

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

int main(int argc, const char *argv[]) {

    // ------------------------------------------------------
    // Parse program arguments
    // ------------------------------------------------------
    args myargs = parse_args(argc, argv);

    //REMOVE COMMENT TO USE TRACE
    int trace_size = myargs.trace_size;
    std::string trace_file = myargs.trace_file;

    // Declaring design constants
    bool VERIFY = true;
    int IN_SIZE = myargs.input_size / sizeof(std::int32_t);
    int OUT_SIZE = myargs.output_size / sizeof(std::int32_t);
    int WEIGHT_SIZE = (16*16) + (16*16);
    int verbosity = myargs.verbosity; 

    // Load instruction sequence
    std::vector<uint32_t> instr_v =
        test_utils::load_instr_binary(myargs.instr);

    if (verbosity >= 1)
        std::cout << "Sequence instr count: " << instr_v.size() << "\n";

    // ------------------------------------------------------
    // Get device, load the xclbin & kernel and register them
    // ------------------------------------------------------
    xrt::device device;
    xrt::kernel kernel;
    test_utils::init_xrt_load_kernel(device, kernel, myargs.verbosity, myargs.xclbin, myargs.kernel);

    // ------------------------------------------------------
    // ------ FILL BUFFER -----------------------------------
    // ------------------------------------------------------
    
    // Setting up the object buffers
    auto bo_instr = xrt::bo(device, instr_v.size() * sizeof(int),
                          XCL_BO_FLAGS_CACHEABLE, kernel.group_id(1));
    auto bo_in_spikes = xrt::bo(device, IN_SIZE * sizeof(int32_t),
                              XRT_BO_FLAGS_HOST_ONLY, kernel.group_id(3));
    auto bo_in_weights = xrt::bo(device, WEIGHT_SIZE * sizeof(float), XRT_BO_FLAGS_HOST_ONLY, kernel.group_id(4)); 
    auto bo_out_spikes = xrt::bo(device, OUT_SIZE * sizeof(int32_t),
                               XRT_BO_FLAGS_HOST_ONLY, kernel.group_id(5));

    //REMOVE COMMENT TO USE TRACE
    //auto bo_trace = xrt::bo(device, trace_size, XRT_BO_FLAGS_HOST_ONLY, kernel.group_id(7));

    if (verbosity >= 1)
        std::cout << "Writing data into buffer objects.\n";

    
    int32_t* buf_in_spikes = bo_in_spikes.map<int32_t *>();
    float* buf_in_weights = bo_in_weights.map<float *>();
    
    generateInput(buf_in_spikes, IN_SIZE, myargs);
    generateWeights(buf_in_weights, WEIGHT_SIZE, myargs);

    // Copy instruction stream to xrt buffer object
    void *bufInstr = bo_instr.map<void *>();
    memcpy(bufInstr, instr_v.data(), instr_v.size() * sizeof(int));

    uint32_t *buf_out_spikes = bo_out_spikes.map<uint32_t *>();

    //REMOVE COMMENT TO USE TRACE
    //char *bufTrace = bo_trace.map<char *>();

    // ------------------------------------------------------
    // SYNC HOST TO DEVICE MEMORIES -------------------------
    // ------------------------------------------------------

    //REMOVE COMMENT TO USE TRACE
    //if (trace_size > 0)
    //    memset(bufTrace, 0, trace_size);
    
    // sync host to device memories
    auto start_data_movement = std::chrono::high_resolution_clock::now();
    
    bo_instr.sync(XCL_BO_SYNC_BO_TO_DEVICE);
    bo_in_spikes.sync(XCL_BO_SYNC_BO_TO_DEVICE);
    bo_in_weights.sync(XCL_BO_SYNC_BO_TO_DEVICE);
    bo_out_spikes.sync(XCL_BO_SYNC_BO_TO_DEVICE);

    // Execute the kernel and wait to finish
    if (verbosity >= 1)
        std::cout << "Running Kernel.\n";

    auto start_kernel = std::chrono::high_resolution_clock::now();
    unsigned int opcode = 3;
    xrt::run run;

    if(myargs.aie_design_type > 1)
        run = kernel(opcode, bo_instr, instr_v.size(), bo_in_spikes, bo_in_weights, bo_out_spikes);
    else
        run = kernel(opcode, bo_instr, instr_v.size(), bo_in_spikes, bo_out_spikes);
    run.wait();

    std::cout<< "Execution finished" << std::endl;
    auto stop_kernel = std::chrono::high_resolution_clock::now();
    
    bo_out_spikes.sync(XCL_BO_SYNC_BO_FROM_DEVICE);

    auto stop_data_movement = std::chrono::high_resolution_clock::now();

    //REMOVE COMMENT TO USE TRACE
    //if (trace_size > 0)
    //    bo_trace.sync(XCL_BO_SYNC_BO_FROM_DEVICE);
  
    // ------------------------------------------------------
    // COMPARING RESULT //
    // ------------------------------------------------------

    //REMOVE COMMENT TO USE TRACE
    // Generate trace
    //if (trace_size > 0)
    //    test_utils::write_out_trace(bufTrace, trace_size, trace_file);
    
    switch (myargs.aie_design_type){
        case 0:
            singlecore_testbench(buf_in_spikes, buf_out_spikes, myargs);
        break;
        case 1:
            multicore_testbench(buf_in_spikes, buf_out_spikes, myargs);
        break;
        case 2:
            denselayer_testbench(buf_in_spikes, buf_out_spikes, myargs);
        break;
        default:
            std::cout << "Not correct value for testbench..." << std::endl;
    }

    computeMetrics(start_kernel, stop_kernel, start_data_movement, stop_data_movement, IN_SIZE);
    
    return 0;
}




