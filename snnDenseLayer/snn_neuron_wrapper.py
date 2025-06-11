import numpy as np
import subprocess
from mlir_wrapper import compile_snn_neuron
import torch
import re

class base_neuron:
    def __init__(self, in1_size=4096, out_size=4096, threshold=5, decay_factor=0.9, reset=-1, hard_reset=1, trace_size=8192, use_placed=False):
        self.in1_size = in1_size
        self.out_size = out_size
        self.threshold = threshold
        self.decay_factor = decay_factor
        self.reset = reset
        self.hard_reset = hard_reset
        self.trace_size = trace_size
        self.use_placed = use_placed
        self.compiled = False

    def to(self, device = "npu"):
        self.device = device
        if(device == "npu"):
            compile_snn_neuron(in1_size=self.in1_size, out_size=self.out_size, threshold=self.threshold, decay_factor= self.decay_factor, reset=self.reset, hard_reset=self.hard_reset, trace_size=self.trace_size, device=self.device, aie_design_name = "snn_neuron", use_placed=self.use_placed)
            self.compiled = True
            print("Model Running on NPU...") 

    def __call__(self, input_data):
        if self.compiled == False:
            self.to_npu()

        # Writing the input to a txt file for test.cpp
        with open("input_spikes.txt", "w") as f:
            for row in input_data:
                f.write(' '.join(map(str, input_data.tolist())))
        
        print("Running the testbench...")
        process_result = subprocess.run([
        "make", "run",
        f"in1_size={self.in1_size}",
        f"out_size={self.out_size}",
        f"threshold={self.threshold}",
        f"decay_factor={self.decay_factor}",
        f"reset={self.reset}",
        f"hard_reset={self.hard_reset}",
        f"trace_size={self.trace_size}",
        f"targetname={self.aie_design_name}"],
        capture_output=True,
        text=True,
        check=False
        )

        # Save the metrics inside a dictionary and return it.
        if(self.vectorized):
            unit = "vectorized"
        else:
            unit = "scalar"
        metrics = {
            "design":f"{self.aie_design_name}_{unit}" ,
            "npu_time": None,
            "throughput": None, # Corrected spelling from 'troughtput'
            "test_status": "UNKNOWN" # Added for 'PASS'/'fail'
        }

        stdout_output = process_result.stdout

        # Compile regex patterns for NPU response time, throughput
        npu_time_pattern = re.compile(r"NPU response time:\s*(\d+\.?\d*)\s*(us)\.?")
        throughput_pattern = re.compile(r"Throughput:\s*(\d+\.?\d*e?\+?\d*)\s*(operations/second)\.")
        passed_pattern = re.compile(r"PASS")
        failed_pattern = re.compile(r"fail")

        # Search for and extract NPU response time
        npu_time_match = npu_time_pattern.search(stdout_output)
        if npu_time_match:
            try:
                metrics["npu_time"] = float(npu_time_match.group(1))
                print(f"Extracted NPU response time: {metrics['npu_time']} us")
            except ValueError:
                print("Could not parse NPU response time.")

        # Search for and extract throughput
        throughput_match = throughput_pattern.search(stdout_output)
        if throughput_match:
            try:
                metrics["throughput"] = float(throughput_match.group(1))
                print(f"Extracted Throughput: {metrics['throughput']} operations/second")
            except ValueError:
                print("Could not parse Throughput.")

        # Determine test status
        if passed_pattern.search(stdout_output):
            metrics["test_status"] = "PASS"
            print("Test status: PASS")
        elif failed_pattern.search(stdout_output):
            metrics["test_status"] = "FAIL"
            print("Test status: FAIL")
        else:
            print("Test status: Not explicitly found (PASS/fail).")

        # Print errors
        print(process_result.stderr)

        
        with open("output_spikes.txt", "r") as f:
            output_spikes = [int(line.strip()) for line in f if line.strip()]

        output_spikes_torch = torch.tensor(output_spikes, dtype=torch.int32)

        return output_spikes_torch, metrics


class snn_neuron_npu_multicore(base_neuron):
    # Overload
    def __init__(self, in1_size=4096, out_size=4096, threshold=5, decay_factor=0.9, reset=-1, hard_reset=1, trace_size=8192, use_placed=False, aie_design_name = "multicore", num_cores = 2, vectorized=True):
        super().__init__(in1_size=in1_size, out_size=out_size, threshold=threshold, decay_factor=decay_factor, reset=reset, hard_reset=hard_reset, trace_size=trace_size, use_placed=use_placed)
        self.aie_design_name = "multicore"
        self.num_cores = 2
        self.vectorized = vectorized

    # Overriding
    # TODO move to only one generic class
    def to(self, device = "npu"):
        self.device = device
        if(device == "npu"):
            compile_snn_neuron(in1_size=self.in1_size, out_size=self.out_size, threshold=self.threshold, decay_factor= self.decay_factor, reset=self.reset, hard_reset=self.hard_reset, trace_size=self.trace_size, device=self.device, aie_design_name = self.aie_design_name, use_placed=self.use_placed)
            self.compiled = True
        print("Model Running on NPU...")

class snn_neuron_npu_singlecore(base_neuron):
    # Overload
    def __init__(self, in1_size=4096, out_size=4096, threshold=5, decay_factor=0.9, reset=-1, hard_reset=1, trace_size=8192, use_placed=False, aie_design_name = "singlecore", vectorized = True):
        super().__init__(in1_size=in1_size, out_size=out_size, threshold=threshold, decay_factor=decay_factor, reset=reset, hard_reset=hard_reset, trace_size=trace_size, use_placed=use_placed)
        self.aie_design_name = "singlecore"
        self.vectorized = vectorized

    # Overriding
    # TODO move to only one generic class
    def to(self, device = "npu"):
        self.device = device
        if(device == "npu"):
            compile_snn_neuron(in1_size=self.in1_size, out_size=self.out_size, threshold=self.threshold, decay_factor= self.decay_factor, reset=self.reset, hard_reset=self.hard_reset, trace_size=self.trace_size, device=self.device, aie_design_name = self.aie_design_name, use_placed=self.use_placed)
            self.compiled = True
        print("Model Running on NPU...")

class snn_neuron_npu_denselayer(base_neuron):
    # Overload
    def __init__(self, in1_size=4096, out_size=4096, threshold=5, decay_factor=0.9, reset=-1, hard_reset=1, trace_size=8192, use_placed=False, aie_design_name = "denselayer", vectorized = False):
        super().__init__(in1_size=in1_size, out_size=out_size, threshold=threshold, decay_factor=decay_factor, reset=reset, hard_reset=hard_reset, trace_size=trace_size, use_placed=use_placed)
        self.aie_design_name = "denselayer"
        self.vectorized = vectorized

    # Overriding
    # TODO move to only one generic class
    def to(self, device = "npu"):
        self.device = device
        if(device == "npu"):
            compile_snn_neuron(in1_size=self.in1_size, out_size=self.out_size, threshold=self.threshold, decay_factor= self.decay_factor, reset=self.reset, hard_reset=self.hard_reset, trace_size=self.trace_size, device=self.device, aie_design_name = self.aie_design_name, use_placed=self.use_placed)
            self.compiled = True
        print("Model Running on NPU...")
    



        
