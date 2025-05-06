import numpy as np
import subprocess
from mlir_wrapper import compile_snn_neuron

class snn_neuron_npu:
    def __init__(self, in1_size=4096, out_size=4096, threshold=10, decay_factor=0.9, reset=-1, hard_reset=1, trace_size=8192, use_placed=False):
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
            compile_snn_neuron(in1_size=self.in1_size, out_size=self.out_size, threshold=self.threshold, decay_factor= self.decay_factor, reset=self.reset, hard_reset=self.hard_reset, trace_size=self.trace_size, device=self.device, use_placed=self.use_placed)
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
        subprocess.run(["make", "run"])
        

        
