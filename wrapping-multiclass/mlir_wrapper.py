import subprocess
import os

# Wrapper for the compilation of the snn_neuron.py

def compile_snn_neuron(
    in1_size=4096,
    out_size=4096,
    threshold=10,
    decay_factor=0.9,
    reset=-1,
    hard_reset=1,
    trace_size=8192,
    device='npu',
    aie_design_name = "snn_neuron",
    use_placed=False
):

    # Clean all   
    subprocess.run(["make", "clean"], check=True)
    
    # Execute make to produce the MLIR file and compile the kernel code
    make_cmd = [
        "make", "all",
        f"in1_size={in1_size}",
        f"out_size={out_size}",
        f"threshold={threshold}",
        f"decay_factor={decay_factor}",
        f"reset={reset}",
        f"hard_reset={hard_reset}",
        f"trace_size={trace_size}",
        f"targetname={aie_design_name}",
    ]

    #print("Running:", " ".join(make_cmd))
    subprocess.run(make_cmd, check=True)

    print("Compilation completed. Artifacts generated: build/final.xclbin, build/insts.bin")
