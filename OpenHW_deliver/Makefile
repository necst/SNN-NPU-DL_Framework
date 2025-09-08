##===- Makefile -----------------------------------------------------------===##
# 
# This file licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# Copyright (C) 2024, Advanced Micro Devices, Inc.
# 
##===----------------------------------------------------------------------===##

srcdir := $(shell dirname $(realpath $(firstword $(MAKEFILE_LIST))))

#include ${srcdir}/../../makefile-common
include ../../mlir-aie/programming_examples/makefile-common

##=== Values to modify ===========================================================##
## Input parameters (input size, output size, threshold, reset (-1 for hard reset)
# They will be overwritten at runtime during the wrapping

# Input size and output size in byte
in1_size = 4096
out_size = 4096
threshold = 2
decay_factor = 1
reset = -1
hard_reset = 1
trace_size = 8192
verbosity = 2
#0 == False 1 == True
vectorized = 1

## Name of the .py array and col of the npu to choose

targetname ?= singlecore
col = $(if $(filter 1,$(NPU2)),0,1)

devicename ?= $(if $(filter 1,$(NPU2)),npu2,npu)

## Choose type of test bench to run

aie_design_test_to_use = 0

ifeq ($(targetname), singlecore)
    aie_design_test_to_use = 0
endif
ifeq ($(targetname), multicore)
    aie_design_test_to_use = 1
endif
ifeq ($(targetname), denselayer)
    aie_design_test_to_use = 2
endif

aie_py_src=${targetname}.py
use_placed?=0

ifeq (${use_placed}, 1)
aie_py_src=${targetname}_placed.py
endif

all: $(info [MAKE] targetname=$(targetname), aie_design_test_to_use=$(aie_design_test_to_use)) build/final.xclbin build/insts.bin

VPATH := ${srcdir}/../../../aie_kernels/aie2

build/scale.o: ${srcdir}/lif_kernel_${targetname}.cc
	mkdir -p ${@D}
	cd ${@D} && ${PEANO_INSTALL_DIR}/bin/clang++ ${PEANOWRAP2_FLAGS} -c $< -o ${@F}

build/aie_trace.mlir: ${srcdir}/${aie_py_src}
	mkdir -p ${@D}
	python3 $< -d ${devicename} -i1s ${in1_size} -os ${out_size} -th ${threshold} -df ${decay_factor} -rs ${reset} -hr ${hard_reset} -t ${trace_size} > $@

build/aie.mlir: ${srcdir}/${aie_py_src}
	mkdir -p ${@D}
	python3 $< -d ${devicename} -i1s ${in1_size} -os ${out_size} -th ${threshold} -df ${decay_factor} -rs ${reset} -hr ${hard_reset} -vt ${vectorized} > $@

build/final.xclbin: build/aie.mlir build/scale.o
	mkdir -p ${@D}
	cd ${@D} && aiecc.py --aie-generate-xclbin --no-compile-host --xclbin-name=${@F} \
    	--no-xchesscc --no-xbridge --peano ${PEANO_INSTALL_DIR} \
				--aie-generate-npu-insts --npu-insts-name=insts.bin $(<:%=../%)

build/final_trace.xclbin: build/aie_trace.mlir build/scale.o
	mkdir -p ${@D}
	cd ${@D} && aiecc.py --aie-generate-xclbin --no-compile-host --xclbin-name=${@F} \
    	--no-xchesscc --no-xbridge \
				--aie-generate-npu-insts --npu-insts-name=insts.bin $(<:%=../%)

${targetname}.exe: ${srcdir}/test.cpp
	rm -rf _build
	mkdir -p _build
	cd _build && ${powershell} cmake ${srcdir} -DTARGET_NAME=${targetname} -DCMAKE_POLICY_VERSION_MINIMUM=3.5
	cd _build && ${powershell} cmake --build . --config Release
ifeq "${powershell}" "powershell.exe"
	cp _build/${targetname}.exe $@
else
	cp _build/${targetname} $@ 
endif 

run: ${targetname}.exe build/final.xclbin
	${powershell} ./$< -x build/final.xclbin -i build/insts.bin -k MLIR_AIE --in1_size ${in1_size} --out_size ${out_size} --threshold ${threshold} --decay_factor ${decay_factor} --hard_reset ${hard_reset} -v ${verbosity} --reset ${reset} --aie_design ${aie_design_test_to_use}

trace: ${targetname}.exe build/final_trace.xclbin build/insts.bin 
	${powershell} ./$< -x build/final_trace.xclbin -i build/insts.bin --in1_size ${in1_size} --out_size ${out_size} --threshold ${threshold} --decay_factor ${decay_factor} --hard_reset ${hard_reset} -v ${verbosity} --reset ${reset} --aie_design ${aie_design_test_to_use} -k MLIR_AIE -t ${trace_size}
	${srcdir}/../../../programming_examples/utils/parse_trace.py --input trace.txt --output trace_${targetname}.json --mlir build/aie_trace.mlir --colshift ${col}
	${srcdir}/../../../programming_examples/utils/get_trace_summary.py --filename trace_${targetname}.json

clean_trace:
	rm -rf tmpTrace trace.txt parse*json trace*json

clean: clean_trace
	rm -rf build _build inst aie.mlir.prj core_* test.elf ${targetname}.exe *.exe
