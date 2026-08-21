#
# Copyright 2019-2021 Xilinx, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# makefile-generator v1.0.3
#
# Points to top directory of Git repository
MK_PATH := $(abspath $(lastword $(MAKEFILE_LIST)))
COMMON_REPO ?= $(shell bash -c 'export MK_PATH=$(MK_PATH); echo $${MK_PATH%ocl_kernels/cl_array_partition_npu/*}')
PWD = $(shell readlink -f .)
XF_PROJ_ROOT = $(shell readlink -f $(COMMON_REPO))


########################## Checking if PLATFORM in allowlist #######################
PLATFORM_BLOCKLIST += nodma 
PLATFORM ?= xilinx_u250_gen3x16_xdma_4_1_202210_1
ifneq ($(shell which platforminfo 2>/dev/null),)
DEV_ARCH := $(shell platforminfo -p $(PLATFORM) | grep 'FPGA Family' | sed 's/.*://' | sed '/ai_engine/d' | sed 's/^[[:space:]]*//')
CPU_TYPE := $(shell platforminfo -p $(PLATFORM) | grep 'CPU Type' | sed 's/.*://' | sed '/ai_engine/d' | sed 's/^[[:space:]]*//')
endif

ifeq ($(CPU_TYPE), cortex-a9)
HOST_ARCH := aarch32
else ifneq (,$(findstring cortex-a, $(CPU_TYPE)))
HOST_ARCH := aarch64
else
HOST_ARCH := x86
endif

ifeq ($(DEV_ARCH), zynquplus)
ifeq ($(HOST_ARCH), aarch64)
include makefile_zynqmp.mk
else
include makefile_us_alveo.mk
endif
else ifeq ($(DEV_ARCH), versal)
ifeq ($(HOST_ARCH), x86)
include makefile_versal_alveo.mk
else
include makefile_versal_ps.mk
endif
else
include makefile_us_alveo.mk
endif

############################## NVIDIA GPU Target ##############################
NVIDIA_CXX      := g++
NVIDIA_CXXFLAGS := -std=c++14 -O2

# nvml.h and libnvidia-ml ship in /usr/include and /usr/lib with the driver
# packages on some systems, but on others only the CUDA toolkit has them -- e.g.
# containers where the runtime injects the driver libraries from a path the
# compiler and linker do not search. Fall back to the toolkit when it is there;
# override CUDA_PATH, NVIDIA_INC or NVIDIA_LIB for a non-standard install.
CUDA_PATH       ?= /usr/local/cuda

ifneq ($(wildcard $(CUDA_PATH)/include/nvml.h),)
NVIDIA_INC      ?= -I$(CUDA_PATH)/include
else
NVIDIA_INC      ?=
endif

# Linking against the stub is enough: it carries the real SONAME, so at run time
# ld.so still loads the driver's own libnvidia-ml.so.1.
ifneq ($(wildcard $(CUDA_PATH)/lib64/stubs/libnvidia-ml.so),)
NVIDIA_LIB      ?= -lOpenCL -L$(CUDA_PATH)/lib64/stubs -lnvidia-ml
else
NVIDIA_LIB      ?= -lOpenCL -lnvidia-ml
endif

.PHONY: nvidia run_nvidia clean_nvidia

nvidia: src/host_nvidia.cpp
	$(NVIDIA_CXX) $(NVIDIA_CXXFLAGS) $(NVIDIA_INC) src/host_nvidia.cpp $(NVIDIA_LIB) -o cl_array_partition_npu_nvidia

run_nvidia: nvidia
	./cl_array_partition_npu_nvidia src/matmul_nvidia.cl

clean_nvidia:
	rm -f cl_array_partition_npu_nvidia

############################## Help Section ##############################
help:
	$(ECHO) "Makefile Usage:"
	$(ECHO) "  make all TARGET=<sw_emu/hw_emu/hw> PLATFORM=<FPGA platform> EDGE_COMMON_SW=<rootfs and kernel image path>"
	$(ECHO) "      Command to generate the design for specified Target and Shell."
	$(ECHO) ""
	$(ECHO) "  make run TARGET=<sw_emu/hw_emu/hw> PLATFORM=<FPGA platform> EMU_PS=<X86/QEMU> EDGE_COMMON_SW=<rootfs and kernel image path>"
	$(ECHO) "      Command to run application in emulation.Default sw_emu will run on x86 ,to launch on qemu specify EMU_PS=QEMU."
	$(ECHO) ""
	$(ECHO) "  make build TARGET=<sw_emu/hw_emu/hw> PLATFORM=<FPGA platform> EDGE_COMMON_SW=<rootfs and kernel image path>"
	$(ECHO) "      Command to build xclbin application."
	$(ECHO) ""
	$(ECHO) "  make host PLATFORM=<FPGA platform> EDGE_COMMON_SW=<rootfs and kernel image path>"
	$(ECHO) "      Command to build host application."
	$(ECHO) "      EDGE_COMMON_SW is required for SoC shells. Please download and use the pre-built image from - "
	$(ECHO) "      https://www.xilinx.com/support/download/index.html/content/xilinx/en/downloadNav/embedded-platforms.html"
	$(ECHO) ""
	$(ECHO) "  make sd_card TARGET=<sw_emu/hw_emu/hw> PLATFORM=<FPGA platform> EDGE_COMMON_SW=<rootfs and kernel image path>"
	$(ECHO) "      Command to prepare sd_card files."
	$(ECHO) ""
	$(ECHO) "  make clean "
	$(ECHO) "      Command to remove the generated non-hardware files."
	$(ECHO) ""
	$(ECHO) "  make cleanall"
	$(ECHO) "      Command to remove all the generated files."
	$(ECHO) ""
