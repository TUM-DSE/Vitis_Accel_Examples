#!/bin/bash

aoc -v -board-package=/tools/Intel/intelFPGA_pro/20.2/hld/board/s10_ref -board=s10gx src/matmul_intel.cl -o partition_cyclicblock.aocx
