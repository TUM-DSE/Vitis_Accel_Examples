#!/bin/bash

aoc -v -board-package=/share/intel-fpga/tools/20.2/hld/board/s10_ref -board=s10gx src/vadd_intel.cl -o wide_mem_rw.aocx
