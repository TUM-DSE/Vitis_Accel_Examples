#!/bin/bash

aoc -v -board-package=/share/intel-fpga/tools/20.2/hld/board/s10_ref -board=s10gx src/fir_intel.cl -o shift_register.aocx
