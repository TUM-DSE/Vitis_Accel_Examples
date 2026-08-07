#!/usr/bin/env bash

set -euo pipefail
# set -x

usage() {
    echo "Usage: $(basename "$0") -j <max_jobs> -s <slow_frequency> -f <fast_frequency>"
}

if [ -z "$XILINX_VITIS" ]; then
    echo "XILINX_VITIS is unset, did you source settings64.sh?"
    exit 1
fi

max_jobs=8
slow_freq=200
fast_freq=650
u50_platform=xilinx_u50_gen3x16_xdma_5_202210_1
u280_platform=xilinx_u280_gen3x16_xdma_1_202211_1
fpgas=(u50 u280)
variants=(hbm-slow hbm-fast ddr-slow ddr-fast)
apps=(cl_array_partition cl_burst_rw cl_dataflow_func cl_dataflow_subfunc cl_helloworld \
      cl_lmem_2rw cl_loop_reorder cl_partition_cyclicblock cl_shift_register cl_systolic_array cl_wide_mem_rw)

while getopts hj:s:f: opt; do
    case $opt in
        "h")
            usage
            exit 0 ;;
        "j")
            max_jobs=$OPTARG ;;
        "s")
            slow_freq=$OPTARG ;;
        "f")
            fast_freq=$OPTARG ;;
        *)
            exit 1 ;;
    esac
done

for app in "${apps[@]}"; do
    for fpga in "${fpgas[@]}"; do
        for variant in "${variants[@]}"; do
            build_dir=$app-$fpga-$variant

            cp -r "$app" "$build_dir"
            pushd "$build_dir"
            # Fix hard coded directory name in Makefile
            sed -i "s/$app/$build_dir/" ./Makefile
            # make --always-make all TARGET=hw PLATFORM=$U50_PLATFORM FREQ=0:$ || echo "$build_dir" failed &

            popd
        done
    done
done
