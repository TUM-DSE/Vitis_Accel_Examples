#!/usr/bin/env bash
set -euo pipefail

script_dir=$(dirname "$(readlink -f "$0")")

build() {
	dir=$1
	build_cmd=$2

	if [ ! -d "$dir" ]; then
		echo "Skipping $dir (no such directory)"
		return
	fi

	echo "Building $dir ($build_cmd)"
	(cd "$dir" && $build_cmd > /dev/null || echo "building $dir failed") &

	wait
}

apps="cl_dataflow_func cl_shift_register cl_systolic_array cl_burst_rw cl_gmem_2banks cl_wide_mem_rw"

for app in $apps; do
	build "$app" "make host"
done

if command -v nvcc &> /dev/null; then
	for app in $apps; do
		build "$app" "make nvidia"
	done
fi
