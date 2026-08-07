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

build cl_systolic_array "make host"

if command -v nvcc &> /dev/null; then
    build cl_systolic_array "make nvidia"
fi
