/**
* Copyright (C) 2019-2021 Xilinx, Inc
*
* Licensed under the Apache License, Version 2.0 (the "License"). You may
* not use this file except in compliance with the License. A copy of the
* License is located at
*
*     http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
* WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
* License for the specific language governing permissions and limitations
* under the License.
*/

// Maximum Matrix Dimension Supported by Kernel
#define MAX_DIM 64 // int8 inputs: 2 * 64 * 64 * 1 B = 8 KB, int32 output: 16 KB

// Elements moved per global-memory beat.
//
// Vitis infers the AXI master's data width from the pointer type, so a plain
// char* would give the input ports a width of 8 bits: one byte per beat, 4096
// beats to load a matrix. Declaring the arguments as vectors makes all three
// ports 128 bits wide instead. char16 is the widest char vector OpenCL defines,
// and int4 matches it at 32 bits per element.
//
// Both loads drop from 4096 beats to 256 and the store from 4096 to 1024. dim
// must be a multiple of VEC_IN (the host checks this).
#define VEC_IN 16 // char16: 16 int8 operands per 128-bit beat
#define VEC_OUT 4 // int4:    4 int32 results per 128-bit beat

// Tripcount identifiers
__constant int c_size = MAX_DIM;

// Computes matrix multiply
// C = AxB, where A, B and C are square matrices of dimension (dimxdim). A and B
// are int8, C is an int32 accumulator, matching the RKNPU's
// INT8_MM_INT8_TO_INT32 matmul mode, so the same GEMM can be measured on FPGA,
// GPU and NPU. Only the partitioned kernel of the original example is kept.
kernel __attribute__((reqd_work_group_size(1, 1, 1))) void matmul_partition(
    const __global char16* in1, // Read-Only Matrix 1 (int8, VEC_IN per beat)
    const __global char16* in2, // Read-Only Matrix 2 (int8, VEC_IN per beat)
    __global int4* out_r,       // Output Result (int32, VEC_OUT per beat)
    int dim) {                  // Matrix Dimension. Assuming Square Matrix

    // Cyclic Partition for A as matrix multiplication needs row-wise parallel
    // access. The wide loads write VEC_IN consecutive elements of a row per
    // cycle, which cyclic partitioning already spreads over VEC_IN banks.
    char A[MAX_DIM * MAX_DIM] __attribute__((xcl_array_partition(cyclic, MAX_DIM, 1)));
    // Block Partition for B as matrix multiplication needs column-wise parallel
    // access: one bank per row of B, which is what block partitioning the
    // flattened MAX_DIM * MAX_DIM array by MAX_DIM gives. Each such row bank
    // would have to take all VEC_IN elements of a load beat at once, though, so
    // B is kept two-dimensional here, block partitioned into its rows as before
    // and each row additionally split cyclically into VEC_IN banks.
    char B[MAX_DIM][MAX_DIM] __attribute__((xcl_array_partition(block, MAX_DIM, 1)))
        __attribute__((xcl_array_partition(cyclic, VEC_IN, 2)));
    // Cyclic Partition for C so the wide store can read VEC_OUT consecutive
    // results per cycle.
    int C[MAX_DIM * MAX_DIM] __attribute__((xcl_array_partition(cyclic, VEC_OUT, 1)));

    // As A and B Matrix are partitioned with the factor of MAX_DIM, so to get
    // parallel row/column access, input square matrix[dimXdim] should be written
    // into local Array in MATRIX[MAX_DIM * MAX_DIM] format

    // Burst read for matrix A, VEC_IN elements per beat
    __attribute__((xcl_pipeline_loop(1))) __attribute__((xcl_loop_tripcount(c_size * c_size / VEC_IN,
                                                                           c_size * c_size / VEC_IN))) readA
        : for (int itr = 0, i = 0, j = 0; itr < (dim * dim) / VEC_IN; itr++, j += VEC_IN) {
        if (j == dim) {
            j = 0;
            i++;
        }
        char lane[VEC_IN] __attribute__((xcl_array_partition(complete, 1)));
        vstore16(in1[itr], 0, lane);
        __attribute__((opencl_unroll_hint(VEC_IN))) for (int e = 0; e < VEC_IN; e++) A[i * MAX_DIM + j + e] = lane[e];
    }

    // Burst read for matrix B, VEC_IN elements per beat
    __attribute__((xcl_pipeline_loop(1))) __attribute__((xcl_loop_tripcount(c_size * c_size / VEC_IN,
                                                                           c_size * c_size / VEC_IN))) readB
        : for (int itr = 0, i = 0, j = 0; itr < (dim * dim) / VEC_IN; itr++, j += VEC_IN) {
        if (j == dim) {
            j = 0;
            i++;
        }
        char lane[VEC_IN] __attribute__((xcl_array_partition(complete, 1)));
        vstore16(in2[itr], 0, lane);
        __attribute__((opencl_unroll_hint(VEC_IN))) for (int e = 0; e < VEC_IN; e++) B[i][j + e] = lane[e];
    }

    __attribute__((xcl_loop_tripcount(c_size, c_size))) lreorder1 : for (int i = 0; i < dim; i++) {
        // As A and B are partition correctly so loop pipelining is applied
        // at 2nd level loop and which will eventually unroll the lower loop
        __attribute__((xcl_pipeline_loop(1))) __attribute__((xcl_loop_tripcount(c_size, c_size))) lreorder2
            : for (int j = 0; j < dim; j++) {
            int result = 0;
            __attribute__((xcl_loop_tripcount(c_size, c_size))) lreorder3 : for (int k = 0; k < MAX_DIM; k++) {
                result += (int)A[i * MAX_DIM + k] * (int)B[k][j];
            }
            C[i * MAX_DIM + j] = result;
        }
    }

    // Burst write from output matrices to global memory, VEC_OUT elements per beat
    // Burst write from matrix C
    __attribute__((xcl_pipeline_loop(1))) __attribute__((xcl_loop_tripcount(c_size * c_size / VEC_OUT,
                                                                           c_size * c_size / VEC_OUT))) writeC
        : for (int itr = 0, i = 0, j = 0; itr < (dim * dim) / VEC_OUT; itr++, j += VEC_OUT) {
        if (j == dim) {
            j = 0;
            i++;
        }
        int lane[VEC_OUT] __attribute__((xcl_array_partition(complete, 1)));
        __attribute__((opencl_unroll_hint(VEC_OUT))) for (int e = 0; e < VEC_OUT; e++) lane[e] = C[i * MAX_DIM + j + e];
        out_r[itr] = vload4(0, lane);
    }
}
