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

// Maximum Array Size
#define MAX_SIZE 128 // int8 inputs: 2 * 128 * 128 * 1 B = 32 KB, int32 output: 64 KB

// Elements moved per global-memory beat.
//
// Vitis infers the AXI master's data width from the pointer type, so a plain
// char* gave the input ports a width of 8 bits: one byte per beat, 16384 beats
// to load a matrix. Declaring the arguments as vectors makes all three ports
// 128 bits wide instead. char16 is the widest char vector OpenCL defines, and
// int4 matches it at 32 bits per element.
//
// Both loads drop from 16384 beats to 1024 and the store from 16384 to 4096.
// The size argument must be a multiple of VEC_IN (the host checks this).
#define VEC_IN 16 // char16: 16 int8 operands per 128-bit beat
#define VEC_OUT 4 // int4:    4 int32 results per 128-bit beat

// Tripcount identifiers
__constant int c_size = MAX_SIZE;

// Matrix multiplication kernel
// This kernel presents array partition concept
kernel __attribute__((reqd_work_group_size(1, 1, 1))) void matmul_partition(
    const __global char16* in1, // Read-Only Matrix 1 (int8, VEC_IN per beat)
    const __global char16* in2, // Read-Only Matrix 2 (int8, VEC_IN per beat)
    __global int4* out,         // Output Result (int32, VEC_OUT per beat)
    int size) {                 // Local memory to store input and output matrices
    // Local memory is implemented as BRAM memory blocks. A and B hold int8
    // operands; the accumulator and the result stay int32, which is what the
    // RKNPU's INT8_MM_INT8_TO_INT32 mode also produces.
    //
    // A is now filled VEC_IN elements at a time, so it needs VEC_IN independent
    // banks to absorb one beat per cycle. Cyclic partitioning on the 2nd
    // dimension gives exactly that, while leaving the compute loop's
    // element-at-a-time A[i][k] read on a single bank. B and C are partitioned
    // completely on their 2nd dimension already, which covers both the wide
    // burst access and the compute loop.
    char A[MAX_SIZE][MAX_SIZE] __attribute__((xcl_array_partition(cyclic, VEC_IN, 2)));

    // Partition Matrix B on 2nd dimension completely
    char B[MAX_SIZE][MAX_SIZE] __attribute__((xcl_array_partition(complete, 2)));

    // Partition Matrix C on 2nd dimension completely
    int C[MAX_SIZE][MAX_SIZE] __attribute__((xcl_array_partition(complete, 2)));

    // Partition Matrix temp_sum completely
    int temp_sum[MAX_SIZE] __attribute__((xcl_array_partition(complete, 1)));

    // Burst reads on input matrices from global memory
    // Burst read for matrix A: one beat covers VEC_IN consecutive columns
    __attribute__((xcl_pipeline_loop(1))) __attribute__((xcl_loop_tripcount(c_size* c_size / VEC_IN,
                                                                           c_size* c_size / VEC_IN))) readA
        : for (int itr = 0, i = 0, j = 0; itr < (size * size) / VEC_IN; itr++, j += VEC_IN) {
        if (j == size) {
            j = 0;
            i++;
        }
        char lane[VEC_IN] __attribute__((xcl_array_partition(complete, 1)));
        vstore16(in1[itr], 0, lane);
        __attribute__((opencl_unroll_hint(VEC_IN))) for (int e = 0; e < VEC_IN; e++) A[i][j + e] = lane[e];
    }

    // Burst read for matrix B: one beat covers VEC_IN consecutive columns
    __attribute__((xcl_pipeline_loop(1))) __attribute__((xcl_loop_tripcount(c_size* c_size / VEC_IN,
                                                                           c_size* c_size / VEC_IN))) readB
        : for (int itr = 0, i = 0, j = 0; itr < (size * size) / VEC_IN; itr++, j += VEC_IN) {
        if (j == size) {
            j = 0;
            i++;
        }
        char lane[VEC_IN] __attribute__((xcl_array_partition(complete, 1)));
        vstore16(in2[itr], 0, lane);
        __attribute__((opencl_unroll_hint(VEC_IN))) for (int e = 0; e < VEC_IN; e++) B[i][j + e] = lane[e];
    }

    // Performs matrix multiply over matrices A and B and stores the result
    // in C. All the matrices are square matrices of the form (size x size)
    // Calculate matrix multiplication using local data buffer based on input size
    // and write results into local buffer for C
    __attribute__((xcl_loop_tripcount(c_size, c_size))) arraypart1 : for (int i = 0; i < size; i++) {
        __attribute__((xcl_pipeline_loop(1))) __attribute__((xcl_loop_tripcount(c_size, c_size))) arraypart2
            : for (int k = 0; k < size; k++) {
            __attribute__((xcl_loop_tripcount(c_size, c_size))) arraypart3 : for (int j = 0; j < MAX_SIZE; j++) {
                int result = (k == 0) ? 0 : temp_sum[j];
                result += (int)A[i][k] * (int)B[k][j];
                temp_sum[j] = result;
                if (k == size - 1) C[i][j] = result;
            }
        }
    }

    // Burst write from output matrices to global memory
    // Burst write from matrix C: one beat covers VEC_OUT consecutive columns
    __attribute__((xcl_pipeline_loop(1))) __attribute__((xcl_loop_tripcount(c_size* c_size / VEC_OUT,
                                                                           c_size* c_size / VEC_OUT))) writeC
        : for (int itr = 0, i = 0, j = 0; itr < (size * size) / VEC_OUT; itr++, j += VEC_OUT) {
        if (j == size) {
            j = 0;
            i++;
        }
        int lane[VEC_OUT] __attribute__((xcl_array_partition(complete, 1)));
        __attribute__((opencl_unroll_hint(VEC_OUT))) for (int e = 0; e < VEC_OUT; e++) lane[e] = C[i][j + e];
        out[itr] = vload4(0, lane);
    }
}
