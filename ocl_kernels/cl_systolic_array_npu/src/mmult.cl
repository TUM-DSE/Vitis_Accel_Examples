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

/*******************************************************************************

Vitis Key Concept :

    This is a matrix multiplication example which showcases the "Systolic Array"
    based algorithm design. Systolic array type of implementation is well suited
    for FPGAs.

*******************************************************************************/

/*

Kernel Description :

    This kernel is a systolic array based matrix multiplication. Though the
    maximum size of the input matrices are restricted to a smaller MAX_SIZE, it
    is still possible to use this approach and get better performance for larger
    matrices by using tiling.

    This variant multiplies int8 operands into an int32 result, matching the
    RKNPU's INT8_MM_INT8_TO_INT32 matmul mode, so the same GEMM can be measured
    on FPGA, GPU and NPU.

    Arguments :

        char16 *a  (input )  --> Input  Matrix A (int8, VEC_IN per beat)
        char16 *b  (input )  --> Input  Matrix B (int8, VEC_IN per beat)
        int4   *c  (output)  --> Output Matrix   (int32, VEC_OUT per beat)
        int  a_row (input )  --> Row Size Matrix A
        int  a_col (input )  --> Col Size Matrix A
        int  b_col (input )  --> Col Size Matrix B

    Kernel Configuration :

        Max Size    --> 32

    Note :
        Max Size is dependent on the available DSP resources in the FPGA.
        The original int32 example stops at 24, which is not a shape the
        RK3588's int8 matmul accepts (K and N must be multiples of 32). An
        int8 x int8 product fits a single DSP where an int32 one needs
        several, so the 32 x 32 array needed here costs fewer DSPs than the
        original 24 x 24 one.
*/

// Maximum Array Size
#define MAX_SIZE 32 // int8 inputs: 2 * 32 * 32 * 1 B = 2 KB, int32 output: 4 KB

// Elements moved per global-memory beat.
//
// Vitis infers the AXI master's data width from the pointer type, so a plain
// char* would give the input ports a width of 8 bits: one byte per beat, 1024
// beats to load a matrix, against only a_col cycles of systolic compute.
// Declaring the arguments as vectors makes all three ports 128 bits wide
// instead. char16 is the widest char vector OpenCL defines, and int4 matches it
// at 32 bits per element.
//
// Both loads drop from 1024 beats to 64 and the store from 1024 to 256. a_col
// and b_col must be multiples of VEC_IN (the host checks this).
#define VEC_IN 16 // char16: 16 int8 operands per 128-bit beat
#define VEC_OUT 4 // int4:    4 int32 results per 128-bit beat

// Tripcount identifiers
__constant int c_size = MAX_SIZE;

__kernel __attribute__((reqd_work_group_size(1, 1, 1))) void mmult(
    const __global char16* a, // Read-Only Matrix A (int8, VEC_IN per beat)
    const __global char16* b, // Read-Only Matrix B (int8, VEC_IN per beat)
    __global int4* c,         // Output Result (int32, VEC_OUT per beat)
    int a_row,                // Matrix A Row Size
    int a_col,                // Matrix A Col Size
    int b_col                 // Matrix B Col Size
    ) {
    int b_row = a_col;
    int c_row = a_row;
    int c_col = b_col;

    // Local memory to store input and output matrices. A and B hold int8
    // operands; the accumulator and the result stay int32, which is what the
    // RKNPU's INT8_MM_INT8_TO_INT32 mode also produces.
    //
    // The systolic loop needs a whole column of A (dimension 1) and a whole row
    // of B (dimension 2) per cycle. The wide loads additionally write VEC_IN
    // consecutive elements of one row per cycle: B's complete partitioning on
    // dimension 2 already absorbs that, but A's row would be a single bank, so
    // A is partitioned completely in both dimensions, like C. At 32 x 32 int8
    // that is 1 KB of registers.
    char localA[MAX_SIZE][MAX_SIZE] __attribute__((xcl_array_partition(complete, 0)));
    char localB[MAX_SIZE][MAX_SIZE] __attribute__((xcl_array_partition(complete, 2)));
    int localC[MAX_SIZE][MAX_SIZE] __attribute__((xcl_array_partition(complete, 0)));

    // Burst reads on input matrices from global memory, VEC_IN elements per beat
    // Read Input A
    __attribute__((xcl_pipeline_loop(1))) __attribute__((xcl_loop_tripcount(c_size * c_size / VEC_IN,
                                                                           c_size * c_size / VEC_IN))) readA
        : for (int loc = 0, i = 0, j = 0; loc < (a_row * a_col) / VEC_IN; loc++, j += VEC_IN) {
        if (j == a_col) {
            i++;
            j = 0;
        }
        char lane[VEC_IN] __attribute__((xcl_array_partition(complete, 1)));
        vstore16(a[loc], 0, lane);
        __attribute__((opencl_unroll_hint(VEC_IN))) for (int e = 0; e < VEC_IN; e++) localA[i][j + e] = lane[e];
    }

    // Read Input B
    __attribute__((xcl_pipeline_loop(1))) __attribute__((xcl_loop_tripcount(c_size * c_size / VEC_IN,
                                                                           c_size * c_size / VEC_IN))) readB
        : for (int loc = 0, i = 0, j = 0; loc < (b_row * b_col) / VEC_IN; loc++, j += VEC_IN) {
        if (j == b_col) {
            i++;
            j = 0;
        }
        char lane[VEC_IN] __attribute__((xcl_array_partition(complete, 1)));
        vstore16(b[loc], 0, lane);
        __attribute__((opencl_unroll_hint(VEC_IN))) for (int e = 0; e < VEC_IN; e++) localB[i][j + e] = lane[e];
    }

    // Perform systolic matrix multiply
    // local matrices localA and localB have been partitioned in dimensions
    // 1 and 2 respectively. local matrix C has been partitioned completely

    // This partitioning enables to access MAX_SIZE elements in parallel in
    // the local matrices. Because of the mode of access of array elements,
    // we are able to perform MAX_SIZE*MAX_SIZE operations in parallel.

    // Note : i, j and k loops are interchanged.

    // The top loop systolic1 runs only for a_col iterations instead of
    // MAX_SIZE like the inner loops. The inner loops have fixed loop
    // iteration counts to enable complete unroll

    // The following diagram explains how the matrix multiply happens
    //
    //        B_0        B_1        B_2        B_3
    //         |          |          |          |
    //         v          v          v          v
    //        ___        ___        ___        ___
    //       |   |      |   |      |   |      |   |
    //  A0_->|C00| ---- |C01| ---- |C02| ---- |C03|
    //       |___|      |___|      |___|      |___|
    //         |          |          |          |
    //        ___        ___        ___        ___
    //       |   |      |   |      |   |      |   |
    //  A1_->|C10| ---- |C11| ---- |C12| ---- |C13|
    //       |___|      |___|      |___|      |___|
    //         |          |          |          |
    //        ___        ___        ___        ___
    //       |   |      |   |      |   |      |   |
    //  A2_->|C20| ---- |C21| ---- |C22| ---- |C23|
    //       |___|      |___|      |___|      |___|
    //         |          |          |          |
    //        ___        ___        ___        ___
    //       |   |      |   |      |   |      |   |
    //  A3_->|C30| ---- |C31| ---- |C32| ---- |C33|
    //       |___|      |___|      |___|      |___|

    __attribute__((xcl_pipeline_loop(1))) __attribute__((xcl_loop_tripcount(c_size, c_size))) systolic1
        : for (int k = 0; k < a_col; k++) {
        __attribute__((xcl_loop_tripcount(c_size, c_size))) systolic2 : for (int i = 0; i < MAX_SIZE; i++) {
            __attribute__((xcl_loop_tripcount(c_size, c_size))) systolic3 : for (int j = 0; j < MAX_SIZE; j++) {
                // Get previous sum
                int last = (k == 0) ? 0 : localC[i][j];

                // Update current sum
                // Handle boundary conditions
                int a_val = (i < a_row && k < a_col) ? (int)localA[i][k] : 0;
                int b_val = (k < b_row && j < b_col) ? (int)localB[k][j] : 0;
                int result = last + a_val * b_val;

                // Write back results
                localC[i][j] = result;
            }
        }
    }

    // Burst write from output matrices to global memory, VEC_OUT elements per beat
    // Burst write from matrix C
    __attribute__((xcl_pipeline_loop(1))) __attribute__((xcl_loop_tripcount(c_size * c_size / VEC_OUT,
                                                                           c_size * c_size / VEC_OUT))) writeC
        : for (int loc = 0, i = 0, j = 0; loc < (c_row * c_col) / VEC_OUT; loc++, j += VEC_OUT) {
        if (j == c_col) {
            i++;
            j = 0;
        }
        int lane[VEC_OUT] __attribute__((xcl_array_partition(complete, 1)));
        __attribute__((opencl_unroll_hint(VEC_OUT))) for (int e = 0; e < VEC_OUT; e++) lane[e] = localC[i][j + e];
        c[loc] = vload4(0, lane);
    }
}
