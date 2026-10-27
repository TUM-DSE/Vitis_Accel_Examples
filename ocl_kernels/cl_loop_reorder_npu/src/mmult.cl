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

    This is a matrix multiplication based example which showcases the how
    reordering the loops helps achieve better pipeline initiation interval(II)
    and better performance.

*******************************************************************************/

/*

Kernel Description :

    Matrix multiply for matrices upto sizes (MAX_SIZE x MAX_SIZE) [MAX_SIZE
defined below].
    This example showcases how reordering the loops helps achieve a better
    pipeline initiation interval (II) and better performance.

    This variant multiplies int8 operands into an int32 result, matching the
    RKNPU's INT8_MM_INT8_TO_INT32 matmul mode, so the same GEMM can be measured
    on FPGA, GPU and NPU.

    Arguments :

        char16 *in1   (input)     --> Input  Matrix 1 (int8, VEC_IN per beat)
        char16 *in2   (input)     --> Input  Matrix 2 (int8, VEC_IN per beat)
        int4   *out_r (output)    --> Output Matrix   (int32, VEC_OUT per beat)
        int  size     (input)     --> Size of one dimension of the matrices

    Kernel Configuration :

        Matrices of upto size (MAX_SIZE x MAX_SIZE) [MAX_SIZE = 64 defined
below]. 64 is already a shape the RK3588's int8 matmul accepts (K and N
must be multiples of 32), so the original example's size is kept.
*/

// Maximum Array Size
#define MAX_SIZE 64 // int8 inputs: 2 * 64 * 64 * 1 B = 8 KB, int32 output: 16 KB

// Elements moved per global-memory beat.
//
// Vitis infers the AXI master's data width from the pointer type, so a plain
// char* would give the input ports a width of 8 bits: one byte per beat, 4096
// beats to load a matrix. Declaring the arguments as vectors makes all three
// ports 128 bits wide instead. char16 is the widest char vector OpenCL defines,
// and int4 matches it at 32 bits per element.
//
// Both loads drop from 4096 beats to 256 and the store from 4096 to 1024. size
// must be a multiple of VEC_IN (the host checks this).
#define VEC_IN 16 // char16: 16 int8 operands per 128-bit beat
#define VEC_OUT 4 // int4:    4 int32 results per 128-bit beat

// Tripcount identifiers
__constant int c_size = MAX_SIZE;

// Computes matrix multiply
// C = AxB, where A, B and C are square matrices of dimension (sizexsize). A and
// B are int8, C is an int32 accumulator.
kernel __attribute__((reqd_work_group_size(1, 1, 1))) void mmult(
    const __global char16* in1, // Read-Only Matrix 1 (int8, VEC_IN per beat)
    const __global char16* in2, // Read-Only Matrix 2 (int8, VEC_IN per beat)
    __global int4* out_r,       // Output Result (int32, VEC_OUT per beat)
    int size) {                 // Local memory to store input and output matrices
    // Local memory is implemented as BRAM memory blocks.
    //
    // All three matrices are kept as rows of beats, M[row][beat][lane], rather
    // than as M[row][col]. A wide load or store touches VEC_IN (VEC_OUT)
    // consecutive elements of one row per cycle, so each row has to be split
    // into that many banks. With M[i][j + e] HLS cannot tell that
    // (j + e) % VEC_IN == e, has to select among all banks at run time, and
    // fails to pipeline the loop; with the lane as its own index, partitioning
    // that index completely gives each lane its own bank by construction.
    //
    // A is read one element per cycle by the multiply, like in the original, so
    // only the lanes the wide load needs are split out.
    char A[MAX_SIZE][MAX_SIZE / VEC_IN][VEC_IN] __attribute__((xcl_array_partition(complete, 3)));
    // B and C are accessed a whole row per cycle by the unrolled lreorder3,
    // which is what the original's complete partitioning on dimension 2 gives;
    // here that row is spread over dimensions 2 and 3.
    char B[MAX_SIZE][MAX_SIZE / VEC_IN][VEC_IN] __attribute__((xcl_array_partition(complete, 2)))
        __attribute__((xcl_array_partition(complete, 3)));
    int C[MAX_SIZE][MAX_SIZE / VEC_OUT][VEC_OUT] __attribute__((xcl_array_partition(complete, 2)))
        __attribute__((xcl_array_partition(complete, 3)));
    int temp_sum[MAX_SIZE] __attribute__((xcl_array_partition(complete, 1)));

    // Burst reads on input matrices from global memory, VEC_IN elements per beat
    // Burst read for matrix A
    __attribute__((xcl_pipeline_loop(1))) __attribute__((xcl_loop_tripcount(c_size * c_size / VEC_IN,
                                                                           c_size * c_size / VEC_IN))) readA
        : for (int itr = 0, i = 0, jb = 0; itr < (size * size) / VEC_IN; itr++, jb++) {
        if (jb == size / VEC_IN) {
            jb = 0;
            i++;
        }
        char lane[VEC_IN] __attribute__((xcl_array_partition(complete, 1)));
        vstore16(in1[itr], 0, lane);
        __attribute__((opencl_unroll_hint(VEC_IN))) for (int e = 0; e < VEC_IN; e++) A[i][jb][e] = lane[e];
    }

    // Burst read for matrix B
    __attribute__((xcl_pipeline_loop(1))) __attribute__((xcl_loop_tripcount(c_size * c_size / VEC_IN,
                                                                           c_size * c_size / VEC_IN))) readB
        : for (int itr = 0, i = 0, jb = 0; itr < (size * size) / VEC_IN; itr++, jb++) {
        if (jb == size / VEC_IN) {
            jb = 0;
            i++;
        }
        char lane[VEC_IN] __attribute__((xcl_array_partition(complete, 1)));
        vstore16(in2[itr], 0, lane);
        __attribute__((opencl_unroll_hint(VEC_IN))) for (int e = 0; e < VEC_IN; e++) B[i][jb][e] = lane[e];
    }

    // Performs matrix multiply over matrices A and B and stores the result
    // in C. All the matrices are square matrices of the form (size x size)

    // Pipeline attribute is specified for the innermost loop (lreorder3)
    // and the order of the loops lreorder2 and lreorder 3 are changed here.

    // When the iteration variables j and k are interchanged between the loops,
    // lreoder2 and lreorder3, the pipeline initiation interval (II) improves
    // and becomes 1 (ideal).

    // Also the reordering avoids creating an adder tree for calculating the
    // sum(output) of a single output element

    // lreorder1: for (int i = 0; i < size; i++) {
    //     lreorder2: for (int j = 0; j < size; j++) {
    //     __attribute__((xcl_pipeline_loop))
    //         lreorder3: for (int k = 0; k < MAX_SIZE; k++) {
    //             int result = (k == 0) ? 0 : temp_sum[j];
    //             result += A[i][k] * B[k][j];
    //             temp_sum[j] = result;
    //             if (k== size -1) C[i][j] = result;
    //         }
    //     }
    // }

    // The above code snippet of the Matrix Multiply kernel in which the loops
    // lreorder2 and lreorder3 are not interchanged, gives a pipeline initiation
    // interval (II) of 32

    // Calculate matrix multiplication using local data buffer based on input size
    // and write results into local buffer for C. j is a compile-time constant in
    // the unrolled lreorder3, so B's and C's indices select a fixed bank.
    __attribute__((xcl_loop_tripcount(c_size, c_size))) lreorder1 : for (int i = 0; i < size; i++) {
        __attribute__((xcl_pipeline_loop(1))) __attribute__((xcl_loop_tripcount(c_size, c_size))) lreorder2
            : for (int k = 0; k < size; k++) {
            int a_val = (int)A[i][k / VEC_IN][k % VEC_IN];
            __attribute__((xcl_loop_tripcount(c_size, c_size))) lreorder3 : for (int j = 0; j < MAX_SIZE; j++) {
                int result = (k == 0) ? 0 : temp_sum[j];
                result += a_val * (int)B[k][j / VEC_IN][j % VEC_IN];
                temp_sum[j] = result;
                if (k == size - 1) C[i][j / VEC_OUT][j % VEC_OUT] = result;
            }
        }
    }

    // Burst write from output matrices to global memory, VEC_OUT elements per beat
    // Burst write from matrix C
    __attribute__((xcl_pipeline_loop(1))) __attribute__((xcl_loop_tripcount(c_size * c_size / VEC_OUT,
                                                                           c_size * c_size / VEC_OUT))) writeC
        : for (int itr = 0, i = 0, jb = 0; itr < (size * size) / VEC_OUT; itr++, jb++) {
        if (jb == size / VEC_OUT) {
            jb = 0;
            i++;
        }
        int lane[VEC_OUT] __attribute__((xcl_array_partition(complete, 1)));
        __attribute__((opencl_unroll_hint(VEC_OUT))) for (int e = 0; e < VEC_OUT; e++) lane[e] = C[i][jb][e];
        out_r[itr] = vload4(0, lane);
    }
}
