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

    Arguments :

        int *in1   (input)     --> Input  Matrix 1
        int *in2   (input)     --> Input  Matrix 2
        int *out_r   (output)    --> Output Matrix
        int  size  (input)     --> Size of one dimension of the matrices

    Kernel Configuration :

        Matrices of upto size (MAX_SIZE x MAX_SIZE) [MAX_SIZE = 64 defined
below]
*/

#define MAX_SIZE 32


__constant int c_size = MAX_SIZE;

__kernel void mmult(
    const __global int* in1,
    const __global int* in2,
    __global int* out_r,
    int size
) {
    __local int A[MAX_SIZE][MAX_SIZE];
    __local int B[MAX_SIZE][MAX_SIZE];
    __local int C[MAX_SIZE][MAX_SIZE];
    int temp_sum[MAX_SIZE];

    #pragma unroll
    for (int itr = 0, i = 0, j = 0; itr < size * size; itr++, j++) {
        if (j == size) {
            j = 0;
            i++;
        }
        A[i][j] = in1[itr];
    }

    #pragma unroll
    for (int itr = 0, i = 0, j = 0; itr < size * size; itr++, j++) {
        if (j == size) {
            j = 0;
            i++;
        }
        B[i][j] = in2[itr];
    }

    #pragma unroll
    for (int i = 0; i < size; i++) {
        #pragma unroll
        for (int k = 0; k < size; k++) {
            #pragma unroll
            for (int j = 0; j < MAX_SIZE; j++) {
                int result = (k == 0) ? 0 : temp_sum[j];
                result += A[i][k] * B[k][j];
                temp_sum[j] = result;
                if (k == size - 1) C[i][j] = result;
            }
        }
    }
    
    #pragma unroll
    for (int itr = 0, i = 0, j = 0; itr < size * size; itr++, j++) {
        if (j == size) {
            j = 0;
            i++;
        }
        out_r[itr] = C[i][j];
    }
}