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

    Arguments :

        int *a     (input )  --> Input  Matrix A
        int *b     (input )  --> Input  Matrix B
        int *c     (output)  --> Output Matrix
        int  a_row (input )  --> Row Size Matrix A
        int  a_col (input )  --> Col Size Matrix A
        int  b_col (input )  --> Col Size Matrix B

    Kernel Configuration :

        Max Size    --> 12

    Note :
        Max Size is dependent on the available DSP resources in the FPGA
*/

#define MAX_SIZE 12

// Tripcount constant
__constant int c_size = MAX_SIZE;

// ToDo(Maybe): Equivalent to xcl_array_partition in Intel?
__kernel void mmult(__global int* a,
                    __global int* b,
                    __global int* c,
                    int a_row,
                    int a_col,
                    int b_col) {
    int b_row = a_col;
    int c_row = a_row;
    int c_col = b_col;

    __local int localA[MAX_SIZE][MAX_SIZE];
    __local int localB[MAX_SIZE][MAX_SIZE];
    __local int localC[MAX_SIZE][MAX_SIZE];

    #pragma unroll
    for (int loc = 0, i = 0, j = 0; loc < a_row * a_col; loc++, j++) {
        if (j == a_col) {
            i++;
            j = 0;
        }
        localA[i][j] = a[loc];
    }


    #pragma unroll
    for (int loc = 0, i = 0, j = 0; loc < b_row * b_col; loc++, j++) {
        if (j == b_col) {
            i++;
            j = 0;
        }
        localB[i][j] = b[loc];
    }


    #pragma unroll
    for (int k = 0; k < a_col; k++) {
        #pragma unroll
        for (int i = 0; i < MAX_SIZE; i++) {
            #pragma unroll
            for (int j = 0; j < MAX_SIZE; j++) {
                int last = (k == 0) ? 0 : localC[i][j];

                int a_val = (i < a_row && k < a_col) ? localA[i][k] : 0;
                int b_val = (k < b_row && j < b_col) ? localB[k][j] : 0;
                int result = last + a_val * b_val;

                localC[i][j] = result;
            }
        }
    }

    #pragma unroll
    for (int loc = 0, i = 0, j = 0; loc < c_row * c_col; loc++, j++) {
        if (j == c_col) {
            i++;
            j = 0;
        }
        c[loc] = localC[i][j];
    }
}