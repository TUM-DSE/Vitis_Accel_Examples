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

#define MAX_DIM 64

constant int c_size = MAX_DIM;

__kernel void matmul_naive(
    const __global int* in1,
    const __global int* in2,
    __global int* out_r,
    int dim)
{
    __local int A[MAX_DIM * MAX_DIM];
    __local int B[MAX_DIM * MAX_DIM];
    __local int C[MAX_DIM * MAX_DIM];


    #pragma unroll
    for (int i = 0; i < dim * dim; i++) {
        A[i] = in1[i];
    }

    #pragma unroll
    for (int i = 0; i < dim * dim; i++) {
        B[i] = in2[i];
    }

    #pragma unroll
    for (int i = 0; i < dim; i++) {
        #pragma unroll
        for (int j = 0; j < MAX_DIM; j++) {
            int result = 0;
            #pragma unroll
            for (int k = 0; k < dim; k++) {
                result += A[i * dim + k] * B[k * dim + j];
            }
            C[i * dim + j] = result;
        }
    }

    #pragma unroll
    for (int i = 0; i < dim * dim; i++) {
        out_r[i] = C[i];
    }
}

// ToDO (Maybe): Equivalent to xcl_array_partition in Intel? 
__kernel void matmul_partition(
    const __global int* in1,
    const __global int* in2,
    __global int* out_r,
    int dim)
{
    __local int A[MAX_DIM * MAX_DIM];
    __local int B[MAX_DIM * MAX_DIM];
    __local int C[MAX_DIM * MAX_DIM];

    #pragma unroll
    for (int i = 0; i < dim * dim; i++) {
        A[i] = in1[i];
    }

    #pragma unroll
    for (int i = 0; i < dim * dim; i++) {
        B[i] = in2[i];
    }

    #pragma unroll
    for (int i = 0; i < dim; i++) {
        #pragma unroll
        for (int j = 0; j < dim; j++) {
            int result = 0;
            #pragma unroll
            for (int k = 0; k < MAX_DIM; k++) {
                result += A[i * MAX_DIM + k] * B[k * MAX_DIM + j];
            }
            C[i * MAX_DIM + j] = result;
        }
    }

    #pragma unroll
    for (int i = 0; i < dim * dim; i++) {
        out_r[i] = C[i];
    }
}
