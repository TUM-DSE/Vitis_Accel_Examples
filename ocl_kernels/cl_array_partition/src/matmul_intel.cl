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
#define MAX_SIZE 16

__kernel void matmul(
    const __global int* in1,
    const __global int* in2,
    __global int* out,
    int size) {
    __local int A[MAX_SIZE][MAX_SIZE];
    __local int B[MAX_SIZE][MAX_SIZE];
    __local int C[MAX_SIZE][MAX_SIZE];
    int temp_sum[MAX_SIZE];

    for (int itr = 0, i = 0, j = 0; itr < size * size; itr++, j++) {
        if (j == size) {
            j = 0;
            i++;
        }
        A[i][j] = in1[itr];
    }

    for (int itr = 0, i = 0, j = 0; itr < size * size; itr++, j++) {
        if (j == size) {
            j = 0;
            i++;
        }
        B[i][j] = in2[itr];
    }

    for (int i = 0; i < size; i++) {
        for (int k = 0; k < size; k++) {
            for (int j = 0; j < MAX_SIZE; j++) {
                int result = (k == 0) ? 0 : temp_sum[j];
                result += A[i][k] * B[k][j];
                temp_sum[j] = result;
                if (k == size - 1) C[i][j] = result;
            }
        }
    }

    for (int itr = 0, i = 0, j = 0; itr < size * size; itr++, j++) {
        if (j == size) {
            j = 0;
            i++;
        }
        out[itr] = C[i][j];
    }
}

// ToDo (Maybe): Is there an equivalent of xcl_array_partition in Intel OpenCL?
__kernel void matmul_partition(
    const __global int* in1,
    const __global int* in2,
    __global int* out,
    int size) {
    int A[MAX_SIZE][MAX_SIZE];

    int B[MAX_SIZE][MAX_SIZE];

    int C[MAX_SIZE][MAX_SIZE];

    int temp_sum[MAX_SIZE];

    for (int itr = 0, i = 0, j = 0; itr < size * size; itr++, j++) {
        if (j == size) {
            j = 0;
            i++;
        }
        A[i][j] = in1[itr];
    }

    for (int itr = 0, i = 0, j = 0; itr < size * size; itr++, j++) {
        if (j == size) {
            j = 0;
            i++;
        }
        B[i][j] = in2[itr];
    }

    for (int i = 0; i < size; i++) {
        for (int k = 0; k < size; k++) {
            for (int j = 0; j < MAX_SIZE; j++) {
                int result = (k == 0) ? 0 : temp_sum[j];
                result += A[i][k] * B[k][j];
                temp_sum[j] = result;
                if (k == size - 1) C[i][j] = result;
            }
        }
    }

    for (int itr = 0, i = 0, j = 0; itr < size * size; itr++, j++) {
        if (j == size) {
            j = 0;
            i++;
        }
        out[itr] = C[i][j];
    }
}