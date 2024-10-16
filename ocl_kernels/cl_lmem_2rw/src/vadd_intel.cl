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
Description:
    OpenCL Kernel to showcase 2 parallel read/write from Local Memory
    Description: This is vector addition to demonstrate how to utilized both
    ports of Local Memory.
*******************************************************************************/

#define BUFFER_SIZE 1024
#define DATA_SIZE 4096

// Tripcount identifiers
__constant int c_size = DATA_SIZE / BUFFER_SIZE;
__constant int c_chunk_size = BUFFER_SIZE;


__kernel void vadd(const __global uint* in1,
                   const __global uint* in2,
                   __global uint* out_r,
                   int size
                   ) {
    __local uint v1_buffer[BUFFER_SIZE];
    __local uint v2_buffer[BUFFER_SIZE];
    __local uint vout_buffer[BUFFER_SIZE];

    for (int i = 0; i < size; i += BUFFER_SIZE) {
        int chunk_size = BUFFER_SIZE;

        if ((i + BUFFER_SIZE) > size) chunk_size = size - i;

        #pragma unroll
        for (int j = 0; j < chunk_size; j++) {
            v1_buffer[j] = in1[i + j];
            v2_buffer[j] = in2[i + j];
        }

        
        #pragma unroll 2
        for (int j = 0; j < chunk_size; j++) {
            vout_buffer[j] = v1_buffer[j] + v2_buffer[j];
        }

        #pragma unroll
        for (int j = 0; j < chunk_size; j++) {
            out_r[i + j] = vout_buffer[j];
        }
    }
}
