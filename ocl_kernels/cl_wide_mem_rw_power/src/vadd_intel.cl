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
    OpenCL Wide Memory Read/write Example
    Description: This is vector addition to demonstrate Wide Memory access of
    512bit Datawidth using uint16 openCL vector datatype.
*******************************************************************************/

#define DATA_SIZE 16384
#define LOCAL_MEM_SIZE 128
#define VECTOR_SIZE 16 // Using uint16 datatype, so vector size is 16


__constant int c_size = LOCAL_MEM_SIZE;
__constant int c_len = ((DATA_SIZE - 1) / VECTOR_SIZE + 1) / LOCAL_MEM_SIZE;


__kernel void vadd(const __global uint16* in1,
                   const __global uint16* in2,
                   __global uint16* out,
                   int size
) {
    __local uint16 v1_local[LOCAL_MEM_SIZE];
    __local uint16 result_local[LOCAL_MEM_SIZE];
    
    int size_in16 = (size - 1) / VECTOR_SIZE + 1;

    #pragma unroll
    for (int i = 0; i < size_in16; i += LOCAL_MEM_SIZE) {
        int chunk_size = LOCAL_MEM_SIZE;

        if ((i + LOCAL_MEM_SIZE) > size_in16) chunk_size = size_in16 - i;

        #pragma unroll
        for (int j = 0; j < chunk_size; j++) {
            v1_local[j] = in1[i + j];
        }

        #pragma unroll
        for (int j = 0; j < chunk_size; j++) {
            uint16 tmpV1 = v1_local[j];
            uint16 tmpV2 = in2[i + j];
            result_local[j] = tmpV1 + tmpV2;
        }

        #pragma unroll
        for (int j = 0; j < chunk_size; j++) {
            out[i + j] = result_local[j];
        }
    }
}