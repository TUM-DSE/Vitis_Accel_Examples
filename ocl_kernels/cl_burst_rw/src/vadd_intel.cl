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
    OpenCL Kernel Example to demonstrate burst read and write access from
    Global Memory
*******************************************************************************/

// Below Macro should be commented for optimized design. Xilinx recommends to
// use for loop approach instead of async_work_grop_copy() API burst read/write
//#define USE_ASYNC_API

#define DATA_SIZE 2048
#define BURSTBUFFERSIZE 256

constant int c_size = BURSTBUFFERSIZE;
constant int c_len = DATA_SIZE / BURSTBUFFERSIZE;

__kernel void vadd(__global int* a, int size, int inc_value) {
    __local int burstbuffer[BURSTBUFFERSIZE];

    for (int i = 0; i < size; i += BURSTBUFFERSIZE) {
        int chunk_size = BURSTBUFFERSIZE;

        if ((i + BURSTBUFFERSIZE) > size) chunk_size = size - i;

        for (int j = 0; j < chunk_size; j++) {
            burstbuffer[j] = a[i + j];
        }

        for (int j = 0; j < chunk_size; j++) {
            burstbuffer[j] = burstbuffer[j] + inc_value;
            a[i + j] = burstbuffer[j];
        }
    }
}
