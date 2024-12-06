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
    OpenCL Dataflow Example using xcl_dataflow attribute
    This is example of vector addition to demonstrate OpenCL Dataflow
xcl_dataflow
    functionality to perform task/subfunction level parallelism using
xcl_dataflow
    attribute. OpenCL xcl_dataflow instruct compiler to run subfunctions inside
kernel
    concurrently. In this Example, the kernel calls run_subfunc API to perform
vector
    addition implementation and which inturn is divided into three sub-function
APIs
    as below:

    1) read_input():
        This API reads the input vector from Global Memory and writes it into
        'buffer_in'.

    2) compute_add():
        This API reads the input vector from 'buffer_in' and increment the value
        by user specified increment. It writes the result into 'buffer_out'.

    3) write_result():
        This API reads the result vector from 'buffer_out' and write the result
        into Global Memory Location.

    Data Flow based Adder will be implemented as below:
                    _____________
                    |             |<----- Input Vector from Global Memory
                    |  read_input |       __
                    |_____________|----->|  |
                     _____________       |  | buffer_in
                    |             |<-----|__|
                    | compute_add |       __
                    |_____________|----->|  |
                     _____________       |  | buffer_out
                    |              |<----|__|
                    | write_result |
                    |______________|-----> Output result to Global Memory


*******************************************************************************/
#define DATA_SIZE (8 * 1024 * 1024) // * sizeof(int) = 32 MB
#define BUFFER_SIZE (8 * 1024 * 1024)

constant int c_size = DATA_SIZE;

static void read_input(__global int* in, int* buffer_in, int size) {
    #pragma unroll
    for (int i = 0; i < size; i++) {
        buffer_in[i] = in[i];
    }
}


static void compute_add(int* buffer_in, int* buffer_out, int inc, int size) {
    #pragma unroll
    for (int i = 0; i < size; i++) {
        buffer_out[i] = buffer_in[i] + inc;
    }
}


static void write_result(__global int* out, int* buffer_out, int size) {
    #pragma unroll
    for (int i = 0; i < size; i++) {
        out[i] = buffer_out[i];
    }
}


void run_subfunc(__global int* in, __global int* out, int inc, int size) {
    int buffer_in[BUFFER_SIZE];
    int buffer_out[BUFFER_SIZE];

    read_input(in, buffer_in, size);
    compute_add(buffer_in, buffer_out, inc, size);
    write_result(out, buffer_out, size);
}


__kernel void adder(__global int* in, __global int* out, int inc, int size) {
    run_subfunc(in, out, inc, size);
}
