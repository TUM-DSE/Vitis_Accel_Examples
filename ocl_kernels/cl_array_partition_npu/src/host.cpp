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
#include "xcl2.hpp"
#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <random>
#include <vector>

using std::default_random_engine;
using std::generate;
using std::uniform_int_distribution;
using std::vector;

// row major, int8 operands accumulated into int32 (RKNN matmul type 2)
void matmul(int* C, const int8_t* A, const int8_t* B, int M) {
    for (int k = 0; k < M; k++) {
        for (int j = 0; j < M; j++) {
            for (int i = 0; i < M; i++) {
                C[k * M + j] += (int)A[k * M + i] * (int)B[i * M + j];
            }
        }
    }
}

// Full signed int8 range, as the RKNN matmul demo also uses, so sign handling is
// exercised on every device instead of only the positive path.
int8_t gen_random() {
    static default_random_engine e;
    static uniform_int_distribution<int> dist(-128, 127);

    return (int8_t)dist(e);
}

template <typename T>
void print(const T* data, int columns, int rows) {
    for (int r = 0; r < 10; r++) {
        for (int c = 0; c < 10; c++) {
            printf("%4d ", (int)data[r * columns + c]);
        }
        printf("…\n");
    }
    for (int r = 0; r < 10; r++) {
        printf("   %s ", "…");
    }
    printf("⋱\n\n");
}

void verify(vector<int, aligned_allocator<int> >& gold, vector<int, aligned_allocator<int> >& output,
            int columns) {
    for (int i = 0; i < (int)output.size(); i++) {
        if (output[i] != gold[i]) {
            printf("Mismatch %d: gold: %d device: %d\n", i, gold[i], output[i]);
            print(output.data(), columns, columns);
            exit(EXIT_FAILURE);
        }
    }
}

// Benchmarks the array-partitioned matmul kernel (C = A x B, int32, row major)
// so the same GEMM can be measured on FPGA, GPU and NPU.
int main(int argc, char** argv) {
    if (argc != 2) {
        std::cout << "Usage: " << argv[0] << " <XCLBIN File>" << std::endl;
        return EXIT_FAILURE;
    }

    std::string binaryFile = argv[1];
    static const int columns = 128;
    static const int rows = 128;
    cl_int err;
    cl::Program program;
    cl::CommandQueue q;
    cl::Context context;

    vector<int8_t, aligned_allocator<int8_t> > A(columns * rows);
    vector<int8_t, aligned_allocator<int8_t> > B(columns * rows);
    vector<int, aligned_allocator<int> > C(columns * rows, 0);
    vector<int, aligned_allocator<int> > gold(columns * rows, 0);

    generate(begin(A), end(A), gen_random);
    generate(begin(B), end(B), gen_random);

    printf("A:\n");
    print(A.data(), columns, rows);
    printf("B:\n");
    print(B.data(), columns, rows);
    matmul(gold.data(), A.data(), B.data(), columns);

    printf("Gold:\n");
    print(gold.data(), columns, rows);
    auto devices = xcl::get_xil_devices();

    // read_binary_file() is a utility API which will load the binaryFile
    // and will return the pointer to file buffer.
    auto fileBuf = xcl::read_binary_file(binaryFile);
    cl::Program::Binaries bins{{fileBuf.data(), fileBuf.size()}};
    bool valid_device = false;
    for (unsigned int i = 0; i < devices.size(); i++) {
        auto device = devices[i];
        // Creating Context and Command Queue for selected Device
        OCL_CHECK(err, context = cl::Context(device, nullptr, nullptr, nullptr, &err));
        OCL_CHECK(err, q = cl::CommandQueue(context, device, CL_QUEUE_PROFILING_ENABLE, &err));

        std::cout << "Trying to program device[" << i << "]: " << device.getInfo<CL_DEVICE_NAME>() << std::endl;
        program = cl::Program(context, {device}, bins, nullptr, &err);
        if (err != CL_SUCCESS) {
            std::cout << "Failed to program device[" << i << "] with xclbin file!\n";
        } else {
            std::cout << "Device[" << i << "]: program successful!\n";
            valid_device = true;
            break; // we break because we found a valid device
        }
    }
    if (!valid_device) {
        std::cout << "Failed to program any device found, exit!\n";
        exit(EXIT_FAILURE);
    }

    // compute the size of array in bytes
    // int8 in, int32 out: the two directions no longer have the same size
    size_t in_size_bytes = columns * rows * sizeof(int8_t);
    size_t out_size_bytes = columns * rows * sizeof(int);
    OCL_CHECK(err,
              // Device-only buffers: the transfers below are explicit, so the buffers must not
              // be backed by host memory. The Funky backend force-adds CL_MEM_USE_HOST_PTR
              // whenever a host pointer reaches it, which would turn each transfer into a
              // host-side copy instead of a real device transfer.
              cl::Buffer buffer_a(context, CL_MEM_READ_ONLY, in_size_bytes,
                                  nullptr, &err));
    OCL_CHECK(err,
              cl::Buffer buffer_b(context, CL_MEM_READ_ONLY, in_size_bytes,
                                  nullptr, &err));
    OCL_CHECK(err,
              cl::Buffer buffer_c(context, CL_MEM_WRITE_ONLY, out_size_bytes,
                                  nullptr, &err));

    OCL_CHECK(err, cl::Kernel matmul_partition_kernel(program, "matmul_partition", &err));
    OCL_CHECK(err, err = matmul_partition_kernel.setArg(0, buffer_a));
    OCL_CHECK(err, err = matmul_partition_kernel.setArg(1, buffer_b));
    OCL_CHECK(err, err = matmul_partition_kernel.setArg(2, buffer_c));
    OCL_CHECK(err, err = matmul_partition_kernel.setArg(3, columns));

    cl::Event event_kernel;
    cl::Event event_data_to_fpga;
    cl::Event event_data_to_fpga_2;
    cl::Event event_data_to_host;
    const int n_warmup = 0;
    const int n_reps = 16000;
    uint64_t nstimestart = 0;
    uint64_t nstimeend = 0;
    uint64_t time_kernel_ocl = 0;
    uint64_t time_data_to_xpu_ocl = 0;
    uint64_t time_data_to_host_ocl = 0;
    // Host-clock accumulator for data-transfer + kernel-execution time only: each interval below
    // is opened right before an OpenCL enqueue call and closed right after it (and any q.finish())
    // completes, so host-side work (loop bookkeeping) is never included.
    uint64_t time_xpu = 0;

    // This is required for proper time measurements in Proteus. We add it here
    // as well to have the same host code for Proteus and native.
    q.finish();

    // Running the array-partitioned matmul kernel
    for (int i = 0; i < n_warmup + n_reps; i++) {
        auto t_xpu_0 = std::chrono::high_resolution_clock::now();
        OCL_CHECK(err, err = q.enqueueWriteBuffer(buffer_a, CL_FALSE, 0, in_size_bytes,
                                                  A.data(), nullptr, &event_data_to_fpga));
        OCL_CHECK(err, err = q.enqueueWriteBuffer(buffer_b, CL_FALSE, 0, in_size_bytes,
                                                  B.data(), nullptr, &event_data_to_fpga_2));
        OCL_CHECK(err, err = q.finish());
        auto t_xpu_1 = std::chrono::high_resolution_clock::now();
        time_xpu += std::chrono::duration_cast<std::chrono::nanoseconds>(t_xpu_1 - t_xpu_0).count();

        auto t_xpu_2 = std::chrono::high_resolution_clock::now();
        OCL_CHECK(err, err = q.enqueueTask(matmul_partition_kernel, nullptr, &event_kernel));
        OCL_CHECK(err, err = q.finish());
        auto t_xpu_3 = std::chrono::high_resolution_clock::now();
        time_xpu += std::chrono::duration_cast<std::chrono::nanoseconds>(t_xpu_3 - t_xpu_2).count();

        auto t_xpu_4 = std::chrono::high_resolution_clock::now();
        OCL_CHECK(err, err = q.enqueueReadBuffer(buffer_c, CL_FALSE, 0, out_size_bytes,
                                                 C.data(), nullptr, &event_data_to_host));
        OCL_CHECK(err, err = q.finish());
        auto t_xpu_5 = std::chrono::high_resolution_clock::now();
        time_xpu += std::chrono::duration_cast<std::chrono::nanoseconds>(t_xpu_5 - t_xpu_4).count();

        OCL_CHECK(err, err = event_data_to_fpga.getProfilingInfo<uint64_t>(CL_PROFILING_COMMAND_START, &nstimestart));
        OCL_CHECK(err, err = event_data_to_fpga.getProfilingInfo<uint64_t>(CL_PROFILING_COMMAND_END, &nstimeend));
        time_data_to_xpu_ocl += nstimeend - nstimestart;

        OCL_CHECK(err, err = event_data_to_fpga_2.getProfilingInfo<uint64_t>(CL_PROFILING_COMMAND_START, &nstimestart));
        OCL_CHECK(err, err = event_data_to_fpga_2.getProfilingInfo<uint64_t>(CL_PROFILING_COMMAND_END, &nstimeend));
        time_data_to_xpu_ocl += nstimeend - nstimestart;

        OCL_CHECK(err, err = event_kernel.getProfilingInfo<uint64_t>(CL_PROFILING_COMMAND_START, &nstimestart));
        OCL_CHECK(err, err = event_kernel.getProfilingInfo<uint64_t>(CL_PROFILING_COMMAND_END, &nstimeend));
        time_kernel_ocl += nstimeend - nstimestart;

        OCL_CHECK(err, err = event_data_to_host.getProfilingInfo<uint64_t>(CL_PROFILING_COMMAND_START, &nstimestart));
        OCL_CHECK(err, err = event_data_to_host.getProfilingInfo<uint64_t>(CL_PROFILING_COMMAND_END, &nstimeend));
        time_data_to_host_ocl += nstimeend - nstimestart;
    }

    verify(gold, C, columns);

    double ns_per_s = 1000000000;
    std::cout << "app_name,in_size,out_size,reps_warmup,reps,time_xpu,time_data_to_xpu,time_kernel,time_data_to_host\n"
              << "cl_array_partition_npu,"
              << in_size_bytes * 2 << ","
              << out_size_bytes << ","
              << n_warmup << ","
              << n_reps << ","
              << time_xpu / ns_per_s << ","
              << time_data_to_xpu_ocl / ns_per_s << ","
              << time_kernel_ocl / ns_per_s << ","
              << time_data_to_host_ocl / ns_per_s
              << "\n";

    printf("TEST PASSED\n\n");

    return EXIT_SUCCESS;
}
