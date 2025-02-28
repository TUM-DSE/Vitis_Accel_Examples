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
#include <cmath>
#include <cassert>
#include <vector>
#include <iomanip>

// Artificial limit for available FPGA memory when over-subscription is enabled
constexpr size_t MEM_LIMIT = 1000 * 1024 * 1024;
// DATA_SIZE should be multiple of 16 as kernel code is using int16 vector
// datatype to read the operands from global memory 16 ints at a time. We aim
// for the 2 input buffers and the output buffer to not fit into MEM_LIMIT to
// simulate memory over-subscription.
constexpr size_t DATA_SIZE = 256 * 1024 * 1024; // 3 buffers (2 input, 1 output) of this size * sizeof(int)

int main(int argc, char** argv) {
    assert(DATA_SIZE % 16 == 0);

    if (argc != 3) {
        std::cout << "Usage: " << argv[0] << " <XCLBIN File> <1: over-subscription, 0: no over-subscription>" << std::endl;
        return EXIT_FAILURE;
    }

    std::string binaryFile = argv[1];
    bool oversub = std::stoi(argv[2]);

    std::cout << "memory over-subscription " << (oversub ? "enabled" : "disabled") << "\n";

    // Allocate Memory in Host Memory
    size_t vector_size_bytes = sizeof(int) * DATA_SIZE;
    std::vector<int, aligned_allocator<int> > source_in1(DATA_SIZE);
    std::vector<int, aligned_allocator<int> > source_in2(DATA_SIZE);
    std::vector<int, aligned_allocator<int> > source_hw_results(DATA_SIZE);
    std::vector<int, aligned_allocator<int> > source_sw_results(DATA_SIZE);

    // Create the test data and Software Result
    for (size_t i = 0; i < DATA_SIZE; i++) {
        source_in1[i] = i;
        source_in2[i] = i * i;
        source_sw_results[i] = i * i + i;
        source_hw_results[i] = 0;
    }

    // OPENCL HOST CODE AREA START
    cl_int err;
    cl::CommandQueue q;
    cl::Context context;
    cl::Kernel krnl_vector_add;
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
        cl::Program program(context, {device}, bins, nullptr, &err);
        if (err != CL_SUCCESS) {
            std::cout << "Failed to program device[" << i << "] with xclbin file!\n";
        } else {
            std::cout << "Device[" << i << "]: program successful!\n";
            OCL_CHECK(err, krnl_vector_add = cl::Kernel(program, "vadd", &err));
            valid_device = true;
            break; // we break because we found a valid device
        }
    }
    if (!valid_device) {
        std::cout << "Failed to program any device found, exit!\n";
        exit(EXIT_FAILURE);
    }

    size_t mem_limit = oversub ? MEM_LIMIT : SIZE_MAX;

    size_t chunk_size = vector_size_bytes;
    if (3 * vector_size_bytes > mem_limit) {
        // Round down to closest multiple of 16
        // TODO: proper alignment, maybe 4096? XRT gives warning about unaligned host pointer
        chunk_size = (mem_limit / 3) & ~0xF;
    }
    size_t num_chunks = std::ceil(vector_size_bytes / (double)chunk_size);
    size_t last_chunk_size = vector_size_bytes % chunk_size;
    if (last_chunk_size == 0) {
        last_chunk_size = chunk_size;
    }
    std::cout << "memory limit:    " << mem_limit << "\n";
    std::cout << "3 * vector size: " << 3 * vector_size_bytes << "\n";
    std::cout << "num chunks:      " << num_chunks << "\n";
    std::cout << "chunk size:      " << chunk_size << "\n";
    std::cout << "last chunk size: " << last_chunk_size << "\n";

    cl::Event event_kernel;
    cl::Event event_data_to_fpga;
    cl::Event event_data_to_host;
    const int iterations = 10;
    std::chrono::high_resolution_clock::time_point start_time, end_time;
    std::chrono::duration<double> duration;
    int64_t nstime_cpu = 0;
    uint64_t nstimestart = 0;
    uint64_t nstimeend = 0;
    uint64_t nstime_kernel_ocl = 0;
    uint64_t nstime_data_to_fpga_ocl = 0;
    uint64_t nstime_data_to_host_ocl = 0;

    for (int i = 0; i < iterations; i++) {
        for (size_t i = 0; i < num_chunks; i++) {
            size_t cur_chunk_size = chunk_size;
            if (i == num_chunks - 1) {
                cur_chunk_size = last_chunk_size;
            }

            auto buf_offset = i * chunk_size / sizeof(int);
            // Allocate Buffer in Global Memory
            OCL_CHECK(err, cl::Buffer buffer_in1(context, CL_MEM_USE_HOST_PTR | CL_MEM_READ_ONLY, cur_chunk_size,
                                                source_in1.data() + buf_offset, &err));
            OCL_CHECK(err, cl::Buffer buffer_in2(context, CL_MEM_USE_HOST_PTR | CL_MEM_READ_ONLY, cur_chunk_size,
                                                source_in2.data() + buf_offset, &err));
            OCL_CHECK(err, cl::Buffer buffer_output(context, CL_MEM_USE_HOST_PTR | CL_MEM_WRITE_ONLY, cur_chunk_size,
                                                source_hw_results.data() + buf_offset, &err));

            // Set the kernel arguments
            int nargs = 0;
            OCL_CHECK(err, err = krnl_vector_add.setArg(nargs++, buffer_in1));
            OCL_CHECK(err, err = krnl_vector_add.setArg(nargs++, buffer_in2));
            OCL_CHECK(err, err = krnl_vector_add.setArg(nargs++, buffer_output));
            OCL_CHECK(err, err = krnl_vector_add.setArg(nargs++, (int)cur_chunk_size));

            // This is required for proper time measurements in Proteus. We add it here
            // as well to have the same host code for Proteus and native.
            q.finish();

            start_time = std::chrono::high_resolution_clock::now();

            for (int i = 0; i < iterations; i++) {
                OCL_CHECK(err, err = q.enqueueMigrateMemObjects({buffer_in1, buffer_in2}, 0 /* 0 means from host*/, nullptr, &event_data_to_fpga));
                OCL_CHECK(err, err = q.finish());
                OCL_CHECK(err, err = q.enqueueTask(krnl_vector_add, nullptr, &event_kernel));
                OCL_CHECK(err, err = q.finish());
                OCL_CHECK(err, err = q.enqueueMigrateMemObjects({buffer_output}, CL_MIGRATE_MEM_OBJECT_HOST, nullptr, &event_data_to_host));
                OCL_CHECK(err, err = q.finish());

                OCL_CHECK(err, err = event_data_to_fpga.getProfilingInfo<uint64_t>(CL_PROFILING_COMMAND_START, &nstimestart));
                OCL_CHECK(err, err = event_data_to_fpga.getProfilingInfo<uint64_t>(CL_PROFILING_COMMAND_END, &nstimeend));
                nstime_data_to_fpga_ocl += nstimeend - nstimestart;

                OCL_CHECK(err, err = event_kernel.getProfilingInfo<uint64_t>(CL_PROFILING_COMMAND_START, &nstimestart));
                OCL_CHECK(err, err = event_kernel.getProfilingInfo<uint64_t>(CL_PROFILING_COMMAND_END, &nstimeend));
                nstime_kernel_ocl += nstimeend - nstimestart;

                OCL_CHECK(err, err = event_data_to_host.getProfilingInfo<uint64_t>(CL_PROFILING_COMMAND_START, &nstimestart));
                OCL_CHECK(err, err = event_data_to_host.getProfilingInfo<uint64_t>(CL_PROFILING_COMMAND_END, &nstimeend));
                nstime_data_to_host_ocl += nstimeend - nstimestart;
            }
            // OPENCL HOST CODE AREA END

            end_time = std::chrono::high_resolution_clock::now();
            duration = std::chrono::duration<double>(end_time - start_time);
            nstime_cpu = std::chrono::duration_cast<std::chrono::nanoseconds>(duration).count();
        }
    }

    // CPU time: measured in host code, OCL time: measured using OpenCL profiling, all times in seconds
    std::cout << "app_name,kernel_input_data_size,kernel_output_data_size,iterations,time_cpu,data_to_fpga_time_ocl,kernel_time_ocl,data_to_host_time_ocl\n";
    std::cout << "cl_wide_mem_rw,"
              << vector_size_bytes * 2 << ","
              << vector_size_bytes << ","
              << iterations << ","
              << std::setprecision(std::numeric_limits<double>::digits10)
              << nstime_cpu / (double)1'000'000'000 << ","
              << nstime_data_to_fpga_ocl / (double)1'000'000'000 << ","
              << nstime_kernel_ocl / (double)1'000'000'000 << ","
              << nstime_data_to_host_ocl / (double)1'000'000'000 << "\n";

    // Compare the results of the Device to the simulation
    int match = 0;
    for (size_t i = 0; i < DATA_SIZE; i++) {
        if (source_hw_results[i] != source_sw_results[i]) {
            std::cout << "Error: Result mismatch" << std::endl;
            std::cout << "i = " << i << " CPU result = " << source_sw_results[i]
                      << " Device result = " << source_hw_results[i] << std::endl;
            match = 1;
            break;
        }
    }

    std::cout << "TEST " << (match ? "FAILED" : "PASSED") << std::endl;
    return (match ? EXIT_FAILURE : EXIT_SUCCESS);
}
