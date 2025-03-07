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
#include <CL/cl2.hpp>
#include <climits>
#include <cmath>
#include <cassert>
#include <cstdint>
#include <cstdlib>
#include <vector>
#include <iomanip>

constexpr size_t MiB = 1024 * 1024;
constexpr size_t ALIGNMENT = 4096;
// Artificial limit for available FPGA memory, overridden by -m option
constexpr size_t MEM_LIMIT = SIZE_MAX;
// DATA_SIZE in bytes should be multiple of 64 as kernel code is using int16
// vector datatype to read the operands from global memory 16 ints at a time. We
// aim for the 2 input buffers and the output buffer to not fit into MEM_LIMIT
// to simulate memory over-subscription. Overridden by -s option.
constexpr size_t DATA_SIZE = 32 * MiB; // 3 buffers (2 input, 1 output) of this size * sizeof(int)
// Whether data transfer and kernel execution should be overlapped where
// possible by having 2 chunks instead of 1 per buffer in FPGA memory.
// Overridden by -o option.
constexpr bool OPTIMIZED = false;

int main(int argc, char** argv) {
    if (argc < 2) {
      std::cout << "Usage: " << argv[0] << " <XCLBIN File>\n"
                << "  [-m <size>] On-FPGA memory limit in MiB. Default: " << MEM_LIMIT / MiB << "\n"
                << "  [-s <size>] Size per buffer in MiB. The application uses 3 buffers. Default: " << DATA_SIZE / MiB << "\n"
                << "  [-o]        Enable over-subscription optimizations (overlapping data transfer and kernel execution)\n\n"
                << "Memory over-subscription is active when memory limit < 3 * buffer size\n";
      return EXIT_FAILURE;
    }

    size_t mem_limit = MEM_LIMIT;
    size_t data_size = DATA_SIZE;
    bool optimized = OPTIMIZED;
    std::string binaryFile = argv[1];

    for (int i = 2; i < argc; i++) {
        if (strcmp("-m", argv[i]) == 0) {
            mem_limit = std::stol(argv[i + 1]) * MiB;
        } else if (strcmp("-s", argv[i]) == 0) {
            data_size = std::stol(argv[i + 1]) * MiB;
        } else if (strcmp("-o", argv[i]) == 0) {
            optimized = true;
        }
    }

    assert(data_size % ALIGNMENT == 0);
    bool oversub = 3 * data_size > mem_limit;

    std::cout << "Memory limit: " << mem_limit / MiB << " MiB\n";
    std::cout << "Buffer size:  " << data_size / MiB << " MiB, 3 buffers in total\n";
    if (oversub) {
        std::cout << "=> Memory over-subscription enabled\n";
        std::cout << "   Over-subscription optimizations " << (optimized ? "enabled" : "disabled") << "\n";
    } else {
        std::cout << "=> Memory over-subscription disabled\n";
    }

    // Allocate Memory in Host Memory
    size_t vector_size = data_size / sizeof(int);
    std::vector<int, aligned_allocator<int> > source_in1(vector_size);
    std::vector<int, aligned_allocator<int> > source_in2(vector_size);
    std::vector<int, aligned_allocator<int> > source_hw_results(vector_size);
    std::vector<int, aligned_allocator<int> > source_sw_results(vector_size);

    // Create the test data and Software Result
    for (size_t i = 0; i < vector_size; i++) {
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
        // OCL_CHECK(err, q = cl::CommandQueue(context, device, CL_QUEUE_PROFILING_ENABLE, &err));
        OCL_CHECK(err, q = cl::CommandQueue(context, device, CL_QUEUE_PROFILING_ENABLE | CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE, &err));

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

    size_t chunk_size = data_size;
    if (oversub) {
      // Without optimizations, 3 chunks have to fit into mem_limit. With
      // optimizations, 6 chunks have to fit. Also round down to closest
      // multiple of ALIGNMENT.
      if (optimized) {
        chunk_size = (mem_limit / 6) & ~(ALIGNMENT - 1);
      } else {
        chunk_size = (mem_limit / 3) & ~(ALIGNMENT - 1);
      }
    }

    // The size parameter of the kernel is type int
    assert(chunk_size <= INT_MAX);

    size_t num_chunks = std::ceil(data_size / (double)chunk_size);
    size_t last_chunk_size = data_size % chunk_size;
    if (last_chunk_size == 0) {
        last_chunk_size = chunk_size;
    }

    std::cout << "memory limit:      " << mem_limit << " B\n";
    std::cout << "3 * buffer size:   " << 3 * data_size << " B\n";
    std::cout << "chunks per buffer: " << num_chunks << "\n";
    std::cout << "chunk size:        " << chunk_size << " B\n";
    std::cout << "last chunk size:   " << last_chunk_size << " B\n";

    cl::Event event_kernel;
    cl::Event event_data_to_fpga;
    cl::Event event_data_to_host;
    const int iterations = 1;
    std::chrono::high_resolution_clock::time_point start_time, end_time;
    std::chrono::duration<double> duration;
    int64_t nstime_cpu = 0;
    uint64_t nstimestart = 0;
    uint64_t nstimeend = 0;
    uint64_t nstime_kernel_ocl = 0;
    uint64_t nstime_data_to_fpga_ocl = 0;
    uint64_t nstime_data_to_host_ocl = 0;

    if (oversub && optimized) {
        for (int i = 0; i < iterations; i++) {
            std::vector<cl::Event> kernel_events(2);
            std::vector<cl::Event> to_host_events(2);
            cl::Buffer buffer_in1[2];
            cl::Buffer buffer_in2[2];
            cl::Buffer buffer_out[2];

            for (size_t i = 0; i < num_chunks; i++) {
                int flag = i % 2;

                if (i >= 2) {
                    OCL_CHECK(err, err = to_host_events[flag].wait());
                }

                size_t cur_chunk_size = chunk_size;
                if (i == num_chunks - 1) {
                    cur_chunk_size = last_chunk_size;
                }
                auto buf_offset = i * chunk_size / sizeof(int);

                OCL_CHECK(err, buffer_in1[flag] = cl::Buffer(context, CL_MEM_USE_HOST_PTR | CL_MEM_READ_ONLY, cur_chunk_size, source_in1.data() + buf_offset, &err));
                OCL_CHECK(err, buffer_in2[flag] = cl::Buffer(context, CL_MEM_USE_HOST_PTR | CL_MEM_READ_ONLY, cur_chunk_size, source_in2.data() + buf_offset, &err));
                OCL_CHECK(err, buffer_out[flag] = cl::Buffer(context, CL_MEM_USE_HOST_PTR | CL_MEM_WRITE_ONLY, cur_chunk_size, source_hw_results.data() + buf_offset, &err));

                std::vector<cl::Event> to_fpga_event(1);

                int nargs = 0;
                OCL_CHECK(err, err = krnl_vector_add.setArg(nargs++, buffer_in1[flag]));
                OCL_CHECK(err, err = krnl_vector_add.setArg(nargs++, buffer_in2[flag]));
                OCL_CHECK(err, err = krnl_vector_add.setArg(nargs++, buffer_out[flag]));
                OCL_CHECK(err, err = krnl_vector_add.setArg(nargs++, (int)(cur_chunk_size / sizeof(int))));

                start_time = std::chrono::high_resolution_clock::now();

                OCL_CHECK(err, err = q.enqueueMigrateMemObjects({buffer_in1[flag], buffer_in2[flag]}, 0 /* 0 means from host*/, nullptr, &to_fpga_event[0]));

                std::vector<cl::Event> wait_kernel{to_fpga_event[0]};
                OCL_CHECK(err, err = q.enqueueTask(krnl_vector_add, &wait_kernel, &kernel_events[flag]));

                std::vector<cl::Event> wait_to_host{kernel_events[flag]};
                OCL_CHECK(err, err = q.enqueueMigrateMemObjects({buffer_out[flag]}, CL_MIGRATE_MEM_OBJECT_HOST, &wait_to_host, &to_host_events[flag]));

                OCL_CHECK(err, err = to_host_events[flag].wait());

                end_time = std::chrono::high_resolution_clock::now();
                duration = std::chrono::duration<double>(end_time - start_time);
                nstime_cpu += std::chrono::duration_cast<std::chrono::nanoseconds>(duration).count();
            }
        }
    } else {
        // No over-subscription or unoptimized over-subscription
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
                OCL_CHECK(err, err = krnl_vector_add.setArg(nargs++, (int)(cur_chunk_size / sizeof(int))));

                // This is required for proper time measurements in Proteus. We add it here
                // as well to have the same host code for Proteus and native.
                q.finish();

                start_time = std::chrono::high_resolution_clock::now();

                OCL_CHECK(err, err = q.enqueueMigrateMemObjects({buffer_in1, buffer_in2}, 0 /* 0 means from host*/, nullptr, &event_data_to_fpga));
                std::vector<cl::Event> wait_kernel{event_data_to_fpga};
                OCL_CHECK(err, err = q.enqueueTask(krnl_vector_add, &wait_kernel, &event_kernel));
                std::vector<cl::Event> wait_to_host{event_kernel};
                OCL_CHECK(err, err = q.enqueueMigrateMemObjects({buffer_output}, CL_MIGRATE_MEM_OBJECT_HOST, &wait_to_host, &event_data_to_host));
                OCL_CHECK(err, err = q.finish());

                end_time = std::chrono::high_resolution_clock::now();
                duration = std::chrono::duration<double>(end_time - start_time);
                nstime_cpu += std::chrono::duration_cast<std::chrono::nanoseconds>(duration).count();

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
        }
    }
    // OPENCL HOST CODE AREA END

    // CPU time: measured in host code, OCL time: measured using OpenCL profiling, all times in seconds
    std::cout << "app_name,kernel_input_data_size,kernel_output_data_size,iterations,time_cpu,data_to_fpga_time_ocl,kernel_time_ocl,data_to_host_time_ocl\n";
    std::cout << "cl_wide_mem_rw,"
              << data_size * 2 << ","
              << data_size << ","
              << iterations << ","
              << std::setprecision(std::numeric_limits<double>::digits10)
              << nstime_cpu / (double)1'000'000'000 << ","
              << nstime_data_to_fpga_ocl / (double)1'000'000'000 << ","
              << nstime_kernel_ocl / (double)1'000'000'000 << ","
              << nstime_data_to_host_ocl / (double)1'000'000'000 << "\n";

    // Compare the results of the Device to the simulation
    int match = 0;
    for (size_t i = 0; i < vector_size; i++) {
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
