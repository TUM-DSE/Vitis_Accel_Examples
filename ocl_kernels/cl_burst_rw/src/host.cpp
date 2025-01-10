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
#include <vector>
#include <chrono>

#define DATA_SIZE (128 * 1024) // * sizeof(int) = 512 KB
#define INCR_VALUE 10
// define internal buffer max size
#define BURSTBUFFERSIZE 256

int main(int argc, char** argv) {
    if (argc != 2) {
        std::cout << "Usage: " << argv[0] << " <XCLBIN File>" << std::endl;
        return EXIT_FAILURE;
    }

    std::string binaryFile = argv[1];

    int size = DATA_SIZE;
    int inc_value = INCR_VALUE;
    cl_int err;
    cl::CommandQueue q;
    cl::Context context;
    cl::Kernel krnl_add;
    // Allocate Memory in Host Memory
    size_t vector_size_bytes = sizeof(int) * size;
    std::vector<int, aligned_allocator<int> > source_inout(size);
    std::vector<int, aligned_allocator<int> > source_sw_results(size);

    // Create the test data and Software Result
    for (int i = 0; i < size; i++) {
        source_inout[i] = i;
        source_sw_results[i] = i + inc_value;
    }

    // OPENCL HOST CODE AREA START
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
            OCL_CHECK(err, krnl_add = cl::Kernel(program, "vadd", &err));
            valid_device = true;
            break; // we break because we found a valid device
        }
    }
    if (!valid_device) {
        std::cout << "Failed to program any device found, exit!\n";
        exit(EXIT_FAILURE);
    }

    // Allocate Buffer in Global Memory
    OCL_CHECK(err, cl::Buffer buffer_rw(context, CL_MEM_READ_WRITE | CL_MEM_USE_HOST_PTR, vector_size_bytes,
                                        source_inout.data(), &err));

    OCL_CHECK(err, err = krnl_add.setArg(0, buffer_rw));
    OCL_CHECK(err, err = krnl_add.setArg(1, size));
    OCL_CHECK(err, err = krnl_add.setArg(2, inc_value));

    // cl::Event event_kernel;
    // cl::Event event_data_to_fpga;
    // cl::Event event_data_to_host;
    const int iterations = 1000;
    // uint64_t nstimestart = 0;
    // uint64_t nstimeend = 0;
    // uint64_t nstime_kernel = 0;
    // uint64_t nstime_data_to_fpga = 0;
    // uint64_t nstime_data_to_host = 0;
    std::chrono::high_resolution_clock::time_point start_time, end_time;
    std::chrono::duration<double> duration;
    int64_t nstime_kernel = 0;
    int64_t nstime_data_to_fpga = 0;
    int64_t nstime_data_to_host = 0;

    for (int i = 0; i < iterations; i++) {
        start_time = std::chrono::high_resolution_clock::now();
        OCL_CHECK(err, err = q.enqueueMigrateMemObjects({buffer_rw}, 0 /* 0 means from host*/));
        q.finish();
        end_time = std::chrono::high_resolution_clock::now();
        duration = std::chrono::duration<double>(end_time - start_time);
        nstime_data_to_fpga += std::chrono::duration_cast<std::chrono::nanoseconds>(duration).count();

        start_time = std::chrono::high_resolution_clock::now();
        OCL_CHECK(err, err = q.enqueueTask(krnl_add));
        q.finish();
        end_time = std::chrono::high_resolution_clock::now();
        duration = std::chrono::duration<double>(end_time - start_time);
        nstime_kernel += std::chrono::duration_cast<std::chrono::nanoseconds>(duration).count();

        start_time = std::chrono::high_resolution_clock::now();
        OCL_CHECK(err, err = q.enqueueMigrateMemObjects({buffer_rw}, CL_MIGRATE_MEM_OBJECT_HOST));
        q.finish();
        end_time = std::chrono::high_resolution_clock::now();
        duration = std::chrono::duration<double>(end_time - start_time);
        nstime_data_to_host += std::chrono::duration_cast<std::chrono::nanoseconds>(duration).count();

        // OCL_CHECK(err, err = q.enqueueMigrateMemObjects({buffer_rw}, 0 /* 0 means from host*/, nullptr, &event_data_to_fpga));
        // OCL_CHECK(err, err = q.enqueueTask(krnl_add, nullptr, &event_kernel));
        // OCL_CHECK(err, err = q.enqueueMigrateMemObjects({buffer_rw}, CL_MIGRATE_MEM_OBJECT_HOST, nullptr, &event_data_to_host));
        // OCL_CHECK(err, err = q.finish());

        // OCL_CHECK(err, err = event_data_to_fpga.getProfilingInfo<uint64_t>(CL_PROFILING_COMMAND_START, &nstimestart));
        // OCL_CHECK(err, err = event_data_to_fpga.getProfilingInfo<uint64_t>(CL_PROFILING_COMMAND_END, &nstimeend));
        // nstime_data_to_fpga += nstimeend - nstimestart;

        // OCL_CHECK(err, err = event_kernel.getProfilingInfo<uint64_t>(CL_PROFILING_COMMAND_START, &nstimestart));
        // OCL_CHECK(err, err = event_kernel.getProfilingInfo<uint64_t>(CL_PROFILING_COMMAND_END, &nstimeend));
        // nstime_kernel += nstimeend - nstimestart;

        // OCL_CHECK(err, err = event_data_to_host.getProfilingInfo<uint64_t>(CL_PROFILING_COMMAND_START, &nstimestart));
        // OCL_CHECK(err, err = event_data_to_host.getProfilingInfo<uint64_t>(CL_PROFILING_COMMAND_END, &nstimeend));
        // nstime_data_to_host += nstimeend - nstimestart;
    }
    // OPENCL HOST CODE AREA END

    std::cout << "app_name,kernel_input_data_size,iterations,data_to_fpga_time,kernel_time,data_to_host_time\n";
    std::cout << "cl_burst_rw,"
              << vector_size_bytes << ","
              << iterations << ","
              << nstime_data_to_fpga << ","
              << nstime_kernel << ","
              << nstime_data_to_host << "\n";

    // OPENCL HOST CODE AREA END

    // Compare the results of the Device to the simulation
    int match = 0;
    for (int i = 0; i < size; i++) {
        if (source_inout[i] != source_sw_results[i]) {
            std::cout << "Error: Result mismatch" << std::endl;
            std::cout << "i = " << i << " CPU result = " << source_sw_results[i]
                      << " Device result = " << source_inout[i] << std::endl;
            match = 1;
            break;
        } else {
            // std::cout << source_inout[i] << " ";
            // if (((i + 1) % 16) == 0) std::cout << std::endl;
        }
    }

    std::cout << "TEST " << (match ? "FAILED" : "PASSED") << std::endl;
    return (match ? EXIT_FAILURE : EXIT_SUCCESS);
}
