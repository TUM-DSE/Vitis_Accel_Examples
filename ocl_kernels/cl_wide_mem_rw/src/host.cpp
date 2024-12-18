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

// DATA_SIZE should be multiple of 16 as Kernel Code is using int16 vector
// datatype
// to read the operands from Global Memory. So every read/write to global memory
// will read 16 integers value.
// As the other examples only read 1 int from memory at once, we use 16 times the
// data size of the other examples
#define DATA_SIZE (1024 * 1024) // * 2 * sizeof(int) = 8 MB

int main(int argc, char** argv) {
    if (argc != 2) {
        std::cout << "Usage: " << argv[0] << " <XCLBIN File>" << std::endl;
        return EXIT_FAILURE;
    }

    std::string binaryFile = argv[1];

    // Allocate Memory in Host Memory
    size_t vector_size_bytes = sizeof(int) * DATA_SIZE;
    std::vector<int, aligned_allocator<int> > source_in1(DATA_SIZE);
    std::vector<int, aligned_allocator<int> > source_in2(DATA_SIZE);
    std::vector<int, aligned_allocator<int> > source_hw_results(DATA_SIZE);
    std::vector<int, aligned_allocator<int> > source_sw_results(DATA_SIZE);

    // Create the test data and Software Result
    for (int i = 0; i < DATA_SIZE; i++) {
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

    // Allocate Buffer in Global Memory
    OCL_CHECK(err, cl::Buffer buffer_in1(context, CL_MEM_USE_HOST_PTR | CL_MEM_READ_ONLY, vector_size_bytes,
                                         source_in1.data(), &err));
    OCL_CHECK(err, cl::Buffer buffer_in2(context, CL_MEM_USE_HOST_PTR | CL_MEM_READ_ONLY, vector_size_bytes,
                                         source_in2.data(), &err));
    OCL_CHECK(err, cl::Buffer buffer_output(context, CL_MEM_USE_HOST_PTR | CL_MEM_WRITE_ONLY, vector_size_bytes,
                                            source_hw_results.data(), &err));

    int size = DATA_SIZE;
    // Set the Kernel Arguments
    int nargs = 0;
    OCL_CHECK(err, err = krnl_vector_add.setArg(nargs++, buffer_in1));
    OCL_CHECK(err, err = krnl_vector_add.setArg(nargs++, buffer_in2));
    OCL_CHECK(err, err = krnl_vector_add.setArg(nargs++, buffer_output));
    OCL_CHECK(err, err = krnl_vector_add.setArg(nargs++, size));

    cl::Event event_kernel;
    cl::Event event_data_to_fpga;
    cl::Event event_data_to_host;
    const int iterations = 1000;
    uint64_t nstimestart = 0;
    uint64_t nstimeend = 0;
    uint64_t nstime_kernel = 0;
    uint64_t nstime_data_to_fpga = 0;
    uint64_t nstime_data_to_host = 0;

    for (int i = 0; i < iterations; i++) {
        OCL_CHECK(err, err = q.enqueueMigrateMemObjects({buffer_in1, buffer_in2}, 0 /* 0 means from host*/, nullptr, &event_data_to_fpga));
        OCL_CHECK(err, err = q.enqueueTask(krnl_vector_add, nullptr, &event_kernel));
        OCL_CHECK(err, err = q.enqueueMigrateMemObjects({buffer_output}, CL_MIGRATE_MEM_OBJECT_HOST, nullptr, &event_data_to_host));
        OCL_CHECK(err, err = q.finish());

        OCL_CHECK(err, err = event_data_to_fpga.getProfilingInfo<uint64_t>(CL_PROFILING_COMMAND_START, &nstimestart));
        OCL_CHECK(err, err = event_data_to_fpga.getProfilingInfo<uint64_t>(CL_PROFILING_COMMAND_END, &nstimeend));
        nstime_data_to_fpga += nstimeend - nstimestart;

        OCL_CHECK(err, err = event_kernel.getProfilingInfo<uint64_t>(CL_PROFILING_COMMAND_START, &nstimestart));
        OCL_CHECK(err, err = event_kernel.getProfilingInfo<uint64_t>(CL_PROFILING_COMMAND_END, &nstimeend));
        nstime_kernel += nstimeend - nstimestart;

        OCL_CHECK(err, err = event_data_to_host.getProfilingInfo<uint64_t>(CL_PROFILING_COMMAND_START, &nstimestart));
        OCL_CHECK(err, err = event_data_to_host.getProfilingInfo<uint64_t>(CL_PROFILING_COMMAND_END, &nstimeend));
        nstime_data_to_host += nstimeend - nstimestart;
    }

    std::cout << "app_name,kernel_input_data_size,iterations,data_to_fpga_avg_time,kernel_avg_time,data_to_host_avg_time\n";
    std::cout << "cl_wide_mem_rw,"
              << vector_size_bytes * 2 << ","
              << iterations << ","
              << nstime_data_to_fpga / iterations << ","
              << nstime_kernel / iterations << ","
              << nstime_data_to_host / iterations << "\n";

    // OPENCL HOST CODE AREA END

    // Compare the results of the Device to the simulation
    int match = 0;
    for (int i = 0; i < DATA_SIZE; i++) {
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
