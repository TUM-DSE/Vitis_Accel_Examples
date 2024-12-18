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

auto constexpr num_cu = 4;

int main(int argc, char** argv) {
    if (argc != 2) {
        std::cout << "Usage: " << argv[0] << " <XCLBIN File>" << std::endl;
        return EXIT_FAILURE;
    }

    std::string binaryFile = argv[1];

    // Allocate Memory in Host Memory
    int data_size = DATA_SIZE * num_cu;
    std::vector<int, aligned_allocator<int> > source_in1(data_size);
    std::vector<int, aligned_allocator<int> > source_in2(data_size);
    std::vector<int, aligned_allocator<int> > source_hw_results(data_size);
    std::vector<int, aligned_allocator<int> > source_sw_results(data_size);

    // Create the test data and Software Result
    for (int i = 0; i < data_size; i++) {
        source_in1[i] = i;
        source_in2[i] = i * i;
        source_sw_results[i] = i * i + i;
        source_hw_results[i] = 0;
    }

    // OPENCL HOST CODE AREA START
    cl_int err;
    cl::CommandQueue q;
    cl::Context context;
    // cl::Kernel krnl_vector_add;
    std::vector<cl::Kernel> krnls(num_cu);

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
        // OCL_CHECK(err, q = cl::CommandQueue(context, device, CL_QUEUE_PROFILING_ENABLE | CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE, &err));

        std::cout << "Trying to program device[" << i << "]: " << device.getInfo<CL_DEVICE_NAME>() << std::endl;
        cl::Program program(context, {device}, bins, nullptr, &err);
        if (err != CL_SUCCESS) {
            std::cout << "Failed to program device[" << i << "] with xclbin file!\n";
        } else {
            std::cout << "Device[" << i << "]: program successful!\n";
            // OCL_CHECK(err, krnl_vector_add = cl::Kernel(program, "vadd", &err));
            // Creating Kernel objects
            for (int i = 0; i < num_cu; i++) {
                OCL_CHECK(err, krnls[i] = cl::Kernel(program, "vadd", &err));
            }
            valid_device = true;
            break; // we break because we found a valid device
        }
    }
    if (!valid_device) {
        std::cout << "Failed to program any device found, exit!\n";
        exit(EXIT_FAILURE);
    }

    // Allocate Buffer in Global Memory
    size_t vector_size_bytes = sizeof(int) * DATA_SIZE;
    std::vector<cl::Buffer> buffer_in1(num_cu);
    std::vector<cl::Buffer> buffer_in2(num_cu);
    std::vector<cl::Buffer> buffer_output(num_cu);

    for (int i = 0; i < num_cu; i++) {
        OCL_CHECK(err, buffer_in1[i]    = cl::Buffer(context, CL_MEM_USE_HOST_PTR | CL_MEM_READ_ONLY, vector_size_bytes,
                                                     source_in1.data() + i * DATA_SIZE, &err));
        OCL_CHECK(err, buffer_in2[i]    = cl::Buffer(context, CL_MEM_USE_HOST_PTR | CL_MEM_READ_ONLY, vector_size_bytes,
                                                     source_in2.data() + i * DATA_SIZE, &err));
        OCL_CHECK(err, buffer_output[i] = cl::Buffer(context, CL_MEM_USE_HOST_PTR | CL_MEM_WRITE_ONLY, vector_size_bytes,
                                                     source_hw_results.data() + i * DATA_SIZE, &err));
    }

    // OCL_CHECK(err, cl::Buffer buffer_in1(context, CL_MEM_USE_HOST_PTR | CL_MEM_READ_ONLY, vector_size_bytes,
    //                                      source_in1.data(), &err));
    // OCL_CHECK(err, cl::Buffer buffer_in2(context, CL_MEM_USE_HOST_PTR | CL_MEM_READ_ONLY, vector_size_bytes,
    //                                      source_in2.data(), &err));
    // OCL_CHECK(err, cl::Buffer buffer_output(context, CL_MEM_USE_HOST_PTR | CL_MEM_WRITE_ONLY, vector_size_bytes,
    //                                         source_hw_results.data(), &err));


    // Set the Kernel Arguments
    // int size = DATA_SIZE;
    // int nargs = 0;
    // OCL_CHECK(err, err = krnl_vector_add.setArg(nargs++, buffer_in1));
    // OCL_CHECK(err, err = krnl_vector_add.setArg(nargs++, buffer_in2));
    // OCL_CHECK(err, err = krnl_vector_add.setArg(nargs++, buffer_output));
    // OCL_CHECK(err, err = krnl_vector_add.setArg(nargs++, size));
    std::cout << "test" << std::endl;
    int size = DATA_SIZE;
    for (int i = 0; i < num_cu; i++) {
        int narg = 0;

        // Setting kernel arguments
        OCL_CHECK(err, err = krnls[i].setArg(narg++, buffer_in1[i]));
        OCL_CHECK(err, err = krnls[i].setArg(narg++, buffer_in2[i]));
        OCL_CHECK(err, err = krnls[i].setArg(narg++, buffer_output[i]));
        OCL_CHECK(err, err = krnls[i].setArg(narg++, size));
    }
    std::cout << "test" << std::endl;

    std::vector<cl::Event> event_kernel(num_cu);
    std::vector<cl::Event> event_data_to_fpga(num_cu);
    std::vector<cl::Event> event_data_to_host(num_cu);
    const int iterations = 1000;
    uint64_t nstimestart = 0;
    uint64_t nstimeend = 0;
    uint64_t nstime_kernel = 0;
    uint64_t nstime_data_to_fpga = 0;
    uint64_t nstime_data_to_host = 0;

    std::chrono::duration<double> to_fpga_time(0);
    std::chrono::duration<double> kernel_time(0);
    std::chrono::duration<double> from_fpga_time(0);

    for (int i = 0; i < iterations; i++) {

        auto to_fpga_start = std::chrono::high_resolution_clock::now();
        for (int i = 0; i < num_cu; i++) {
          OCL_CHECK(err, err = q.enqueueMigrateMemObjects({buffer_in1[i], buffer_in2[i]}, 0 /* 0 means from host*/, nullptr, &event_data_to_fpga[i]));
        }
        // OCL_CHECK(err, err = q.enqueueMigrateMemObjects({buffer_in1, buffer_in2}, 0 /* 0 means from host*/, nullptr, &event_data_to_fpga));
        OCL_CHECK(err, err = q.finish());
        auto to_fpga_end = std::chrono::high_resolution_clock::now();

        auto kernel_start = std::chrono::high_resolution_clock::now();
        for (int i = 0; i < num_cu; i++) {
          // Launch the kernel
          OCL_CHECK(err, err = q.enqueueTask(krnls[i], nullptr, &event_kernel[i]));
        }
        // OCL_CHECK(err, err = q.enqueueTask(krnl_vector_add, nullptr, &event_kernel));
        OCL_CHECK(err, err = q.finish());
        auto kernel_end = std::chrono::high_resolution_clock::now();

        auto from_fpga_start = std::chrono::high_resolution_clock::now();
        // Copy result from device global memory to host local memory
        for (int i = 0; i < num_cu; i++) {
          OCL_CHECK(err, err = q.enqueueMigrateMemObjects({buffer_output[i]}, CL_MIGRATE_MEM_OBJECT_HOST, nullptr, &event_data_to_host[i]));
        }
        // OCL_CHECK(err, err = q.enqueueMigrateMemObjects({buffer_output}, CL_MIGRATE_MEM_OBJECT_HOST, nullptr, &event_data_to_host));
        OCL_CHECK(err, err = q.finish());
        auto from_fpga_end = std::chrono::high_resolution_clock::now();


        for (int i = 0; i < num_cu; i++) {
          OCL_CHECK(err, err = event_data_to_fpga[i].getProfilingInfo<uint64_t>(CL_PROFILING_COMMAND_START, &nstimestart));
          OCL_CHECK(err, err = event_data_to_fpga[i].getProfilingInfo<uint64_t>(CL_PROFILING_COMMAND_END, &nstimeend));
          nstime_data_to_fpga += nstimeend - nstimestart;
        }

        for (int i = 0; i < num_cu; i++) {
          OCL_CHECK(err, err = event_kernel[i].getProfilingInfo<uint64_t>(CL_PROFILING_COMMAND_START, &nstimestart));
          OCL_CHECK(err, err = event_kernel[i].getProfilingInfo<uint64_t>(CL_PROFILING_COMMAND_END, &nstimeend));
          nstime_kernel += nstimeend - nstimestart;
        }

        for (int i = 0; i < num_cu; i++) {
          OCL_CHECK(err, err = event_data_to_host[i].getProfilingInfo<uint64_t>(CL_PROFILING_COMMAND_START, &nstimestart));
          OCL_CHECK(err, err = event_data_to_host[i].getProfilingInfo<uint64_t>(CL_PROFILING_COMMAND_END, &nstimeend));
          nstime_data_to_host += nstimeend - nstimestart;
        }

        to_fpga_time += std::chrono::duration<double>(to_fpga_end - to_fpga_start);
        kernel_time += std::chrono::duration<double>(kernel_end - kernel_start);
        from_fpga_time += std::chrono::duration<double>(from_fpga_end - from_fpga_start);

    }

    std::cout << "app_name,kernel_data_size,iterations,data_to_fpga_avg_time,kernel_avg_time,data_to_host_avg_time\n";
    std::cout << "cl_wide_mem_rw_4x,"
              << vector_size_bytes * 3 * num_cu << ","
              << iterations << ","
              << nstime_data_to_fpga / iterations / num_cu << ","
              << nstime_kernel / iterations / num_cu  << ","
              << nstime_data_to_host / iterations / num_cu << "\n";

    std::cout << "data_to_fpga_cpu_time,kernel_cpu_time,data_to_host_cpu_time\n";
    std::cout << "cl_wide_mem_rw_4x," 
              << to_fpga_time.count() / iterations << "," 
              << kernel_time.count() / iterations << "," 
              << from_fpga_time.count() / iterations << "," 
              << std::endl;

    // OPENCL HOST CODE AREA END

    // Compare the results of the Device to the simulation
    int match = 0;
    for (int i = 0; i < data_size; i++) {
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
