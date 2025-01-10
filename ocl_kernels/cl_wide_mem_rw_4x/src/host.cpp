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
#include <iomanip>

// DATA_SIZE should be multiple of 16 as Kernel Code is using int16 vector
// datatype
// to read the operands from Global Memory. So every read/write to global memory
// will read 16 integers value.
// As the other examples only read 1 int from memory at once, we use 16 times the
// data size of the other examples
#define DATA_SIZE (1024 * 1024) // * 2 * sizeof(int) = 8 MB

// Number of HBM PCs required
#define MAX_HBM_PC_COUNT 32
#define PC_NAME(n) n | XCL_MEM_TOPOLOGY
const int pc[MAX_HBM_PC_COUNT] = {
    PC_NAME(0),  PC_NAME(1),  PC_NAME(2),  PC_NAME(3),  PC_NAME(4),  PC_NAME(5),  PC_NAME(6),  PC_NAME(7),
    PC_NAME(8),  PC_NAME(9),  PC_NAME(10), PC_NAME(11), PC_NAME(12), PC_NAME(13), PC_NAME(14), PC_NAME(15),
    PC_NAME(16), PC_NAME(17), PC_NAME(18), PC_NAME(19), PC_NAME(20), PC_NAME(21), PC_NAME(22), PC_NAME(23),
    PC_NAME(24), PC_NAME(25), PC_NAME(26), PC_NAME(27), PC_NAME(28), PC_NAME(29), PC_NAME(30), PC_NAME(31)};

#define MAX_DDR_PC_COUNT 2
const int pc_ddr[MAX_DDR_PC_COUNT] = {
    XCL_MEM_DDR_BANK0, XCL_MEM_DDR_BANK1 //, XCL_MEM_DDR_BANK2, XCL_MEM_DDR_BANK3
};

auto constexpr num_cu = 4;
auto constexpr pc_per_cu = 4;

int main(int argc, char** argv) {
    if (argc != 3) {
        std::cout << "Usage: " << argv[0] << " <XCLBIN File> <Memory Type: 0 (HBM) or 1 (DDR)>" << std::endl;
        return EXIT_FAILURE;
    }

    std::string binaryFile = argv[1];
    std::string memoryType = argv[2];
    auto ddr_flag = std::stoi(memoryType);

    if(ddr_flag)
      std::cout << "DDR is selected. " << std::endl;
    else
      std::cout << "HBM is selected. " << std::endl;

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

    // For Allocating Buffer to specific Global Memory PC, user has to use
    // cl_mem_ext_ptr_t
    // and provide the PCs
    std::vector<cl_mem_ext_ptr_t> inBufExt1(num_cu);
    std::vector<cl_mem_ext_ptr_t> inBufExt2(num_cu);
    std::vector<cl_mem_ext_ptr_t> outBufExt(num_cu);

    for (int i = 0; i < num_cu; i++) {
        inBufExt1[i].obj = source_in1.data() + (i * DATA_SIZE);
        inBufExt1[i].param = 0;
        if(ddr_flag)
          inBufExt1[i].flags = pc_ddr[(i%MAX_DDR_PC_COUNT)];
        else
          inBufExt1[i].flags = pc[(i*(pc_per_cu))];

        inBufExt2[i].obj = source_in2.data() + (i * DATA_SIZE);
        inBufExt2[i].param = 0;
        if(ddr_flag)
          inBufExt2[i].flags = pc_ddr[(i%MAX_DDR_PC_COUNT)];
        else
          inBufExt2[i].flags = pc[(i*(pc_per_cu))+1];

        outBufExt[i].obj = source_hw_results.data() + (i * DATA_SIZE);
        outBufExt[i].param = 0;
        if(ddr_flag)
          outBufExt[i].flags = pc_ddr[(i%MAX_DDR_PC_COUNT)];
        else
          outBufExt[i].flags = pc[(i*(pc_per_cu))+2];
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
        OCL_CHECK(err, q = cl::CommandQueue(context, device, CL_QUEUE_PROFILING_ENABLE | CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE, &err));

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
        // std::cout << "buffer_in1: " << inBufExt1[i].flags << std::endl;
        OCL_CHECK(err, buffer_in1[i]    = cl::Buffer(context, CL_MEM_READ_ONLY | CL_MEM_EXT_PTR_XILINX | CL_MEM_USE_HOST_PTR,
                                                     vector_size_bytes, &inBufExt1[i], &err));
        // std::cout << "buffer_in2: " << inBufExt2[i].flags << std::endl;
        OCL_CHECK(err, buffer_in2[i]    = cl::Buffer(context, CL_MEM_READ_ONLY | CL_MEM_EXT_PTR_XILINX | CL_MEM_USE_HOST_PTR,
                                                     vector_size_bytes, &inBufExt2[i], &err));
        // std::cout << "buffer_in3: " << outBufExt[i].flags << std::endl;
        OCL_CHECK(err, buffer_output[i] = cl::Buffer(context, CL_MEM_WRITE_ONLY | CL_MEM_EXT_PTR_XILINX | CL_MEM_USE_HOST_PTR,
                                                     vector_size_bytes, &outBufExt[i], &err));
    }

    // Set the Kernel Arguments
    int size = DATA_SIZE;
    for (int i = 0; i < num_cu; i++) {
        int narg = 0;

        // Setting kernel arguments
        OCL_CHECK(err, err = krnls[i].setArg(narg++, buffer_in1[i]));
        OCL_CHECK(err, err = krnls[i].setArg(narg++, buffer_in2[i]));
        OCL_CHECK(err, err = krnls[i].setArg(narg++, buffer_output[i]));
        OCL_CHECK(err, err = krnls[i].setArg(narg++, size));
    }

    std::vector<cl::Event> event_kernel(num_cu);
    std::vector<cl::Event> event_data_to_fpga(num_cu);
    std::vector<cl::Event> event_data_to_host(num_cu);
    const int iterations = 1000;
    int64_t nstime_kernel_cpu = 0;
    int64_t nstime_data_to_fpga_cpu = 0;
    int64_t nstime_data_to_host_cpu = 0;
    uint64_t nstimestart = 0;
    uint64_t nstimeend = 0;
    uint64_t nstime_kernel_ocl = 0;
    uint64_t nstime_data_to_fpga_ocl = 0;
    uint64_t nstime_data_to_host_ocl = 0;

    std::chrono::duration<double> to_fpga_time(0);
    std::chrono::duration<double> kernel_time(0);
    std::chrono::duration<double> from_fpga_time(0);

    for (int i = 0; i < iterations; i++) {
        auto to_fpga_start = std::chrono::high_resolution_clock::now();
        for (int i = 0; i < num_cu; i++) {
          OCL_CHECK(err, err = q.enqueueMigrateMemObjects({buffer_in1[i], buffer_in2[i]}, 0 /* 0 means from host*/, nullptr, &event_data_to_fpga[i]));
          OCL_CHECK(err, err = q.finish());
        }
        auto to_fpga_end = std::chrono::high_resolution_clock::now();
        auto to_fgpa_duration = std::chrono::duration<double>(to_fpga_end - to_fpga_start);
        nstime_data_to_fpga_cpu += std::chrono::duration_cast<std::chrono::nanoseconds>(to_fgpa_duration).count();

        auto kernel_start = std::chrono::high_resolution_clock::now();
        for (int i = 0; i < num_cu; i++) {
          // Launch the kernel
          OCL_CHECK(err, err = q.enqueueTask(krnls[i], nullptr, &event_kernel[i]));
          OCL_CHECK(err, err = q.finish());
        }
        auto kernel_end = std::chrono::high_resolution_clock::now();
        auto kernel_duration = std::chrono::duration<double>(kernel_end - kernel_start);
        nstime_kernel_cpu += std::chrono::duration_cast<std::chrono::nanoseconds>(kernel_duration).count();

        auto from_fpga_start = std::chrono::high_resolution_clock::now();
        // Copy result from device global memory to host local memory
        for (int i = 0; i < num_cu; i++) {
          OCL_CHECK(err, err = q.enqueueMigrateMemObjects({buffer_output[i]}, CL_MIGRATE_MEM_OBJECT_HOST, nullptr, &event_data_to_host[i]));
          OCL_CHECK(err, err = q.finish());
        }
        auto from_fpga_end = std::chrono::high_resolution_clock::now();
        auto from_fpga_duration = std::chrono::duration<double>(from_fpga_end - from_fpga_start);
        nstime_data_to_host_cpu += std::chrono::duration_cast<std::chrono::nanoseconds>(from_fpga_duration).count();

        for (int i = 0; i < num_cu; i++) {
          OCL_CHECK(err, err = event_data_to_fpga[i].getProfilingInfo<uint64_t>(CL_PROFILING_COMMAND_START, &nstimestart));
          OCL_CHECK(err, err = event_data_to_fpga[i].getProfilingInfo<uint64_t>(CL_PROFILING_COMMAND_END, &nstimeend));
          nstime_data_to_fpga_ocl += nstimeend - nstimestart;
        }

        for (int i = 0; i < num_cu; i++) {
          OCL_CHECK(err, err = event_kernel[i].getProfilingInfo<uint64_t>(CL_PROFILING_COMMAND_START, &nstimestart));
          OCL_CHECK(err, err = event_kernel[i].getProfilingInfo<uint64_t>(CL_PROFILING_COMMAND_END, &nstimeend));
          nstime_kernel_ocl += nstimeend - nstimestart;
        }

        for (int i = 0; i < num_cu; i++) {
          OCL_CHECK(err, err = event_data_to_host[i].getProfilingInfo<uint64_t>(CL_PROFILING_COMMAND_START, &nstimestart));
          OCL_CHECK(err, err = event_data_to_host[i].getProfilingInfo<uint64_t>(CL_PROFILING_COMMAND_END, &nstimeend));
          nstime_data_to_host_ocl += nstimeend - nstimestart;
        }
    }

    // CPU time: measured in host code, OCL time: measured using OpenCL profiling, all times in seconds
    std::cout << "app_name,kernel_input_data_size,iterations,data_to_fpga_time_cpu,kernel_time_cpu,data_to_host_time_cpu,data_to_fpga_time_ocl,kernel_time_ocl,data_to_host_time_ocl\n";
    std::cout << "cl_wide_mem_rw_2x,"
              << vector_size_bytes * 2 * num_cu << ","
              << iterations << ","
              << std::setprecision(std::numeric_limits<double>::digits10)
              << nstime_data_to_fpga_cpu / (double)1'000'000'000 << ","
              << nstime_kernel_cpu / (double)1'000'000'000 << ","
              << nstime_data_to_host_cpu / (double)1'000'000'000 << ","
              << nstime_data_to_fpga_ocl / (double)1'000'000'000 << ","
              << nstime_kernel_ocl / (double)1'000'000'000 << ","
              << nstime_data_to_host_ocl / (double)1'000'000'000 << "\n";

    // std::cout << "data_to_fpga_cpu_time,kernel_cpu_time,data_to_host_cpu_time\n";
    // std::cout << "cl_wide_mem_rw_2x,"
    //           << to_fpga_time.count() / iterations << ","
    //           << kernel_time.count() / iterations << ","
    //           << from_fpga_time.count() / iterations << ","
    //           << std::endl;
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
