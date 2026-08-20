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
#include <vector>
#include <chrono>
#include <iomanip>

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
    // source_input is the immutable baseline; source_inout is the
    // CL_MEM_USE_HOST_PTR-backed buffer re-seeded from it every iteration
    // below, so the kernel's in-place "a[gid] += inc_value" always starts
    // from the same known state instead of silently drifting further with
    // every rep (the buffer is both the kernel's input and output).
    std::vector<int, aligned_allocator<int> > source_input(size);
    std::vector<int, aligned_allocator<int> > source_inout(size);
    std::vector<int, aligned_allocator<int> > source_sw_results(size);

    // Create the test data and Software Result
    for (int i = 0; i < size; i++) {
        source_input[i] = i;
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
    // Device-only buffers: the transfers below are explicit, so the buffers must not
    // be backed by host memory. The Funky backend force-adds CL_MEM_USE_HOST_PTR
    // whenever a host pointer reaches it, which would turn each transfer into a
    // host-side copy instead of a real device transfer.
    OCL_CHECK(err, cl::Buffer buffer_rw(context, CL_MEM_READ_WRITE, vector_size_bytes,
                                        nullptr, &err));

    OCL_CHECK(err, err = krnl_add.setArg(0, buffer_rw));
    OCL_CHECK(err, err = krnl_add.setArg(1, size));
    OCL_CHECK(err, err = krnl_add.setArg(2, inc_value));

    cl::Event event_kernel;
    cl::Event event_data_to_fpga;
    cl::Event event_data_to_host;
    const int n_warmup = 0;
    const int n_reps = 1000;
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

    for (int i = 0; i < n_warmup + n_reps; i++) {
        std::copy(source_input.begin(), source_input.end(), source_inout.begin());

        auto t_xpu_0 = std::chrono::high_resolution_clock::now();
        OCL_CHECK(err, err = q.enqueueWriteBuffer(buffer_rw, CL_FALSE, 0, vector_size_bytes,
                                                  source_inout.data(), nullptr, &event_data_to_fpga));
        OCL_CHECK(err, err = q.finish());
        auto t_xpu_1 = std::chrono::high_resolution_clock::now();
        time_xpu += std::chrono::duration_cast<std::chrono::nanoseconds>(t_xpu_1 - t_xpu_0).count();

        auto t_xpu_2 = std::chrono::high_resolution_clock::now();
        OCL_CHECK(err, err = q.enqueueTask(krnl_add, nullptr, &event_kernel));
        OCL_CHECK(err, err = q.finish());
        auto t_xpu_3 = std::chrono::high_resolution_clock::now();
        time_xpu += std::chrono::duration_cast<std::chrono::nanoseconds>(t_xpu_3 - t_xpu_2).count();

        auto t_xpu_4 = std::chrono::high_resolution_clock::now();
        OCL_CHECK(err, err = q.enqueueReadBuffer(buffer_rw, CL_FALSE, 0, vector_size_bytes,
                                                 source_inout.data(), nullptr, &event_data_to_host));
        OCL_CHECK(err, err = q.finish());
        auto t_xpu_5 = std::chrono::high_resolution_clock::now();
        time_xpu += std::chrono::duration_cast<std::chrono::nanoseconds>(t_xpu_5 - t_xpu_4).count();

        OCL_CHECK(err, err = event_data_to_fpga.getProfilingInfo<uint64_t>(CL_PROFILING_COMMAND_START, &nstimestart));
        OCL_CHECK(err, err = event_data_to_fpga.getProfilingInfo<uint64_t>(CL_PROFILING_COMMAND_END, &nstimeend));
        time_data_to_xpu_ocl += nstimeend - nstimestart;

        OCL_CHECK(err, err = event_kernel.getProfilingInfo<uint64_t>(CL_PROFILING_COMMAND_START, &nstimestart));
        OCL_CHECK(err, err = event_kernel.getProfilingInfo<uint64_t>(CL_PROFILING_COMMAND_END, &nstimeend));
        time_kernel_ocl += nstimeend - nstimestart;

        OCL_CHECK(err, err = event_data_to_host.getProfilingInfo<uint64_t>(CL_PROFILING_COMMAND_START, &nstimestart));
        OCL_CHECK(err, err = event_data_to_host.getProfilingInfo<uint64_t>(CL_PROFILING_COMMAND_END, &nstimeend));
        time_data_to_host_ocl += nstimeend - nstimestart;
    }
    // OPENCL HOST CODE AREA END

    double ns_per_s = 1000000000;
    std::cout << "app_name,in_size,out_size,reps_warmup,reps,time_xpu,time_data_to_xpu,time_kernel,time_data_to_host\n"
              << "cl_burst_rw,"
              << vector_size_bytes << ","
              << vector_size_bytes << ","
              << n_warmup << ","
              << n_reps << ","
              << time_xpu / ns_per_s << ","
              << time_data_to_xpu_ocl / ns_per_s << ","
              << time_kernel_ocl / ns_per_s << ","
              << time_data_to_host_ocl / ns_per_s
              << "\n";

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
