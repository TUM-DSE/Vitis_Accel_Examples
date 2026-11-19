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
#include "power_fpga.h"
#include <vector>
#include <cstdint>
#include <chrono>
#include <iomanip>

// DATA_SIZE should be multiple of 16 as Kernel Code is using int16 vector
// datatype
// to read the operands from Global Memory. So every read/write to global memory
// will read 16 integers value.
// As the other examples only read 1 int from memory at once, we use 16 times the
// data size of the other examples
#define DATA_SIZE (4 * 1024 * 1024) // 4 * 2 * sizeof(int) = 32 MB

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
    cl::Device selected_device;
    for (unsigned int i = 0; i < devices.size(); i++) {
        auto device = devices[i];
        selected_device = device;
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
    // Device-only buffers: the transfers below are explicit, so the buffers must not
    // be backed by host memory. The Funky backend force-adds CL_MEM_USE_HOST_PTR
    // whenever a host pointer reaches it, which would turn each transfer into a
    // host-side copy instead of a real device transfer.
    OCL_CHECK(err, cl::Buffer buffer_in1(context, CL_MEM_READ_ONLY, vector_size_bytes,
                                         nullptr, &err));
    OCL_CHECK(err, cl::Buffer buffer_in2(context, CL_MEM_READ_ONLY, vector_size_bytes,
                                         nullptr, &err));
    OCL_CHECK(err, cl::Buffer buffer_output(context, CL_MEM_WRITE_ONLY, vector_size_bytes,
                                            nullptr, &err));

    int size = DATA_SIZE;
    // Set the Kernel Arguments
    int nargs = 0;
    OCL_CHECK(err, err = krnl_vector_add.setArg(nargs++, buffer_in1));
    OCL_CHECK(err, err = krnl_vector_add.setArg(nargs++, buffer_in2));
    OCL_CHECK(err, err = krnl_vector_add.setArg(nargs++, buffer_output));
    OCL_CHECK(err, err = krnl_vector_add.setArg(nargs++, size));

    cl::Event event_kernel;
    cl::Event event_data_to_fpga;
    cl::Event event_data_to_fpga_2;
    cl::Event event_data_to_host;
    const int n_warmup = 0;
    const int n_reps = 10000;
    uint64_t nstimestart = 0;
    uint64_t nstimeend = 0;
    uint64_t time_kernel_ocl = 0;
    uint64_t time_data_to_xpu_ocl = 0;
    uint64_t time_data_to_host_ocl = 0;
    // Host-clock accumulator for data-transfer + kernel-execution time only: each interval below
    // is opened right before an OpenCL enqueue call and closed right after it (and any q.finish())
    // completes, so host-side work (loop bookkeeping) is never included.
    uint64_t time_xpu = 0;

    PowerMeter power;
    bool have_energy = power.open(selected_device());

    // This is required for proper time measurements in Proteus. We add it here
    // as well to have the same host code for Proteus and native.
    q.finish();

    // Idle baseline, taken in the same process state as the run that follows:
    // bitstream programmed, buffers allocated, nothing enqueued.
    const double idle_window_s = 20.0;
    double idle_w = 0.0;
    bool have_idle = have_energy && power.measure_idle(idle_window_s, &idle_w);
    if (have_energy && !have_idle) {
        std::cerr << "power: idle baseline measurement failed, reporting gross energy only\n";
    }

    // The sampled window covers the whole loop, idle time included, so the
    // matching denominator for average power is wall-clock time across the loop
    // rather than time_xpu (which excludes host-side bookkeeping).
    if (have_energy) power.start();
    auto t_loop_0 = std::chrono::steady_clock::now();

    for (int i = 0; i < n_warmup + n_reps; i++) {
        auto t_xpu_0 = std::chrono::high_resolution_clock::now();
        OCL_CHECK(err, err = q.enqueueWriteBuffer(buffer_in1, CL_FALSE, 0, vector_size_bytes,
                                                  source_in1.data(), nullptr, &event_data_to_fpga));
        OCL_CHECK(err, err = q.enqueueWriteBuffer(buffer_in2, CL_FALSE, 0, vector_size_bytes,
                                                  source_in2.data(), nullptr, &event_data_to_fpga_2));
        OCL_CHECK(err, err = q.finish());
        auto t_xpu_1 = std::chrono::high_resolution_clock::now();
        time_xpu += std::chrono::duration_cast<std::chrono::nanoseconds>(t_xpu_1 - t_xpu_0).count();

        auto t_xpu_2 = std::chrono::high_resolution_clock::now();
        OCL_CHECK(err, err = q.enqueueTask(krnl_vector_add, nullptr, &event_kernel));
        OCL_CHECK(err, err = q.finish());
        auto t_xpu_3 = std::chrono::high_resolution_clock::now();
        time_xpu += std::chrono::duration_cast<std::chrono::nanoseconds>(t_xpu_3 - t_xpu_2).count();

        auto t_xpu_4 = std::chrono::high_resolution_clock::now();
        OCL_CHECK(err, err = q.enqueueReadBuffer(buffer_output, CL_FALSE, 0, vector_size_bytes,
                                                 source_hw_results.data(), nullptr, &event_data_to_host));
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
    // OPENCL HOST CODE AREA END

    auto t_loop_1 = std::chrono::steady_clock::now();
    if (have_energy) have_energy = power.finish();
    uint64_t time_loop =
        std::chrono::duration_cast<std::chrono::nanoseconds>(t_loop_1 - t_loop_0).count();

    double ns_per_s = 1000000000;
    print_power_csv("cl_wide_mem_rw_power", vector_size_bytes * 2, vector_size_bytes, n_warmup,
                    n_reps, time_xpu / ns_per_s, time_data_to_xpu_ocl / ns_per_s,
                    time_kernel_ocl / ns_per_s, time_data_to_host_ocl / ns_per_s,
                    time_loop / ns_per_s, have_energy, power.energy_j, have_idle, idle_w,
                    idle_window_s);

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
