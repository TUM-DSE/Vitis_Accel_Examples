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

/*
   Shift Register

   This example demonstrates how to perform a shift register operation to
   implement a Finite Impulse Response(FIR) filter.

   NOTE: See the fir.cl file for additional information.
  */
#include "xcl2.hpp"

#include <algorithm>
#include <random>
#include <string>
#include <vector>
#include <chrono>
#include <iomanip>

#define SIGNAL_SIZE (128 * 1024) // * sizeof(int) = 512 KB
#define SIGNAL_SIZE_IN_EMU 1024

using std::default_random_engine;
using std::inner_product;
using std::string;
using std::uniform_int_distribution;
using std::vector;

// helping functions
void fir_sw(vector<int, aligned_allocator<int> >& output,
            const vector<int, aligned_allocator<int> >& signal,
            const vector<int, aligned_allocator<int> >& coeff);

void verify(const vector<int, aligned_allocator<int> >& gold, const vector<int, aligned_allocator<int> >& out);
uint64_t get_duration_ns(const cl::Event& event);
void print_summary(std::string k1, std::string k2, uint64_t t1, uint64_t t2, int iterations);
int gen_random();

int main(int argc, char** argv) {
    if (argc != 2) {
        std::cout << "Usage: " << argv[0] << " <XCLBIN File>" << std::endl;
        return EXIT_FAILURE;
    }

    std::string binaryFile = argv[1];
    int signal_size = xcl::is_emulation() ? SIGNAL_SIZE_IN_EMU : SIGNAL_SIZE;
    vector<int, aligned_allocator<int> > signal(signal_size);
    vector<int, aligned_allocator<int> > out(signal_size);
    vector<int, aligned_allocator<int> > coeff = {{53, 0, -91, 0, 313, 500, 313, 0, -91, 0, 53}};
    vector<int, aligned_allocator<int> > gold(signal_size, 0);
    generate(begin(signal), end(signal), gen_random);

    fir_sw(gold, signal, coeff);

    size_t size_in_bytes = signal_size * sizeof(int);
    size_t coeff_size_in_bytes = coeff.size() * sizeof(int);
    cl_int err;
    cl::CommandQueue q;
    cl::Context context;
    cl::Program program;

    // Initialize OpenCL context and load xclbin binary
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

    // Allocate Buffer in Global Memory
    OCL_CHECK(err, cl::Buffer buffer_signal_A(context, CL_MEM_USE_HOST_PTR | CL_MEM_READ_ONLY, size_in_bytes,
                                              signal.data(), &err));
    OCL_CHECK(err, cl::Buffer buffer_coeff_A(context, CL_MEM_USE_HOST_PTR | CL_MEM_READ_ONLY, coeff_size_in_bytes,
                                             coeff.data(), &err));
    OCL_CHECK(err, cl::Buffer buffer_output_A(context, CL_MEM_USE_HOST_PTR | CL_MEM_WRITE_ONLY, size_in_bytes,
                                              out.data(), &err));
    OCL_CHECK(err, cl::Buffer buffer_signal_B(context, CL_MEM_USE_HOST_PTR | CL_MEM_READ_ONLY, size_in_bytes,
                                              signal.data(), &err));
    OCL_CHECK(err, cl::Buffer buffer_coeff_B(context, CL_MEM_USE_HOST_PTR | CL_MEM_READ_ONLY, coeff_size_in_bytes,
                                             coeff.data(), &err));
    OCL_CHECK(err, cl::Buffer buffer_output_B(context, CL_MEM_USE_HOST_PTR | CL_MEM_WRITE_ONLY, size_in_bytes,
                                              out.data(), &err));

    // Creating Naive Kernel Object and setting args
    OCL_CHECK(err, cl::Kernel fir_naive_kernel(program, "fir_naive", &err));

    OCL_CHECK(err, err = fir_naive_kernel.setArg(0, buffer_output_A));
    OCL_CHECK(err, err = fir_naive_kernel.setArg(1, buffer_signal_A));
    OCL_CHECK(err, err = fir_naive_kernel.setArg(2, buffer_coeff_A));
    OCL_CHECK(err, err = fir_naive_kernel.setArg(3, signal_size));

    // cl::Event event_kernel;
    // cl::Event event_data_to_fpga;
    // cl::Event event_data_to_host;
    int iterations = xcl::is_emulation() ? 2 : 1000;
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

    // Running naive kernel iterations times
    for (int i = 0; i < iterations / 2; i++) {
        start_time = std::chrono::high_resolution_clock::now();
        OCL_CHECK(err, err = q.enqueueMigrateMemObjects({buffer_signal_A, buffer_coeff_A}, 0 /* 0 means from host*/));
        q.finish();
        end_time = std::chrono::high_resolution_clock::now();
        duration = std::chrono::duration<double>(end_time - start_time);
        nstime_data_to_fpga += std::chrono::duration_cast<std::chrono::nanoseconds>(duration).count();

        start_time = std::chrono::high_resolution_clock::now();
        OCL_CHECK(err, err = q.enqueueTask(fir_naive_kernel));
        q.finish();
        end_time = std::chrono::high_resolution_clock::now();
        duration = std::chrono::duration<double>(end_time - start_time);
        nstime_kernel += std::chrono::duration_cast<std::chrono::nanoseconds>(duration).count();

        start_time = std::chrono::high_resolution_clock::now();
        OCL_CHECK(err, err = q.enqueueMigrateMemObjects({buffer_output_A}, CL_MIGRATE_MEM_OBJECT_HOST));
        q.finish();
        end_time = std::chrono::high_resolution_clock::now();
        duration = std::chrono::duration<double>(end_time - start_time);
        nstime_data_to_host += std::chrono::duration_cast<std::chrono::nanoseconds>(duration).count();

        // OCL_CHECK(err, err = q.enqueueMigrateMemObjects({buffer_signal_A, buffer_coeff_A}, 0 /* 0 means from host*/, nullptr, &event_data_to_fpga));
        // OCL_CHECK(err, err = q.enqueueTask(fir_naive_kernel, nullptr, &event_kernel));
        // OCL_CHECK(err, err = q.enqueueMigrateMemObjects({buffer_output_A}, CL_MIGRATE_MEM_OBJECT_HOST, nullptr, &event_data_to_host));
        // q.finish();

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

    verify(gold, out);

    // Creating FIR Shift Register Kernel object and setting args
    OCL_CHECK(err, cl::Kernel fir_sr_kernel(program, "fir_shift_register", &err));

    OCL_CHECK(err, err = fir_sr_kernel.setArg(0, buffer_output_B));
    OCL_CHECK(err, err = fir_sr_kernel.setArg(1, buffer_signal_B));
    OCL_CHECK(err, err = fir_sr_kernel.setArg(2, buffer_coeff_B));
    OCL_CHECK(err, err = fir_sr_kernel.setArg(3, signal_size));

    // Running Shift Register FIR iterations times
    for (int i = 0; i < iterations / 2; i++) {
        start_time = std::chrono::high_resolution_clock::now();
        OCL_CHECK(err, err = q.enqueueMigrateMemObjects({buffer_signal_B, buffer_coeff_B}, 0 /* 0 means from host*/));
        q.finish();
        end_time = std::chrono::high_resolution_clock::now();
        duration = std::chrono::duration<double>(end_time - start_time);
        nstime_data_to_fpga += std::chrono::duration_cast<std::chrono::nanoseconds>(duration).count();

        start_time = std::chrono::high_resolution_clock::now();
        OCL_CHECK(err, err = q.enqueueTask(fir_sr_kernel));
        q.finish();
        end_time = std::chrono::high_resolution_clock::now();
        duration = std::chrono::duration<double>(end_time - start_time);
        nstime_kernel += std::chrono::duration_cast<std::chrono::nanoseconds>(duration).count();

        start_time = std::chrono::high_resolution_clock::now();
        OCL_CHECK(err, err = q.enqueueMigrateMemObjects({buffer_output_B}, CL_MIGRATE_MEM_OBJECT_HOST));
        q.finish();
        end_time = std::chrono::high_resolution_clock::now();
        duration = std::chrono::duration<double>(end_time - start_time);
        nstime_data_to_host += std::chrono::duration_cast<std::chrono::nanoseconds>(duration).count();

        // OCL_CHECK(err, err = q.enqueueMigrateMemObjects({buffer_signal_B, buffer_coeff_B}, 0 /* 0 means from host*/, nullptr, &event_data_to_fpga));
        // OCL_CHECK(err, err = q.enqueueTask(fir_sr_kernel, nullptr, &event_kernel));
        // OCL_CHECK(err, err = q.enqueueMigrateMemObjects({buffer_output_B}, CL_MIGRATE_MEM_OBJECT_HOST, nullptr, &event_data_to_host));
        // q.finish();

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

    verify(gold, out);

    printf("Example Testdata Signal_Length=%u for %d iteration\n", signal_size, iterations);
    // print_summary("fir_naive", "fir_shift_register", fir_naive_time, fir_sr_time, iterations);

    std::cout << "app_name,kernel_input_data_size,iterations,data_to_fpga_time,kernel_time,data_to_host_time\n";
    std::cout << "cl_shift_register,"
              << size_in_bytes + coeff_size_in_bytes << ","
              << iterations << ","
              << std::setprecision(std::numeric_limits<double>::digits10)
              << nstime_data_to_fpga / (double)1'000'000'000 << ","
              << nstime_kernel / (double)1'000'000'000 << ","
              << nstime_data_to_host / (double)1'000'000'000 << "\n";

    printf("TEST PASSED\n");
    return EXIT_SUCCESS;
}

// Finite Impulse Response Filter
void fir_sw(vector<int, aligned_allocator<int> >& output,
            const vector<int, aligned_allocator<int> >& signal,
            const vector<int, aligned_allocator<int> >& coeff) {
    auto out_iter = begin(output);
    auto rsignal_iter = signal.rend() - 1;

    int i = 0;
    while (rsignal_iter != signal.rbegin() - 1) {
        int elements = std::min((int)coeff.size(), i++);
        *(out_iter++) = inner_product(begin(coeff), begin(coeff) + elements, rsignal_iter--, 0);
    }
}

int gen_random() {
    static default_random_engine e;
    static uniform_int_distribution<int> dist(0, 100);

    return dist(e);
}

// Verifies the gold and the out data are equal
void verify(const vector<int, aligned_allocator<int> >& gold, const vector<int, aligned_allocator<int> >& out) {
    bool match = equal(begin(gold), end(gold), begin(out));
    if (!match) {
        printf("TEST FAILED\n");
        exit(EXIT_FAILURE);
    }
}

uint64_t get_duration_ns(const cl::Event& event) {
    uint64_t nstimestart, nstimeend;
    cl_int err;
    OCL_CHECK(err, err = event.getProfilingInfo<uint64_t>(CL_PROFILING_COMMAND_START, &nstimestart));
    OCL_CHECK(err, err = event.getProfilingInfo<uint64_t>(CL_PROFILING_COMMAND_END, &nstimeend));
    return (nstimeend - nstimestart);
}
void print_summary(std::string k1, std::string k2, uint64_t t1, uint64_t t2, int iterations) {
    double speedup = (double)t1 / (double)t2;
    printf(
        "|-------------------------+-------------------------|\n"
        "| Kernel(%3d iterations)  |    Wall-Clock Time (ns) |\n"
        "|-------------------------+-------------------------|\n",
        iterations);
    printf("| %-23s | %23lu |\n", k1.c_str(), t1);
    printf("| %-23s | %23lu |\n", k2.c_str(), t2);
    printf("|-------------------------+-------------------------|\n");
    printf("| Speedup: | %23lf |\n", speedup);
    printf("|-------------------------+-------------------------|\n");
    printf(
        "Note: Wall Clock Time is meaningful for real hardware execution "
        "only, not for emulation.\n");
    printf(
        "Please refer to profile summary for kernel execution time for "
        "hardware emulation.\n");

    // Performance check for real hardware. t2 must be less than t1.
    if (!xcl::is_emulation() && (t1 < t2)) {
        printf("ERROR: Unexpected Performance is observed\n");
        exit(EXIT_FAILURE);
    }
}
