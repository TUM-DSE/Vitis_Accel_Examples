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
#include <CL/cl_ext_xilinx.h>
#include <dirent.h>
#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <fstream>
#include <random>
#include <string>
#include <thread>
#include <vector>

using std::default_random_engine;
using std::generate;
using std::uniform_int_distribution;
using std::vector;

// ---------------------------------------------------------------------------
// Board power sampling.
//
// Unlike NVML on the GPU side, XRT exposes no cumulative energy counter for
// Alveo cards, only instantaneous power, so energy has to be integrated from
// samples. The card's XMC refreshes its sensors once per second (measured on a
// U280), which sets the accuracy floor: a run has to last tens of seconds for
// the integral to be meaningful, and each end of the window carries roughly one
// sample of uncertainty.
//
// Total board power is the sum of the three supply rails, which reproduces the
// figure xbutil reports to the last digit:  12V PEX + 12V AUX + 3V3 PEX.
// ---------------------------------------------------------------------------

static bool read_sysfs_long(const std::string& path, long* out) {
    std::ifstream f(path);
    long v = 0;
    if (!f || !(f >> v)) return false;
    *out = v;
    return true;
}

// The XMC node is named xmc.<instance>, so the instance number has to be found
// rather than hardcoded.
static std::string find_xmc_dir(const std::string& bdf) {
    std::string base = "/sys/bus/pci/devices/" + bdf + "/";
    DIR* d = opendir(base.c_str());
    if (!d) return "";
    std::string found;
    while (struct dirent* e = readdir(d)) {
        std::string name(e->d_name);
        if (name.rfind("xmc.", 0) == 0) {
            found = base + name + "/";
            break;
        }
    }
    closedir(d);
    return found;
}

// Rails are reported in millivolts and milliamps.
static bool read_board_power(const std::string& xmc, double* watts) {
    static const char* rails[3][2] = {
        {"xmc_12v_pex_vol", "xmc_12v_pex_curr"},
        {"xmc_12v_aux_vol", "xmc_12v_aux_curr"},
        {"xmc_3v3_pex_vol", "xmc_3v3_pex_curr"},
    };
    double sum = 0.0;
    for (int i = 0; i < 3; i++) {
        long mv = 0, ma = 0;
        if (!read_sysfs_long(xmc + rails[i][0], &mv)) return false;
        if (!read_sysfs_long(xmc + rails[i][1], &ma)) return false;
        sum += (mv / 1000.0) * (ma / 1000.0);
    }
    *watts = sum;
    return true;
}

// Samples every 250 ms -- oversampling the 1 Hz sensor so each refresh is picked
// up promptly -- and accumulates energy trapezoidally. Best-effort, exactly like
// the NVML path on the GPU: any failure warns and reports no energy rather than
// taking the benchmark down with it.
struct PowerSampler {
    std::string xmc;
    std::atomic<bool> stop{false};
    std::atomic<bool> failed{false};
    std::thread th;
    double energy_j = 0.0;
    unsigned samples = 0;
    double last_w = 0.0;
    std::chrono::steady_clock::time_point last_t;
    bool active = false;

    bool open(const cl::Device& device) {
        char bdf[64] = {0};
        if (clGetDeviceInfo(device(), CL_DEVICE_PCIE_BDF, sizeof(bdf) - 1, bdf, nullptr) !=
            CL_SUCCESS) {
            std::cerr << "power: device BDF unavailable, energy not measured\n";
            return false;
        }
        xmc = find_xmc_dir(bdf);
        if (xmc.empty()) {
            std::cerr << "power: no XMC sysfs node under " << bdf << ", energy not measured\n";
            return false;
        }
        double w = 0.0;
        if (!read_board_power(xmc, &w)) {
            std::cerr << "power: cannot read the rails in " << xmc << ", energy not measured\n";
            return false;
        }
        active = true;
        return true;
    }

    void start() {
        if (!active) return;
        if (!read_board_power(xmc, &last_w)) {
            failed = true;
            active = false;
            return;
        }
        last_t = std::chrono::steady_clock::now();
        th = std::thread([this] {
            while (!stop.load(std::memory_order_relaxed)) {
                std::this_thread::sleep_for(std::chrono::milliseconds(250));
                accumulate();
            }
        });
    }

    void accumulate() {
        double w = 0.0;
        if (!read_board_power(xmc, &w)) {
            failed = true;
            return;
        }
        auto t = std::chrono::steady_clock::now();
        energy_j += 0.5 * (last_w + w) * std::chrono::duration<double>(t - last_t).count();
        samples++;
        last_w = w;
        last_t = t;
    }

    // Closes the window with one final sample so the tail between the last
    // periodic sample and the end of the loop is not dropped.
    bool finish() {
        if (!active) return false;
        stop = true;
        if (th.joinable()) th.join();
        accumulate();
        if (failed.load()) {
            std::cerr << "power: a rail read failed during the run, energy not measured\n";
            return false;
        }
        return true;
    }
};

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
    cl::Device selected_device;
    for (unsigned int i = 0; i < devices.size(); i++) {
        auto device = devices[i];
        selected_device = device;
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

    PowerSampler power;
    bool have_energy = power.open(selected_device);

    // This is required for proper time measurements in Proteus. We add it here
    // as well to have the same host code for Proteus and native.
    q.finish();

    // The sampled window covers the whole loop, idle time included, so the
    // matching denominator for average power is wall-clock time across the loop
    // rather than time_xpu (which excludes host-side bookkeeping).
    if (have_energy) power.start();
    auto t_loop_0 = std::chrono::high_resolution_clock::now();

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

    auto t_loop_1 = std::chrono::high_resolution_clock::now();
    if (have_energy) have_energy = power.finish();
    uint64_t time_loop =
        std::chrono::duration_cast<std::chrono::nanoseconds>(t_loop_1 - t_loop_0).count();

    verify(gold, C, columns);

    double ns_per_s = 1000000000;
    double time_loop_s = time_loop / ns_per_s;
    std::cout << "app_name,in_size,out_size,reps_warmup,reps,time_xpu,time_data_to_xpu,time_kernel,"
                 "time_data_to_host,time_loop,energy_j,avg_power_w\n"
              << "cl_array_partition_npu,"
              << in_size_bytes * 2 << ","
              << out_size_bytes << ","
              << n_warmup << ","
              << n_reps << ","
              << time_xpu / ns_per_s << ","
              << time_data_to_xpu_ocl / ns_per_s << ","
              << time_kernel_ocl / ns_per_s << ","
              << time_data_to_host_ocl / ns_per_s << ","
              << time_loop_s << ",";
    // Left empty rather than zero when unavailable, so a missing measurement cannot
    // be mistaken for a board that drew no power.
    if (have_energy) {
        std::cout << power.energy_j << ","
                  << (time_loop_s > 0.0 ? power.energy_j / time_loop_s : 0.0);
    } else {
        std::cout << ",";
    }
    std::cout << "\n";

    printf("TEST PASSED\n\n");

    return EXIT_SUCCESS;
}
