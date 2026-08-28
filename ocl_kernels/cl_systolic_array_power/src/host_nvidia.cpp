#define CL_HPP_TARGET_OPENCL_VERSION 200
#define CL_HPP_MINIMUM_OPENCL_VERSION 110
#include <CL/cl2.hpp>

#include "power_nvidia.h"

#include <chrono>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <vector>

#define DATA_SIZE 24
// Must match TILE_SIZE in mmult_nvidia.cl
#define TILE_SIZE 16

static int roundUp(int n, int m) { return ((n + m - 1) / m) * m; }

void m_softwareGold(std::vector<int>& in1, std::vector<int>& in2, std::vector<int>& out) {
    for (int i = 0; i < DATA_SIZE; i++)
        for (int j = 0; j < DATA_SIZE; j++)
            for (int k = 0; k < DATA_SIZE; k++)
                out[i * DATA_SIZE + j] += in1[i * DATA_SIZE + k] * in2[k * DATA_SIZE + j];
}

int main(int argc, char** argv) {
    if (argc != 2) {
        std::cout << "Usage: " << argv[0] << " <mmult_nvidia.cl>" << std::endl;
        return EXIT_FAILURE;
    }

    std::ifstream f(argv[1]);
    if (!f) {
        std::cerr << "Cannot open kernel: " << argv[1] << std::endl;
        return EXIT_FAILURE;
    }
    std::string src((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());

    std::vector<cl::Platform> platforms;
    cl::Platform::get(&platforms);
    cl::Device device;
    bool found = false;
    for (auto& p : platforms) {
        std::vector<cl::Device> devs;
        if (p.getDevices(CL_DEVICE_TYPE_GPU, &devs) != CL_SUCCESS || devs.empty())
            continue;
        for (auto& d : devs) {
            if (d.getInfo<CL_DEVICE_VENDOR>().find("NVIDIA") != std::string::npos) {
                device = d;
                found = true;
                break;
            }
        }
        if (!found) { device = devs[0]; found = true; }
        if (found) break;
    }
    if (!found) {
        std::cerr << "No GPU device found" << std::endl;
        return EXIT_FAILURE;
    }
    std::cout << "Device: " << device.getInfo<CL_DEVICE_NAME>() << std::endl;

    cl::Context context(device);
    cl::CommandQueue q(context, device, CL_QUEUE_PROFILING_ENABLE);

    cl::Program::Sources sources;
    sources.push_back({src.c_str(), src.size()});
    cl::Program program(context, sources);
    if (program.build({device}) != CL_SUCCESS) {
        std::cerr << "Build error:\n"
                  << program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(device) << std::endl;
        return EXIT_FAILURE;
    }
    cl::Kernel kernel(program, "mmult");

    const size_t matrix_size       = DATA_SIZE * DATA_SIZE;
    const size_t matrix_size_bytes = sizeof(int) * matrix_size;

    std::vector<int> in1(matrix_size), in2(matrix_size);
    std::vector<int> hw_result(matrix_size, 0), sw_result(matrix_size, 0);
    for (size_t i = 0; i < matrix_size; i++) {
        in1[i] = i % 10;
        in2[i] = i % 10;
    }

    cl::Buffer buf_a(context, CL_MEM_READ_ONLY,  matrix_size_bytes);
    cl::Buffer buf_b(context, CL_MEM_READ_ONLY,  matrix_size_bytes);
    cl::Buffer buf_c(context, CL_MEM_WRITE_ONLY, matrix_size_bytes);

    int a_row = DATA_SIZE, a_col = DATA_SIZE, b_col = DATA_SIZE;
    kernel.setArg(0, buf_a);
    kernel.setArg(1, buf_b);
    kernel.setArg(2, buf_c);
    kernel.setArg(3, a_row);
    kernel.setArg(4, a_col);
    kernel.setArg(5, b_col);

    cl::NDRange global(roundUp(a_row, TILE_SIZE), roundUp(b_col, TILE_SIZE));
    cl::NDRange local(TILE_SIZE, TILE_SIZE);

    const int n_warmup = 0;
    const int n_reps = 500000;
    uint64_t time_kernel_ocl = 0;
    uint64_t time_data_to_xpu_ocl = 0;
    uint64_t time_data_to_host_ocl = 0;
    // Host-clock accumulator for data-transfer + kernel-execution time only: each interval below
    // is opened right before an OpenCL enqueue call and closed right after it (and any q.finish())
    // completes, so host-side work (loop bookkeeping) is never included.
    uint64_t time_xpu = 0;

    PowerMeter power;
    bool have_energy = power.open(device());

    q.finish();

    // Idle baseline, taken in the same process state as the run that follows:
    // context and program built, buffers allocated, nothing enqueued.
    const double idle_window_s = 20.0;
    double idle_w = 0.0;
    bool have_idle = have_energy && power.measure_idle(idle_window_s, &idle_w);
    if (have_energy && !have_idle) {
        std::cerr << "NVML: idle baseline measurement failed, reporting gross energy only\n";
    }

    // The energy counter covers the whole loop window, idle time included, so the
    // matching denominator for average power is wall-clock time across the loop
    // rather than time_xpu (which excludes host-side bookkeeping).
    if (have_energy) power.start();
    auto t_loop_0 = std::chrono::steady_clock::now();

    for (int iter = 0; iter < n_warmup + n_reps; iter++) {
        cl::Event ev_a, ev_b, ev_k, ev_r;

        auto t_xpu_0 = std::chrono::high_resolution_clock::now();
        q.enqueueWriteBuffer(buf_a, CL_FALSE, 0, matrix_size_bytes, in1.data(), nullptr, &ev_a);
        q.enqueueWriteBuffer(buf_b, CL_FALSE, 0, matrix_size_bytes, in2.data(), nullptr, &ev_b);
        q.finish();
        auto t_xpu_1 = std::chrono::high_resolution_clock::now();
        time_xpu += std::chrono::duration_cast<std::chrono::nanoseconds>(t_xpu_1 - t_xpu_0).count();

        auto t_xpu_2 = std::chrono::high_resolution_clock::now();
        q.enqueueNDRangeKernel(kernel, cl::NullRange, global, local, nullptr, &ev_k);
        q.finish();
        auto t_xpu_3 = std::chrono::high_resolution_clock::now();
        time_xpu += std::chrono::duration_cast<std::chrono::nanoseconds>(t_xpu_3 - t_xpu_2).count();

        auto t_xpu_4 = std::chrono::high_resolution_clock::now();
        q.enqueueReadBuffer(buf_c, CL_FALSE, 0, matrix_size_bytes, hw_result.data(), nullptr, &ev_r);
        q.finish();
        auto t_xpu_5 = std::chrono::high_resolution_clock::now();
        time_xpu += std::chrono::duration_cast<std::chrono::nanoseconds>(t_xpu_5 - t_xpu_4).count();

        cl_ulong s, e;
        ev_a.getProfilingInfo(CL_PROFILING_COMMAND_START, &s);
        ev_a.getProfilingInfo(CL_PROFILING_COMMAND_END,   &e);
        time_data_to_xpu_ocl += e - s;
        ev_b.getProfilingInfo(CL_PROFILING_COMMAND_START, &s);
        ev_b.getProfilingInfo(CL_PROFILING_COMMAND_END,   &e);
        time_data_to_xpu_ocl += e - s;

        ev_k.getProfilingInfo(CL_PROFILING_COMMAND_START, &s);
        ev_k.getProfilingInfo(CL_PROFILING_COMMAND_END,   &e);
        time_kernel_ocl += e - s;

        ev_r.getProfilingInfo(CL_PROFILING_COMMAND_START, &s);
        ev_r.getProfilingInfo(CL_PROFILING_COMMAND_END,   &e);
        time_data_to_host_ocl += e - s;
    }

    auto t_loop_1 = std::chrono::steady_clock::now();
    if (have_energy) have_energy = power.finish();
    power.close();
    uint64_t time_loop =
        std::chrono::duration_cast<std::chrono::nanoseconds>(t_loop_1 - t_loop_0).count();

    double ns_per_s = 1000000000;
    print_power_csv("cl_systolic_array_power", matrix_size_bytes * 2, matrix_size_bytes, n_warmup,
                    n_reps, time_xpu / ns_per_s, time_data_to_xpu_ocl / ns_per_s,
                    time_kernel_ocl / ns_per_s, time_data_to_host_ocl / ns_per_s,
                    time_loop / ns_per_s, have_energy, power.energy_j, have_idle, idle_w,
                    idle_window_s);

    m_softwareGold(in1, in2, sw_result);

    int match = 0;
    for (int i = 0; i < DATA_SIZE * DATA_SIZE; i++) {
        if (hw_result[i] != sw_result[i]) {
            std::cout << "Error: Result mismatch at i=" << i
                      << " CPU=" << sw_result[i]
                      << " GPU=" << hw_result[i] << std::endl;
            match = 1;
            break;
        }
    }
    std::cout << "TEST " << (match ? "FAILED" : "PASSED") << std::endl;
    return match ? EXIT_FAILURE : EXIT_SUCCESS;
}
