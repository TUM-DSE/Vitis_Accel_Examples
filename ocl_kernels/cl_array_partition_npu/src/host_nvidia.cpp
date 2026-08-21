#define CL_HPP_TARGET_OPENCL_VERSION 200
#define CL_HPP_MINIMUM_OPENCL_VERSION 110
#include <CL/cl2.hpp>

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <fstream>
#include <iostream>
#include <random>
#include <vector>

// Must match TILE_SIZE in matmul_nvidia.cl
#define TILE_SIZE 16

using std::default_random_engine;
using std::generate;
using std::uniform_int_distribution;
using std::vector;

static int roundUp(int n, int m) { return ((n + m - 1) / m) * m; }

// row major
void matmul(int* C, int* A, int* B, int M) {
    for (int k = 0; k < M; k++) {
        for (int j = 0; j < M; j++) {
            for (int i = 0; i < M; i++) {
                C[k * M + j] += A[k * M + i] * B[i * M + j];
            }
        }
    }
}

int gen_random() {
    static default_random_engine e;
    static uniform_int_distribution<int> dist(0, 10);

    return dist(e);
}

void print(int* data, int columns, int rows) {
    for (int r = 0; r < 10; r++) {
        for (int c = 0; c < 10; c++) {
            printf("%4d ", data[r * columns + c]);
        }
        printf("…\n");
    }
    for (int r = 0; r < 10; r++) {
        printf("   %s ", "…");
    }
    printf("⋱\n\n");
}

void verify(vector<int>& gold, vector<int>& output) {
    for (int i = 0; i < (int)output.size(); i++) {
        if (output[i] != gold[i]) {
            printf("Mismatch %d: gold: %d device: %d\n", i, gold[i], output[i]);
            print(output.data(), 16, 16);
            exit(EXIT_FAILURE);
        }
    }
}

// This example illustrates how to use array partitioning attributes in OpenCL
// kernels for FPGA devices using matmul. On the GPU the partitioned variant is
// replaced by a local-memory tiled one, see matmul_nvidia.cl.
int main(int argc, char** argv) {
    if (argc != 2) {
        std::cout << "Usage: " << argv[0] << " <matmul_nvidia.cl>" << std::endl;
        return EXIT_FAILURE;
    }

    std::ifstream f(argv[1]);
    if (!f) {
        std::cerr << "Cannot open kernel: " << argv[1] << std::endl;
        return EXIT_FAILURE;
    }
    std::string src((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());

    static const int columns = 64;
    static const int rows = 64;

    vector<int> A(columns * rows);
    vector<int> B(columns * rows);
    vector<int> gold1(columns * rows, 0);
    vector<int> C(columns * rows, 0);
    vector<int> D(columns * rows);
    vector<int> E(columns * rows);
    vector<int> F(columns * rows, 0);
    vector<int> gold2(columns * rows, 0);

    generate(begin(A), end(A), gen_random);
    generate(begin(B), end(B), gen_random);
    generate(begin(D), end(D), gen_random);
    generate(begin(E), end(E), gen_random);

    printf("A:\n");
    print(A.data(), columns, rows);
    printf("B:\n");
    print(B.data(), columns, rows);
    matmul(gold1.data(), A.data(), B.data(), columns);

    printf("Gold1:\n");
    print(gold1.data(), columns, rows);
    std::cout << "D:\n";
    print(D.data(), columns, rows);
    std::cout << "E:\n";
    print(E.data(), columns, rows);
    matmul(gold2.data(), D.data(), E.data(), columns);
    std::cout << "Gold2:\n";
    print(gold2.data(), columns, rows);

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

    // compute the size of array in bytes
    size_t array_size_bytes = columns * rows * sizeof(int);
    cl::Buffer buffer_a(context, CL_MEM_READ_ONLY, array_size_bytes);
    cl::Buffer buffer_b(context, CL_MEM_READ_ONLY, array_size_bytes);
    cl::Buffer buffer_c(context, CL_MEM_WRITE_ONLY, array_size_bytes);
    cl::Buffer buffer_d(context, CL_MEM_READ_ONLY, array_size_bytes);
    cl::Buffer buffer_e(context, CL_MEM_READ_ONLY, array_size_bytes);
    cl::Buffer buffer_f(context, CL_MEM_WRITE_ONLY, array_size_bytes);

    cl::Kernel matmul_kernel(program, "matmul");
    matmul_kernel.setArg(0, buffer_a);
    matmul_kernel.setArg(1, buffer_b);
    matmul_kernel.setArg(2, buffer_c);
    matmul_kernel.setArg(3, columns);

    cl::NDRange global(roundUp(rows, TILE_SIZE), roundUp(columns, TILE_SIZE));
    cl::NDRange local(TILE_SIZE, TILE_SIZE);

    const int n_warmup = 0;
    const int n_reps = 16000;
    uint64_t time_kernel_ocl = 0;
    uint64_t time_data_to_xpu_ocl = 0;
    uint64_t time_data_to_host_ocl = 0;
    // Host-clock accumulator for data-transfer + kernel-execution time only: each interval below
    // is opened right before an OpenCL enqueue call and closed right after it (and any q.finish())
    // completes, so host-side work (loop bookkeeping) is never included.
    uint64_t time_xpu = 0;

    q.finish();

    // Running the naive matmul kernel for half of the reps
    for (int iter = 0; iter < (n_warmup + n_reps) / 2; iter++) {
        cl::Event ev_a, ev_b, ev_k, ev_r;

        auto t_xpu_0 = std::chrono::high_resolution_clock::now();
        q.enqueueWriteBuffer(buffer_a, CL_FALSE, 0, array_size_bytes, A.data(), nullptr, &ev_a);
        q.enqueueWriteBuffer(buffer_b, CL_FALSE, 0, array_size_bytes, B.data(), nullptr, &ev_b);
        q.finish();
        auto t_xpu_1 = std::chrono::high_resolution_clock::now();
        time_xpu += std::chrono::duration_cast<std::chrono::nanoseconds>(t_xpu_1 - t_xpu_0).count();

        auto t_xpu_2 = std::chrono::high_resolution_clock::now();
        q.enqueueNDRangeKernel(matmul_kernel, cl::NullRange, global, local, nullptr, &ev_k);
        q.finish();
        auto t_xpu_3 = std::chrono::high_resolution_clock::now();
        time_xpu += std::chrono::duration_cast<std::chrono::nanoseconds>(t_xpu_3 - t_xpu_2).count();

        auto t_xpu_4 = std::chrono::high_resolution_clock::now();
        q.enqueueReadBuffer(buffer_c, CL_FALSE, 0, array_size_bytes, C.data(), nullptr, &ev_r);
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

    verify(gold1, C);

    cl::Kernel matmul_partition_kernel(program, "matmul_partition");
    matmul_partition_kernel.setArg(0, buffer_d);
    matmul_partition_kernel.setArg(1, buffer_e);
    matmul_partition_kernel.setArg(2, buffer_f);
    matmul_partition_kernel.setArg(3, columns);

    // Running the tiled matmul kernel for the other half of the reps
    for (int iter = 0; iter < (n_warmup + n_reps) / 2; iter++) {
        cl::Event ev_a, ev_b, ev_k, ev_r;

        auto t_xpu_0 = std::chrono::high_resolution_clock::now();
        q.enqueueWriteBuffer(buffer_d, CL_FALSE, 0, array_size_bytes, D.data(), nullptr, &ev_a);
        q.enqueueWriteBuffer(buffer_e, CL_FALSE, 0, array_size_bytes, E.data(), nullptr, &ev_b);
        q.finish();
        auto t_xpu_1 = std::chrono::high_resolution_clock::now();
        time_xpu += std::chrono::duration_cast<std::chrono::nanoseconds>(t_xpu_1 - t_xpu_0).count();

        auto t_xpu_2 = std::chrono::high_resolution_clock::now();
        q.enqueueNDRangeKernel(matmul_partition_kernel, cl::NullRange, global, local, nullptr, &ev_k);
        q.finish();
        auto t_xpu_3 = std::chrono::high_resolution_clock::now();
        time_xpu += std::chrono::duration_cast<std::chrono::nanoseconds>(t_xpu_3 - t_xpu_2).count();

        auto t_xpu_4 = std::chrono::high_resolution_clock::now();
        q.enqueueReadBuffer(buffer_f, CL_FALSE, 0, array_size_bytes, F.data(), nullptr, &ev_r);
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

    verify(gold2, F);

    double ns_per_s = 1000000000;
    std::cout << "app_name,in_size,out_size,reps_warmup,reps,time_xpu,time_data_to_xpu,time_kernel,time_data_to_host\n"
              << "cl_array_partition_npu,"
              << array_size_bytes * 2 << ","
              << array_size_bytes << ","
              << n_warmup << ","
              << n_reps << ","
              << time_xpu / ns_per_s << ","
              << time_data_to_xpu_ocl / ns_per_s << ","
              << time_kernel_ocl / ns_per_s << ","
              << time_data_to_host_ocl / ns_per_s
              << "\n";

    printf("TEST PASSED\n\n");

    return EXIT_SUCCESS;
}
