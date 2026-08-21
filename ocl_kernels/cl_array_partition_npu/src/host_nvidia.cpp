#define CL_HPP_TARGET_OPENCL_VERSION 200
#define CL_HPP_MINIMUM_OPENCL_VERSION 110
#include <CL/cl2.hpp>
#include <nvml.h>

#include <algorithm>
#include <chrono>
#include <cstdint>
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

// From cl_nv_device_attribute_query, so NVML can be pointed at the very board
// OpenCL selected instead of assuming there is only one GPU in the machine.
#define CL_DEVICE_PCI_BUS_ID_NV 0x4008
#define CL_DEVICE_PCI_SLOT_ID_NV 0x4009

// Binds NVML to the OpenCL device, matching on PCI address and falling back to
// NVML index 0 if the driver does not expose the extension. Energy measurement
// is best-effort: every failure path warns and leaves the benchmark itself
// running, it just reports no energy.
static bool nvml_open(const cl::Device& device, nvmlDevice_t* out) {
    nvmlReturn_t r = nvmlInit();
    if (r != NVML_SUCCESS) {
        std::cerr << "NVML: init failed (" << nvmlErrorString(r) << "), energy not measured\n";
        return false;
    }

    cl_uint bus = 0, slot = 0;
    if (device.getInfo(CL_DEVICE_PCI_BUS_ID_NV, &bus) == CL_SUCCESS &&
        device.getInfo(CL_DEVICE_PCI_SLOT_ID_NV, &slot) == CL_SUCCESS) {
        char pci[32];
        snprintf(pci, sizeof(pci), "0000:%02x:%02x.0", bus, slot);
        r = nvmlDeviceGetHandleByPciBusId(pci, out);
        if (r == NVML_SUCCESS) return true;
        std::cerr << "NVML: no device at " << pci << " (" << nvmlErrorString(r)
                  << "), falling back to NVML index 0\n";
    }

    r = nvmlDeviceGetHandleByIndex(0, out);
    if (r != NVML_SUCCESS) {
        std::cerr << "NVML: no device (" << nvmlErrorString(r) << "), energy not measured\n";
        nvmlShutdown();
        return false;
    }
    return true;
}

// Board-wide energy counter in millijoules since the driver was last reloaded.
// Monotonic, so the loop's energy is the difference of two reads. Needs Volta or
// newer; older GPUs answer NVML_ERROR_NOT_SUPPORTED.
static bool nvml_energy_mj(nvmlDevice_t dev, unsigned long long* mj) {
    return nvmlDeviceGetTotalEnergyConsumption(dev, mj) == NVML_SUCCESS;
}

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

void verify(vector<int>& gold, vector<int>& output, int columns) {
    for (int i = 0; i < (int)output.size(); i++) {
        if (output[i] != gold[i]) {
            printf("Mismatch %d: gold: %d device: %d\n", i, gold[i], output[i]);
            print(output.data(), columns, columns);
            exit(EXIT_FAILURE);
        }
    }
}

// Benchmarks the matmul kernel (C = A x B, int32, row major) so the same GEMM
// can be measured on FPGA, GPU and NPU. The FPGA's array-partitioned variant is
// replaced here by a local-memory tiled one, see matmul_nvidia.cl.
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

    static const int columns = 128;
    static const int rows = 128;

    vector<int8_t> A(columns * rows);
    vector<int8_t> B(columns * rows);
    vector<int> C(columns * rows, 0);
    vector<int> gold(columns * rows, 0);

    generate(begin(A), end(A), gen_random);
    generate(begin(B), end(B), gen_random);

    printf("A:\n");
    print(A.data(), columns, rows);
    printf("B:\n");
    print(B.data(), columns, rows);
    matmul(gold.data(), A.data(), B.data(), columns);

    printf("Gold:\n");
    print(gold.data(), columns, rows);

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
    // int8 in, int32 out: the two directions no longer have the same size
    size_t in_size_bytes = columns * rows * sizeof(int8_t);
    size_t out_size_bytes = columns * rows * sizeof(int);
    cl::Buffer buffer_a(context, CL_MEM_READ_ONLY, in_size_bytes);
    cl::Buffer buffer_b(context, CL_MEM_READ_ONLY, in_size_bytes);
    cl::Buffer buffer_c(context, CL_MEM_WRITE_ONLY, out_size_bytes);

    cl::Kernel matmul_partition_kernel(program, "matmul_partition");
    matmul_partition_kernel.setArg(0, buffer_a);
    matmul_partition_kernel.setArg(1, buffer_b);
    matmul_partition_kernel.setArg(2, buffer_c);
    matmul_partition_kernel.setArg(3, columns);

    cl::NDRange global(roundUp(rows, TILE_SIZE), roundUp(columns, TILE_SIZE));
    cl::NDRange local(TILE_SIZE, TILE_SIZE);

    const int n_warmup = 0;
    const int n_reps = 100000;
    uint64_t time_kernel_ocl = 0;
    uint64_t time_data_to_xpu_ocl = 0;
    uint64_t time_data_to_host_ocl = 0;
    // Host-clock accumulator for data-transfer + kernel-execution time only: each interval below
    // is opened right before an OpenCL enqueue call and closed right after it (and any q.finish())
    // completes, so host-side work (loop bookkeeping) is never included.
    uint64_t time_xpu = 0;

    nvmlDevice_t nvml_dev{};
    unsigned long long energy_start_mj = 0;
    unsigned long long energy_end_mj = 0;
    bool have_nvml = nvml_open(device, &nvml_dev);
    bool have_energy = have_nvml && nvml_energy_mj(nvml_dev, &energy_start_mj);
    if (have_nvml && !have_energy) {
        std::cerr << "NVML: total energy counter unsupported on this GPU"
                     " (needs Volta or newer), energy not measured\n";
    }

    q.finish();

    // The energy counter covers the whole loop window, idle time included, so the
    // matching denominator for average power is wall-clock time across the loop
    // rather than time_xpu (which excludes host-side bookkeeping).
    auto t_loop_0 = std::chrono::steady_clock::now();

    // Running the tiled matmul kernel
    for (int iter = 0; iter < n_warmup + n_reps; iter++) {
        cl::Event ev_a, ev_b, ev_k, ev_r;

        auto t_xpu_0 = std::chrono::steady_clock::now();
        q.enqueueWriteBuffer(buffer_a, CL_FALSE, 0, in_size_bytes, A.data(), nullptr, &ev_a);
        q.enqueueWriteBuffer(buffer_b, CL_FALSE, 0, in_size_bytes, B.data(), nullptr, &ev_b);
        q.finish();
        auto t_xpu_1 = std::chrono::steady_clock::now();
        time_xpu += std::chrono::duration_cast<std::chrono::nanoseconds>(t_xpu_1 - t_xpu_0).count();

        auto t_xpu_2 = std::chrono::steady_clock::now();
        q.enqueueNDRangeKernel(matmul_partition_kernel, cl::NullRange, global, local, nullptr, &ev_k);
        q.finish();
        auto t_xpu_3 = std::chrono::steady_clock::now();
        time_xpu += std::chrono::duration_cast<std::chrono::nanoseconds>(t_xpu_3 - t_xpu_2).count();

        auto t_xpu_4 = std::chrono::steady_clock::now();
        q.enqueueReadBuffer(buffer_c, CL_FALSE, 0, out_size_bytes, C.data(), nullptr, &ev_r);
        q.finish();
        auto t_xpu_5 = std::chrono::steady_clock::now();
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
    uint64_t time_loop =
        std::chrono::duration_cast<std::chrono::nanoseconds>(t_loop_1 - t_loop_0).count();

    if (have_energy && !nvml_energy_mj(nvml_dev, &energy_end_mj)) {
        std::cerr << "NVML: reading the energy counter after the loop failed,"
                     " energy not measured\n";
        have_energy = false;
    }
    if (have_nvml) nvmlShutdown();

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
    // be mistaken for a GPU that drew no power.
    if (have_energy) {
        double energy_j = (energy_end_mj - energy_start_mj) / 1000.0;
        std::cout << energy_j << "," << (time_loop_s > 0.0 ? energy_j / time_loop_s : 0.0);
    } else {
        std::cout << ",";
    }
    std::cout << "\n";

    printf("TEST PASSED\n\n");

    return EXIT_SUCCESS;
}
