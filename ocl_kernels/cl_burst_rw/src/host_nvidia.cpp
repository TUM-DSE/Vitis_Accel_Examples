#define CL_HPP_TARGET_OPENCL_VERSION 200
#define CL_HPP_MINIMUM_OPENCL_VERSION 110
#include <CL/cl2.hpp>

#include <chrono>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <vector>

#define DATA_SIZE (128 * 1024) // * sizeof(int) = 512 KB
#define INCR_VALUE 10
// Must match LOCAL_SIZE in vadd_nvidia.cl
#define LOCAL_SIZE 256

static int roundUp(int n, int m) { return ((n + m - 1) / m) * m; }

int main(int argc, char** argv) {
    if (argc != 2) {
        std::cout << "Usage: " << argv[0] << " <vadd_nvidia.cl>" << std::endl;
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
    cl::Kernel kernel(program, "vadd");

    const int data_size = DATA_SIZE;
    const size_t vector_size_bytes = sizeof(int) * data_size;

    // source_input never changes — each iteration below writes it fresh to
    // the device so the kernel's in-place "a[gid] += inc_value" always
    // starts from the same known state and every iteration is a real,
    // independently-timed, correct increment (rather than the previous
    // iteration's already-incremented result silently drifting further with
    // every rep).
    std::vector<int> source_input(data_size);
    std::vector<int> source_hw_results(data_size, 0);
    std::vector<int> source_sw_results(data_size);

    for (int i = 0; i < data_size; i++) {
        source_input[i] = i;
        source_sw_results[i] = i + INCR_VALUE;
    }

    cl::Buffer buffer_rw(context, CL_MEM_READ_WRITE, vector_size_bytes);

    int size = data_size;
    int inc_value = INCR_VALUE;
    kernel.setArg(0, buffer_rw);
    kernel.setArg(1, size);
    kernel.setArg(2, inc_value);

    cl::NDRange global(roundUp(data_size, LOCAL_SIZE));
    cl::NDRange local(LOCAL_SIZE);

    const int n_warmup = 0;
    const int n_reps = 1000;
    uint64_t time_kernel_ocl = 0;
    uint64_t time_data_to_xpu_ocl = 0;
    uint64_t time_data_to_host_ocl = 0;
    // Host-clock accumulator for data-transfer + kernel-execution time only: each interval below
    // is opened right before an OpenCL enqueue call and closed right after it (and any q.finish())
    // completes, so host-side work (loop bookkeeping) is never included.
    uint64_t time_xpu = 0;

    q.finish();

    for (int iter = 0; iter < n_warmup + n_reps; iter++) {
        cl::Event ev_in, ev_k, ev_out;

        auto t_xpu_0 = std::chrono::high_resolution_clock::now();
        q.enqueueWriteBuffer(buffer_rw, CL_FALSE, 0, vector_size_bytes, source_input.data(), nullptr, &ev_in);
        q.finish();
        auto t_xpu_1 = std::chrono::high_resolution_clock::now();
        time_xpu += std::chrono::duration_cast<std::chrono::nanoseconds>(t_xpu_1 - t_xpu_0).count();

        auto t_xpu_2 = std::chrono::high_resolution_clock::now();
        q.enqueueNDRangeKernel(kernel, cl::NullRange, global, local, nullptr, &ev_k);
        q.finish();
        auto t_xpu_3 = std::chrono::high_resolution_clock::now();
        time_xpu += std::chrono::duration_cast<std::chrono::nanoseconds>(t_xpu_3 - t_xpu_2).count();

        auto t_xpu_4 = std::chrono::high_resolution_clock::now();
        q.enqueueReadBuffer(buffer_rw, CL_FALSE, 0, vector_size_bytes, source_hw_results.data(), nullptr, &ev_out);
        q.finish();
        auto t_xpu_5 = std::chrono::high_resolution_clock::now();
        time_xpu += std::chrono::duration_cast<std::chrono::nanoseconds>(t_xpu_5 - t_xpu_4).count();

        cl_ulong s, e;
        ev_in.getProfilingInfo(CL_PROFILING_COMMAND_START, &s);
        ev_in.getProfilingInfo(CL_PROFILING_COMMAND_END,   &e);
        time_data_to_xpu_ocl += e - s;

        ev_k.getProfilingInfo(CL_PROFILING_COMMAND_START, &s);
        ev_k.getProfilingInfo(CL_PROFILING_COMMAND_END,   &e);
        time_kernel_ocl += e - s;

        ev_out.getProfilingInfo(CL_PROFILING_COMMAND_START, &s);
        ev_out.getProfilingInfo(CL_PROFILING_COMMAND_END,   &e);
        time_data_to_host_ocl += e - s;
    }

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

    int match = 0;
    for (int i = 0; i < data_size; i++) {
        if (source_hw_results[i] != source_sw_results[i]) {
            std::cout << "Error: Result mismatch at i=" << i
                      << " CPU=" << source_sw_results[i]
                      << " GPU=" << source_hw_results[i] << std::endl;
            match = 1;
            break;
        }
    }
    std::cout << "TEST " << (match ? "FAILED" : "PASSED") << std::endl;
    return match ? EXIT_FAILURE : EXIT_SUCCESS;
}
