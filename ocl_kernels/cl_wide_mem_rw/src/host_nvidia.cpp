#define CL_HPP_TARGET_OPENCL_VERSION 200
#define CL_HPP_MINIMUM_OPENCL_VERSION 110
#include <CL/cl2.hpp>

#include <chrono>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <vector>

// DATA_SIZE is in ints; the kernel reads/writes it 16 at a time as uint16
// vectors, so DATA_SIZE must be a multiple of 16.
#define DATA_SIZE (4 * 1024 * 1024) // 4M ints = 16 MB per vector
#define VECTOR_SIZE 16
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
    const int size_in16 = data_size / VECTOR_SIZE;
    const size_t vector_size_bytes = sizeof(int) * data_size;

    std::vector<int> source_in1(data_size), source_in2(data_size);
    std::vector<int> source_hw_results(data_size, 0), source_sw_results(data_size);

    for (int i = 0; i < data_size; i++) {
        source_in1[i] = i;
        source_in2[i] = i * i;
        source_sw_results[i] = i * i + i;
    }

    cl::Buffer buffer_in1(context, CL_MEM_READ_ONLY, vector_size_bytes);
    cl::Buffer buffer_in2(context, CL_MEM_READ_ONLY, vector_size_bytes);
    cl::Buffer buffer_out(context, CL_MEM_WRITE_ONLY, vector_size_bytes);

    kernel.setArg(0, buffer_in1);
    kernel.setArg(1, buffer_in2);
    kernel.setArg(2, buffer_out);
    kernel.setArg(3, size_in16);

    cl::NDRange global(roundUp(size_in16, LOCAL_SIZE));
    cl::NDRange local(LOCAL_SIZE);

    const int iterations = 1000;
    cl_ulong ns_to_dev = 0, ns_kernel = 0, ns_to_host = 0;

    q.finish();
    auto t0 = std::chrono::high_resolution_clock::now();

    for (int iter = 0; iter < iterations; iter++) {
        cl::Event ev_a, ev_b, ev_k, ev_out;

        q.enqueueWriteBuffer(buffer_in1, CL_FALSE, 0, vector_size_bytes, source_in1.data(), nullptr, &ev_a);
        q.enqueueWriteBuffer(buffer_in2, CL_FALSE, 0, vector_size_bytes, source_in2.data(), nullptr, &ev_b);
        q.finish();

        q.enqueueNDRangeKernel(kernel, cl::NullRange, global, local, nullptr, &ev_k);
        q.finish();

        q.enqueueReadBuffer(buffer_out, CL_FALSE, 0, vector_size_bytes, source_hw_results.data(), nullptr, &ev_out);
        q.finish();

        cl_ulong s, e;
        ev_a.getProfilingInfo(CL_PROFILING_COMMAND_START, &s);
        ev_a.getProfilingInfo(CL_PROFILING_COMMAND_END,   &e);
        ns_to_dev += e - s;
        ev_b.getProfilingInfo(CL_PROFILING_COMMAND_START, &s);
        ev_b.getProfilingInfo(CL_PROFILING_COMMAND_END,   &e);
        ns_to_dev += e - s;

        ev_k.getProfilingInfo(CL_PROFILING_COMMAND_START, &s);
        ev_k.getProfilingInfo(CL_PROFILING_COMMAND_END,   &e);
        ns_kernel += e - s;

        ev_out.getProfilingInfo(CL_PROFILING_COMMAND_START, &s);
        ev_out.getProfilingInfo(CL_PROFILING_COMMAND_END,   &e);
        ns_to_host += e - s;
    }

    auto t1 = std::chrono::high_resolution_clock::now();
    int64_t ns_cpu = std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0).count();

    std::cout << "app_name,kernel_input_data_size,kernel_output_data_size,iterations,"
                 "time_cpu,data_to_fpga_time_ocl,kernel_time_ocl,data_to_host_time_ocl\n";
    std::cout << "cl_wide_mem_rw_nvidia,"
              << vector_size_bytes * 2 << ","
              << vector_size_bytes << ","
              << iterations << ","
              << std::setprecision(std::numeric_limits<double>::digits10)
              << ns_cpu    / 1e9 << ","
              << ns_to_dev / 1e9 << ","
              << ns_kernel / 1e9 << ","
              << ns_to_host / 1e9 << "\n";

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
