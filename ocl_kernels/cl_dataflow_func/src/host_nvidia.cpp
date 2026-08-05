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
// Must match LOCAL_SIZE in adder_nvidia.cl
#define LOCAL_SIZE 256

static int roundUp(int n, int m) { return ((n + m - 1) / m) * m; }

int main(int argc, char** argv) {
    if (argc != 2) {
        std::cout << "Usage: " << argv[0] << " <adder_nvidia.cl>" << std::endl;
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
    cl::Kernel kernel(program, "adder");

    const int data_size = DATA_SIZE;
    const size_t vector_size_bytes = sizeof(int) * data_size;

    std::vector<int> source_input(data_size);
    std::vector<int> source_hw_results(data_size, 0);
    std::vector<int> source_sw_results(data_size);

    // Create the test data and Software Result
    for (int i = 0; i < data_size; i++) {
        source_input[i] = i;
        source_sw_results[i] = i + INCR_VALUE;
    }

    cl::Buffer buffer_input(context, CL_MEM_READ_ONLY, vector_size_bytes);
    cl::Buffer buffer_output(context, CL_MEM_WRITE_ONLY, vector_size_bytes);

    int inc = INCR_VALUE;
    int size = data_size;
    kernel.setArg(0, buffer_input);
    kernel.setArg(1, buffer_output);
    kernel.setArg(2, inc);
    kernel.setArg(3, size);

    cl::NDRange global(roundUp(data_size, LOCAL_SIZE));
    cl::NDRange local(LOCAL_SIZE);

    const int iterations = 1000;
    cl_ulong ns_to_dev = 0, ns_kernel = 0, ns_to_host = 0;

    q.finish();
    auto t0 = std::chrono::high_resolution_clock::now();

    for (int iter = 0; iter < iterations; iter++) {
        cl::Event ev_in, ev_k, ev_out;

        q.enqueueWriteBuffer(buffer_input, CL_FALSE, 0, vector_size_bytes, source_input.data(), nullptr, &ev_in);
        q.finish();

        q.enqueueNDRangeKernel(kernel, cl::NullRange, global, local, nullptr, &ev_k);
        q.finish();

        q.enqueueReadBuffer(buffer_output, CL_FALSE, 0, vector_size_bytes, source_hw_results.data(), nullptr, &ev_out);
        q.finish();

        cl_ulong s, e;
        ev_in.getProfilingInfo(CL_PROFILING_COMMAND_START, &s);
        ev_in.getProfilingInfo(CL_PROFILING_COMMAND_END,   &e);
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
    std::cout << "cl_dataflow_func_nvidia,"
              << vector_size_bytes << ","
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
