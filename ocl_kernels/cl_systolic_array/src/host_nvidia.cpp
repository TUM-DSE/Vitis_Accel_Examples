#define CL_HPP_TARGET_OPENCL_VERSION 200
#define CL_HPP_MINIMUM_OPENCL_VERSION 110
#include <CL/cl2.hpp>

#include <chrono>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
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

    const int iterations = 65400;
    cl_ulong ns_to_dev = 0, ns_kernel = 0, ns_to_host = 0;

    q.finish();
    auto t0 = std::chrono::high_resolution_clock::now();

    for (int iter = 0; iter < iterations; iter++) {
        cl::Event ev_a, ev_b, ev_k, ev_r;

        q.enqueueWriteBuffer(buf_a, CL_FALSE, 0, matrix_size_bytes, in1.data(), nullptr, &ev_a);
        q.enqueueWriteBuffer(buf_b, CL_FALSE, 0, matrix_size_bytes, in2.data(), nullptr, &ev_b);
        q.finish();

        q.enqueueNDRangeKernel(kernel, cl::NullRange, global, local, nullptr, &ev_k);
        q.finish();

        q.enqueueReadBuffer(buf_c, CL_FALSE, 0, matrix_size_bytes, hw_result.data(), nullptr, &ev_r);
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

        ev_r.getProfilingInfo(CL_PROFILING_COMMAND_START, &s);
        ev_r.getProfilingInfo(CL_PROFILING_COMMAND_END,   &e);
        ns_to_host += e - s;
    }

    auto t1 = std::chrono::high_resolution_clock::now();
    int64_t ns_cpu = std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0).count();

    std::cout << "app_name,kernel_input_data_size,kernel_output_data_size,iterations,"
                 "time_cpu,data_to_fpga_time_ocl,kernel_time_ocl,data_to_host_time_ocl\n";
    std::cout << "cl_systolic_array_nvidia,"
              << matrix_size_bytes * 2 << ","
              << matrix_size_bytes << ","
              << iterations << ","
              << std::setprecision(std::numeric_limits<double>::digits10)
              << ns_cpu    / 1e9 << ","
              << ns_to_dev / 1e9 << ","
              << ns_kernel / 1e9 << ","
              << ns_to_host / 1e9 << "\n";

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
