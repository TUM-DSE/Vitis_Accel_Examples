#define CL_HPP_TARGET_OPENCL_VERSION 200
#define CL_HPP_MINIMUM_OPENCL_VERSION 110
#include <CL/cl2.hpp>

#include <algorithm>
#include <chrono>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <numeric>
#include <random>
#include <vector>

#define N_COEFF 11
#define SIGNAL_SIZE (128 * 1024) // * sizeof(int) = 512 KB
// Must match LOCAL_SIZE in fir_nvidia.cl
#define LOCAL_SIZE 256

using std::vector;

// Finite Impulse Response filter, computed on the host as the reference result
void fir_sw(vector<int>& output, const vector<int>& signal, const vector<int>& coeff) {
    auto out_iter = begin(output);
    auto rsignal_iter = signal.rend() - 1;

    int i = 0;
    while (rsignal_iter != signal.rbegin() - 1) {
        int elements = std::min((int)coeff.size(), i++);
        *(out_iter++) = std::inner_product(begin(coeff), begin(coeff) + elements, rsignal_iter--, 0);
    }
}

int gen_random() {
    static std::default_random_engine e;
    static std::uniform_int_distribution<int> dist(0, 100);
    return dist(e);
}

int main(int argc, char** argv) {
    if (argc != 2) {
        std::cout << "Usage: " << argv[0] << " <fir_nvidia.cl>" << std::endl;
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
    cl::Kernel krnl_naive(program, "fir_naive");
    cl::Kernel krnl_sr(program, "fir_shift_register");

    const int signal_size = SIGNAL_SIZE;
    const size_t size_in_bytes = signal_size * sizeof(int);
    const size_t coeff_size_in_bytes = N_COEFF * sizeof(int);

    vector<int> signal(signal_size);
    vector<int> coeff = {53, 0, -91, 0, 313, 500, 313, 0, -91, 0, 53};
    vector<int> gold(signal_size, 0);
    vector<int> out_naive(signal_size, 0);
    vector<int> out_sr(signal_size, 0);
    std::generate(signal.begin(), signal.end(), gen_random);

    fir_sw(gold, signal, coeff);

    cl::Buffer buf_signal_A(context, CL_MEM_READ_ONLY, size_in_bytes);
    cl::Buffer buf_coeff_A(context, CL_MEM_READ_ONLY, coeff_size_in_bytes);
    cl::Buffer buf_output_A(context, CL_MEM_WRITE_ONLY, size_in_bytes);
    cl::Buffer buf_signal_B(context, CL_MEM_READ_ONLY, size_in_bytes);
    cl::Buffer buf_coeff_B(context, CL_MEM_READ_ONLY, coeff_size_in_bytes);
    cl::Buffer buf_output_B(context, CL_MEM_WRITE_ONLY, size_in_bytes);

    krnl_naive.setArg(0, buf_output_A);
    krnl_naive.setArg(1, buf_signal_A);
    krnl_naive.setArg(2, buf_coeff_A);
    krnl_naive.setArg(3, signal_size);

    krnl_sr.setArg(0, buf_output_B);
    krnl_sr.setArg(1, buf_signal_B);
    krnl_sr.setArg(2, buf_coeff_B);
    krnl_sr.setArg(3, signal_size);

    const size_t global_size = ((signal_size + LOCAL_SIZE - 1) / LOCAL_SIZE) * LOCAL_SIZE;
    cl::NDRange global(global_size);
    cl::NDRange local(LOCAL_SIZE);

    const int iterations = 1000;
    cl_ulong ns_to_dev = 0, ns_kernel = 0, ns_to_host = 0;

    q.finish();
    auto t0 = std::chrono::high_resolution_clock::now();

    // Running fir_naive for iterations/2 runs
    for (int iter = 0; iter < iterations / 2; iter++) {
        cl::Event ev_s, ev_c, ev_k, ev_r;

        q.enqueueWriteBuffer(buf_signal_A, CL_FALSE, 0, size_in_bytes, signal.data(), nullptr, &ev_s);
        q.enqueueWriteBuffer(buf_coeff_A, CL_FALSE, 0, coeff_size_in_bytes, coeff.data(), nullptr, &ev_c);
        q.finish();

        q.enqueueNDRangeKernel(krnl_naive, cl::NullRange, global, local, nullptr, &ev_k);
        q.finish();

        q.enqueueReadBuffer(buf_output_A, CL_FALSE, 0, size_in_bytes, out_naive.data(), nullptr, &ev_r);
        q.finish();

        cl_ulong s, e;
        ev_s.getProfilingInfo(CL_PROFILING_COMMAND_START, &s);
        ev_s.getProfilingInfo(CL_PROFILING_COMMAND_END,   &e);
        ns_to_dev += e - s;
        ev_c.getProfilingInfo(CL_PROFILING_COMMAND_START, &s);
        ev_c.getProfilingInfo(CL_PROFILING_COMMAND_END,   &e);
        ns_to_dev += e - s;

        ev_k.getProfilingInfo(CL_PROFILING_COMMAND_START, &s);
        ev_k.getProfilingInfo(CL_PROFILING_COMMAND_END,   &e);
        ns_kernel += e - s;

        ev_r.getProfilingInfo(CL_PROFILING_COMMAND_START, &s);
        ev_r.getProfilingInfo(CL_PROFILING_COMMAND_END,   &e);
        ns_to_host += e - s;
    }

    // Running fir_shift_register for iterations/2 runs
    for (int iter = 0; iter < iterations / 2; iter++) {
        cl::Event ev_s, ev_c, ev_k, ev_r;

        q.enqueueWriteBuffer(buf_signal_B, CL_FALSE, 0, size_in_bytes, signal.data(), nullptr, &ev_s);
        q.enqueueWriteBuffer(buf_coeff_B, CL_FALSE, 0, coeff_size_in_bytes, coeff.data(), nullptr, &ev_c);
        q.finish();

        q.enqueueNDRangeKernel(krnl_sr, cl::NullRange, global, local, nullptr, &ev_k);
        q.finish();

        q.enqueueReadBuffer(buf_output_B, CL_FALSE, 0, size_in_bytes, out_sr.data(), nullptr, &ev_r);
        q.finish();

        cl_ulong s, e;
        ev_s.getProfilingInfo(CL_PROFILING_COMMAND_START, &s);
        ev_s.getProfilingInfo(CL_PROFILING_COMMAND_END,   &e);
        ns_to_dev += e - s;
        ev_c.getProfilingInfo(CL_PROFILING_COMMAND_START, &s);
        ev_c.getProfilingInfo(CL_PROFILING_COMMAND_END,   &e);
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
    std::cout << "cl_shift_register_nvidia,"
              << size_in_bytes + coeff_size_in_bytes << ","
              << size_in_bytes << ","
              << iterations << ","
              << std::setprecision(std::numeric_limits<double>::digits10)
              << ns_cpu    / 1e9 << ","
              << ns_to_dev / 1e9 << ","
              << ns_kernel / 1e9 << ","
              << ns_to_host / 1e9 << "\n";

    int match = 0;
    for (int i = 0; i < signal_size; i++) {
        if (out_naive[i] != gold[i]) {
            std::cout << "Error: fir_naive mismatch at i=" << i
                      << " CPU=" << gold[i] << " GPU=" << out_naive[i] << std::endl;
            match = 1;
            break;
        }
    }
    if (!match) {
        for (int i = 0; i < signal_size; i++) {
            if (out_sr[i] != gold[i]) {
                std::cout << "Error: fir_shift_register mismatch at i=" << i
                          << " CPU=" << gold[i] << " GPU=" << out_sr[i] << std::endl;
                match = 1;
                break;
            }
        }
    }
    std::cout << "TEST " << (match ? "FAILED" : "PASSED") << std::endl;
    return match ? EXIT_FAILURE : EXIT_SUCCESS;
}
