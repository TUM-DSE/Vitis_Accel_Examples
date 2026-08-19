#define CL_HPP_TARGET_OPENCL_VERSION 200
#define CL_HPP_MINIMUM_OPENCL_VERSION 110
#include <CL/cl2.hpp>
#include "bitmap.h"

#include <chrono>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <vector>

// Work-group geometry used at kernel-launch time below.
#define LOCAL_SIZE_X 16
#define LOCAL_SIZE_Y 16

// Input image and golden reference are fixed, not passed on the command line.
// Both are relative to the example directory the executable is run from, which
// is what "make run" / "make run_nvidia" do.
#define INPUT_BMP "../../common/data/xilinx_img.bmp"
#define GOLDEN_BMP "data/golden.bmp"

static int roundUp(int n, int m) { return ((n + m - 1) / m) * m; }

int main(int argc, char** argv) {
    if (argc != 2) {
        std::cout << "Usage: " << argv[0] << " <apply_watermark_nvidia.cl>" << std::endl;
        return EXIT_FAILURE;
    }

    std::ifstream f(argv[1]);
    if (!f) {
        std::cerr << "Cannot open kernel: " << argv[1] << std::endl;
        return EXIT_FAILURE;
    }
    std::string src((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());

    // Same input image and expected-output image the FPGA flow uses, read with
    // the same BitmapInterface helper.
    BitmapInterface inputImage(INPUT_BMP);
    if (!inputImage.readBitmapFile()) {
        std::cerr << "ERROR: Unable to read input bitmap file " << INPUT_BMP << std::endl;
        return EXIT_FAILURE;
    }
    BitmapInterface goldenImage(GOLDEN_BMP);
    if (!goldenImage.readBitmapFile()) {
        std::cerr << "ERROR: Unable to read golden bitmap file " << GOLDEN_BMP << std::endl;
        return EXIT_FAILURE;
    }
    if (inputImage.getWidth() != goldenImage.getWidth() || inputImage.getHeight() != goldenImage.getHeight()) {
        std::cerr << "ERROR: Input and golden bitmap dimensions differ" << std::endl;
        return EXIT_FAILURE;
    }

    const int width = inputImage.getWidth();
    const int height = inputImage.getHeight();
    const size_t num_pixels = inputImage.numPixels();
    const size_t image_size_bytes = sizeof(int) * num_pixels;

    std::vector<int> input_image(inputImage.bitmap(), inputImage.bitmap() + num_pixels);
    std::vector<int> golden_result(goldenImage.bitmap(), goldenImage.bitmap() + num_pixels);
    std::vector<int> hw_result(num_pixels, 0);

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
    cl::Kernel kernel(program, "apply_watermark");

    cl::Buffer buf_in(context, CL_MEM_READ_ONLY, image_size_bytes);
    cl::Buffer buf_out(context, CL_MEM_WRITE_ONLY, image_size_bytes);

    kernel.setArg(0, buf_in);
    kernel.setArg(1, buf_out);
    kernel.setArg(2, width);
    kernel.setArg(3, height);

    cl::NDRange global(roundUp(width, LOCAL_SIZE_X), roundUp(height, LOCAL_SIZE_Y));
    cl::NDRange local(LOCAL_SIZE_X, LOCAL_SIZE_Y);

    const int n_warmup = 0;
    const int n_reps = 500;
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
        q.enqueueWriteBuffer(buf_in, CL_FALSE, 0, image_size_bytes, input_image.data(), nullptr, &ev_in);
        q.finish();
        auto t_xpu_1 = std::chrono::high_resolution_clock::now();
        time_xpu += std::chrono::duration_cast<std::chrono::nanoseconds>(t_xpu_1 - t_xpu_0).count();

        auto t_xpu_2 = std::chrono::high_resolution_clock::now();
        q.enqueueNDRangeKernel(kernel, cl::NullRange, global, local, nullptr, &ev_k);
        q.finish();
        auto t_xpu_3 = std::chrono::high_resolution_clock::now();
        time_xpu += std::chrono::duration_cast<std::chrono::nanoseconds>(t_xpu_3 - t_xpu_2).count();

        auto t_xpu_4 = std::chrono::high_resolution_clock::now();
        q.enqueueReadBuffer(buf_out, CL_FALSE, 0, image_size_bytes, hw_result.data(), nullptr, &ev_out);
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
              << "cl_gmem_2banks,"
              << image_size_bytes << ","
              << image_size_bytes << ","
              << n_warmup << ","
              << n_reps << ","
              << time_xpu / ns_per_s << ","
              << time_data_to_xpu_ocl / ns_per_s << ","
              << time_kernel_ocl / ns_per_s << ","
              << time_data_to_host_ocl / ns_per_s
              << "\n";

    int match = 0;
    for (size_t i = 0; i < num_pixels; i++) {
        if (hw_result[i] != golden_result[i]) {
            std::cout << "Error: Pixel mismatch at i=" << i
                      << " Golden=" << golden_result[i]
                      << " GPU=" << hw_result[i] << std::endl;
            match = 1;
            break;
        }
    }
    std::cout << "TEST " << (match ? "FAILED" : "PASSED") << std::endl;
    return match ? EXIT_FAILURE : EXIT_SUCCESS;
}
