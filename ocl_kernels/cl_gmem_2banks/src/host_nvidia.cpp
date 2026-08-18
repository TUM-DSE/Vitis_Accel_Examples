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

static int roundUp(int n, int m) { return ((n + m - 1) / m) * m; }

int main(int argc, char** argv) {
    if (argc < 2 || argc > 4) {
        std::cout << "Usage: " << argv[0] << " <apply_watermark_nvidia.cl> [input.bmp] [golden.bmp]" << std::endl;
        return EXIT_FAILURE;
    }

    std::ifstream f(argv[1]);
    if (!f) {
        std::cerr << "Cannot open kernel: " << argv[1] << std::endl;
        return EXIT_FAILURE;
    }
    std::string src((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());

    // Same input image and expected-output image the FPGA flow uses (see
    // description.json's launch cmd_args: -i REPO_DIR/common/data/xilinx_img.bmp
    // -c PROJECT/data/golden.bmp), read with the same BitmapInterface helper.
    std::string input_path = (argc > 2) ? argv[2] : "../../common/data/xilinx_img.bmp";
    std::string golden_path = (argc > 3) ? argv[3] : "data/golden.bmp";

    BitmapInterface inputImage(input_path.c_str());
    if (!inputImage.readBitmapFile()) {
        std::cerr << "ERROR: Unable to read input bitmap file " << input_path << std::endl;
        return EXIT_FAILURE;
    }
    BitmapInterface goldenImage(golden_path.c_str());
    if (!goldenImage.readBitmapFile()) {
        std::cerr << "ERROR: Unable to read golden bitmap file " << golden_path << std::endl;
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

    const int iterations = 500;
    cl_ulong ns_to_dev = 0, ns_kernel = 0, ns_to_host = 0;

    q.finish();
    auto t0 = std::chrono::high_resolution_clock::now();

    for (int iter = 0; iter < iterations; iter++) {
        cl::Event ev_in, ev_k, ev_out;

        q.enqueueWriteBuffer(buf_in, CL_FALSE, 0, image_size_bytes, input_image.data(), nullptr, &ev_in);
        q.finish();

        q.enqueueNDRangeKernel(kernel, cl::NullRange, global, local, nullptr, &ev_k);
        q.finish();

        q.enqueueReadBuffer(buf_out, CL_FALSE, 0, image_size_bytes, hw_result.data(), nullptr, &ev_out);
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
    std::cout << "cl_gmem_2banks_nvidia,"
              << image_size_bytes << ","
              << image_size_bytes << ","
              << iterations << ","
              << std::setprecision(std::numeric_limits<double>::digits10)
              << ns_cpu    / 1e9 << ","
              << ns_to_dev / 1e9 << ","
              << ns_kernel / 1e9 << ","
              << ns_to_host / 1e9 << "\n";

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
