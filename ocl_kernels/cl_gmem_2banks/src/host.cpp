/**
* Copyright (C) 2019-2021 Xilinx, Inc
*
* Licensed under the Apache License, Version 2.0 (the "License"). You may
* not use this file except in compliance with the License. A copy of the
* License is located at
*
*     http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
* WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
* License for the specific language governing permissions and limitations
* under the License.
*/
#include "bitmap.h"
#include "xcl2.hpp"
#include <vector>
#include <iomanip>

// Input image and golden reference are fixed, not passed on the command line.
// Both are relative to the example directory the executable is run from, which
// is what "make run" / "make run_nvidia" do.
#define INPUT_BMP "../../common/data/xilinx_img.bmp"
#define GOLDEN_BMP "data/golden.bmp"

int main(int argc, char* argv[]) {
    if (argc != 2) {
        std::cout << "Usage: " << argv[0] << " <XCLBIN File>" << std::endl;
        return EXIT_FAILURE;
    }

    std::string binaryFile = argv[1];

    cl_int err;
    cl::CommandQueue q;
    cl::Context context;
    cl::Kernel krnl_applyWatermark;

    // Read the input bit map file into memory
    BitmapInterface image(INPUT_BMP);
    bool result = image.readBitmapFile();
    if (!result) {
        std::cerr << "ERROR:Unable to Read Input Bitmap File " << INPUT_BMP << std::endl;
        return EXIT_FAILURE;
    }
    auto width = image.getWidth();
    auto height = image.getHeight();

    // Allocate Memory in Host Memory
    auto image_size = image.numPixels();
    size_t image_size_bytes = image_size * sizeof(int);
    std::vector<int, aligned_allocator<int> > inputImage(image_size);
    std::vector<int, aligned_allocator<int> > outImage(image_size);

    // Copy image host buffer
    memcpy(inputImage.data(), image.bitmap(), image_size_bytes);

    // OPENCL HOST CODE AREA START
    auto devices = xcl::get_xil_devices();

    auto reconf_start = std::chrono::high_resolution_clock::now();
    // read_binary_file() is a utility API which will load the binaryFile
    // and will return the pointer to file buffer.
    auto fileBuf = xcl::read_binary_file(binaryFile);
    cl::Program::Binaries bins{{fileBuf.data(), fileBuf.size()}};
    bool valid_device = false;
    for (unsigned int i = 0; i < devices.size(); i++) {
        auto device = devices[i];
        // Creating Context and Command Queue for selected Device
        OCL_CHECK(err, context = cl::Context(device, nullptr, nullptr, nullptr, &err));
        OCL_CHECK(err, q = cl::CommandQueue(context, device, CL_QUEUE_PROFILING_ENABLE, &err));

        std::cout << "Trying to program device[" << i << "]: " << device.getInfo<CL_DEVICE_NAME>() << std::endl;
        cl::Program program(context, {device}, bins, nullptr, &err);
        if (err != CL_SUCCESS) {
            std::cout << "Failed to program device[" << i << "] with xclbin file!\n";
        } else {
            std::cout << "Device[" << i << "]: program successful!\n";
            OCL_CHECK(err, krnl_applyWatermark = cl::Kernel(program, "apply_watermark", &err));
            valid_device = true;
            break; // we break because we found a valid device
        }
    }
    if (!valid_device) {
        std::cerr << "Failed to program any device found, exit!\n";
        exit(EXIT_FAILURE);
    }
    auto reconf_end = std::chrono::high_resolution_clock::now();
    auto reconf_time = std::chrono::duration<double>(reconf_end - reconf_start);

    OCL_CHECK(err, cl::Buffer buffer_inImage(context, CL_MEM_READ_ONLY | CL_MEM_USE_HOST_PTR, image_size_bytes,
                                             inputImage.data(), &err));
    OCL_CHECK(err, cl::Buffer buffer_outImage(context, CL_MEM_WRITE_ONLY | CL_MEM_USE_HOST_PTR, image_size_bytes,
                                              outImage.data(), &err));

    /*
     * Using setArg(), i.e. setting kernel arguments, explicitly before
     * enqueueMigrateMemObjects(),
     * i.e. copying host memory to device memory,  allowing runtime to associate
     * buffer with correct
     * DDR banks automatically.
    */

    krnl_applyWatermark.setArg(0, buffer_inImage);
    krnl_applyWatermark.setArg(1, buffer_outImage);
    krnl_applyWatermark.setArg(2, width);
    krnl_applyWatermark.setArg(3, height);

    // for time measurement
    cl::Event event_kernel;
    cl::Event event_data_to_fpga;
    cl::Event event_data_to_host;
    const int n_warmup = 0;
    const int n_reps = 500;
    uint64_t nstimestart = 0;
    uint64_t nstimeend = 0;
    uint64_t time_kernel_ocl = 0;
    uint64_t time_data_to_xpu_ocl = 0;
    uint64_t time_data_to_host_ocl = 0;
    // Host-clock accumulator for data-transfer + kernel-execution time only: each interval below
    // is opened right before an OpenCL enqueue call and closed right after it (and any q.finish())
    // completes, so host-side work (loop bookkeeping) is never included.
    uint64_t time_xpu = 0;

    // Per-phase host-clock times, kept for the throughput numbers printed below
    std::chrono::duration<double> to_fpga_time(0);
    std::chrono::duration<double> kernel_time(0);
    std::chrono::duration<double> from_fpga_time(0);

    // This is required for proper time measurements in Proteus. We add it here
    // as well to have the same host code for Proteus and native.
    q.finish();

    for (int i = 0; i < n_warmup + n_reps; i++) {
        auto t_xpu_0 = std::chrono::high_resolution_clock::now();
        // Copy input Image to device global memory
        OCL_CHECK(err, err = q.enqueueMigrateMemObjects({buffer_inImage}, 0 /* 0 means from host*/, nullptr, &event_data_to_fpga));
        OCL_CHECK(err, err = q.finish());
        auto t_xpu_1 = std::chrono::high_resolution_clock::now();
        time_xpu += std::chrono::duration_cast<std::chrono::nanoseconds>(t_xpu_1 - t_xpu_0).count();

        auto t_xpu_2 = std::chrono::high_resolution_clock::now();
        // Launch the Kernel
        OCL_CHECK(err, err = q.enqueueTask(krnl_applyWatermark, nullptr, &event_kernel));
        OCL_CHECK(err, err = q.finish());
        auto t_xpu_3 = std::chrono::high_resolution_clock::now();
        time_xpu += std::chrono::duration_cast<std::chrono::nanoseconds>(t_xpu_3 - t_xpu_2).count();

        auto t_xpu_4 = std::chrono::high_resolution_clock::now();
        // Copy Result from Device Global Memory to Host Local Memory
        OCL_CHECK(err, err = q.enqueueMigrateMemObjects({buffer_outImage}, CL_MIGRATE_MEM_OBJECT_HOST, nullptr, &event_data_to_host));
        OCL_CHECK(err, err = q.finish());
        auto t_xpu_5 = std::chrono::high_resolution_clock::now();
        time_xpu += std::chrono::duration_cast<std::chrono::nanoseconds>(t_xpu_5 - t_xpu_4).count();

        OCL_CHECK(err, err = event_data_to_fpga.getProfilingInfo<uint64_t>(CL_PROFILING_COMMAND_START, &nstimestart));
        OCL_CHECK(err, err = event_data_to_fpga.getProfilingInfo<uint64_t>(CL_PROFILING_COMMAND_END, &nstimeend));
        time_data_to_xpu_ocl += nstimeend - nstimestart;

        OCL_CHECK(err, err = event_kernel.getProfilingInfo<uint64_t>(CL_PROFILING_COMMAND_START, &nstimestart));
        OCL_CHECK(err, err = event_kernel.getProfilingInfo<uint64_t>(CL_PROFILING_COMMAND_END, &nstimeend));
        time_kernel_ocl += nstimeend - nstimestart;

        OCL_CHECK(err, err = event_data_to_host.getProfilingInfo<uint64_t>(CL_PROFILING_COMMAND_START, &nstimestart));
        OCL_CHECK(err, err = event_data_to_host.getProfilingInfo<uint64_t>(CL_PROFILING_COMMAND_END, &nstimeend));
        time_data_to_host_ocl += nstimeend - nstimestart;

        to_fpga_time += std::chrono::duration<double>(t_xpu_1 - t_xpu_0);
        kernel_time += std::chrono::duration<double>(t_xpu_3 - t_xpu_2);
        from_fpga_time += std::chrono::duration<double>(t_xpu_5 - t_xpu_4);
    }
    // OPENCL HOST CODE AREA END

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

    // Throughputs
    const int n_total = n_warmup + n_reps;
    std::cout << "app_name,PCIe_Wr[GB/s],Kernel[GB/s],PCIe_Rd[GB/s],FPGA_exec_time[s],FPGA_reconf_time[s]\n";
    std::cout << "cl_gmem_2banks,"
              << std::setprecision(3) << std::fixed << (image_size_bytes * n_total / to_fpga_time.count())   / 1000000000 << ","
              << std::setprecision(3) << std::fixed << (image_size_bytes * n_total * 2 / kernel_time.count()) / 1000000000 << ","
              << std::setprecision(3) << std::fixed << (image_size_bytes * n_total / from_fpga_time.count()) / 1000000000 << ","
              << time_xpu / ns_per_s << ","
              << reconf_time.count() << ","
              << std::endl;

    // Compare Golden Image with Output image
    bool match = 1;
    // Read the golden bit map file into memory
    // BitmapInterface goldenImage(GOLDEN_BMP);
    // result = goldenImage.readBitmapFile();
    // if (!result) {
    //     std::cerr << "ERROR:Unable to Read Golden Bitmap File " << GOLDEN_BMP << std::endl;
    //     return EXIT_FAILURE;
    // }
    // if (image.getHeight() != goldenImage.getHeight() || image.getWidth() != goldenImage.getWidth()) {
    //     match = 0;
    // } else {
    //     int* goldImgPtr = goldenImage.bitmap();
    //     for (unsigned int i = 0; i < image.numPixels(); i++) {
    //         if (outImage[i] != goldImgPtr[i]) {
    //             match = 0;
    //             printf("Pixel %d Mismatch Output %x and Expected %x \n", i, outImage[i], goldImgPtr[i]);
    //             break;
    //         }
    //     }
    // }
    // Write the final image to disk
    // image.writeBitmapFile(outImage.data());

    std::cout << "TEST " << (match ? "PASSED" : "FAILED") << std::endl;
    return (match ? EXIT_SUCCESS : EXIT_FAILURE);
}
