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
#include "cmdlineparser.h"
#include "xcl2.hpp"
#include <vector>
#include <iomanip>

int main(int argc, char* argv[]) {
    // Command Line Parser
    sda::utils::CmdLineParser parser;

    // Switches
    //**************//"<Full Arg>",  "<Short Arg>", "<Description>", "<Default>"
    parser.addSwitch("--xclbin_file", "-x", "input binary file string", "");
    parser.addSwitch("--input_file", "-i", "input test data file", "");
    parser.addSwitch("--compare_file", "-c", "Compare File to compare result", "");
    parser.parse(argc, argv);

    // Read settings
    auto binaryFile = parser.value("xclbin_file");
    std::string bitmapFilename = parser.value("input_file");
    std::string goldenFilename = parser.value("compare_file");

    if (argc != 7) {
        parser.printHelp();
        return EXIT_FAILURE;
    }
    cl_int err;
    cl::CommandQueue q;
    cl::Context context;
    cl::Kernel krnl_applyWatermark;

    // Read the input bit map file into memory
    BitmapInterface image(bitmapFilename.data());
    bool result = image.readBitmapFile();
    if (!result) {
        std::cerr << "ERROR:Unable to Read Input Bitmap File " << bitmapFilename.data() << std::endl;
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
    const int iterations = 500;
    uint64_t nstimestart = 0;
    uint64_t nstimeend = 0;
    uint64_t nstime_kernel = 0;
    uint64_t nstime_data_to_fpga = 0;
    uint64_t nstime_data_to_host = 0;

    std::chrono::duration<double> to_fpga_time(0);
    std::chrono::duration<double> kernel_time(0);
    std::chrono::duration<double> from_fpga_time(0);

    auto loop_start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < iterations; i++) {

        auto to_fpga_start = std::chrono::high_resolution_clock::now();
        // Copy input Image to device global memory
        OCL_CHECK(err, err = q.enqueueMigrateMemObjects({buffer_inImage}, 0 /* 0 means from host*/, nullptr, &event_data_to_fpga));
        OCL_CHECK(err, err = q.finish());
        auto to_fpga_end = std::chrono::high_resolution_clock::now();

        auto kernel_start = std::chrono::high_resolution_clock::now();
        // Launch the Kernel
        OCL_CHECK(err, err = q.enqueueTask(krnl_applyWatermark, nullptr, &event_kernel));
        OCL_CHECK(err, err = q.finish());
        auto kernel_end = std::chrono::high_resolution_clock::now();

        auto from_fpga_start = std::chrono::high_resolution_clock::now();
        // Copy Result from Device Global Memory to Host Local Memory
        OCL_CHECK(err, err = q.enqueueMigrateMemObjects({buffer_outImage}, CL_MIGRATE_MEM_OBJECT_HOST, nullptr, &event_data_to_host));
        OCL_CHECK(err, err = q.finish());
        auto from_fpga_end = std::chrono::high_resolution_clock::now();

        OCL_CHECK(err, err = event_data_to_fpga.getProfilingInfo<uint64_t>(CL_PROFILING_COMMAND_START, &nstimestart));
        OCL_CHECK(err, err = event_data_to_fpga.getProfilingInfo<uint64_t>(CL_PROFILING_COMMAND_END, &nstimeend));
        nstime_data_to_fpga += nstimeend - nstimestart;

        OCL_CHECK(err, err = event_kernel.getProfilingInfo<uint64_t>(CL_PROFILING_COMMAND_START, &nstimestart));
        OCL_CHECK(err, err = event_kernel.getProfilingInfo<uint64_t>(CL_PROFILING_COMMAND_END, &nstimeend));
        nstime_kernel += nstimeend - nstimestart;

        OCL_CHECK(err, err = event_data_to_host.getProfilingInfo<uint64_t>(CL_PROFILING_COMMAND_START, &nstimestart));
        OCL_CHECK(err, err = event_data_to_host.getProfilingInfo<uint64_t>(CL_PROFILING_COMMAND_END, &nstimeend));
        nstime_data_to_host += nstimeend - nstimestart;

        to_fpga_time += std::chrono::duration<double>(to_fpga_end - to_fpga_start);
        kernel_time += std::chrono::duration<double>(kernel_end - kernel_start);
        from_fpga_time += std::chrono::duration<double>(from_fpga_end - from_fpga_start);
    }
    // OPENCL HOST CODE AREA END
    auto loop_end   = std::chrono::high_resolution_clock::now();
    // std::chrono::duration<double> total_loop_time(0);
    auto total_loop_time = std::chrono::duration<double>(loop_end - loop_start);

    std::cout << "app_name,kernel_input_data_size,kernel_output_data_size,iterations,time_cpu,data_to_fpga_time_ocl,kernel_time_ocl,data_to_host_time_ocl\n";
    std::cout << "cl_gmem_2banks,"
              << image_size_bytes << ","
              << image_size_bytes << ","
              << iterations << ","
              << std::setprecision(std::numeric_limits<double>::digits10)
              << total_loop_time.count() << ","
              << nstime_data_to_fpga / (double)1'000'000'000 << ","
              << nstime_kernel / (double)1'000'000'000 << ","
              << nstime_data_to_host / (double)1'000'000'000 << "\n";

    // Throughputs
    std::cout << "app_name,PCIe_Wr[GB/s],Kernel[GB/s],PCIe_Rd[GB/s],FPGA_exec_time[s],FPGA_reconf_time[s]\n";
    std::cout << "cl_gmem_2banks,"
              << std::setprecision(3) << std::fixed << (image_size_bytes * iterations / to_fpga_time.count())   / 1000000000 << ","
              << std::setprecision(3) << std::fixed << (image_size_bytes * iterations * 2 / kernel_time.count()) / 1000000000 << ","
              << std::setprecision(3) << std::fixed << (image_size_bytes * iterations / from_fpga_time.count()) / 1000000000 << ","
              << total_loop_time.count() << ","
              << reconf_time.count() << ","
              << std::endl;

    // Compare Golden Image with Output image
    bool match = 1;
    // Read the golden bit map file into memory
    BitmapInterface goldenImage(goldenFilename.data());
    result = goldenImage.readBitmapFile();
    if (!result) {
        std::cerr << "ERROR:Unable to Read Golden Bitmap File " << goldenFilename.data() << std::endl;
        return EXIT_FAILURE;
    }
    if (image.getHeight() != goldenImage.getHeight() || image.getWidth() != goldenImage.getWidth()) {
        match = 0;
    } else {
        int* goldImgPtr = goldenImage.bitmap();
        for (unsigned int i = 0; i < image.numPixels(); i++) {
            if (outImage[i] != goldImgPtr[i]) {
                match = 0;
                printf("Pixel %d Mismatch Output %x and Expected %x \n", i, outImage[i], goldImgPtr[i]);
                break;
            }
        }
    }
    // Write the final image to disk
    // image.writeBitmapFile(outImage.data());

    std::cout << "TEST " << (match ? "PASSED" : "FAILED") << std::endl;
    return (match ? EXIT_SUCCESS : EXIT_FAILURE);
}
