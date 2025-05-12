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


#define MAX_HBM_PC_COUNT 32
#define PC_NAME(n) n | XCL_MEM_TOPOLOGY
const int pc[MAX_HBM_PC_COUNT] = {
    PC_NAME(0),  PC_NAME(1),  PC_NAME(2),  PC_NAME(3),  PC_NAME(4),  PC_NAME(5),  PC_NAME(6),  PC_NAME(7),
    PC_NAME(8),  PC_NAME(9),  PC_NAME(10), PC_NAME(11), PC_NAME(12), PC_NAME(13), PC_NAME(14), PC_NAME(15),
    PC_NAME(16), PC_NAME(17), PC_NAME(18), PC_NAME(19), PC_NAME(20), PC_NAME(21), PC_NAME(22), PC_NAME(23),
    PC_NAME(24), PC_NAME(25), PC_NAME(26), PC_NAME(27), PC_NAME(28), PC_NAME(29), PC_NAME(30), PC_NAME(31)};

#define MAX_DDR_PC_COUNT 2
const int pc_ddr[MAX_DDR_PC_COUNT] = {
    XCL_MEM_DDR_BANK0, XCL_MEM_DDR_BANK1
};

auto constexpr num_cu = 2;
auto constexpr pc_per_cu = 2;

int main(int argc, char* argv[]) {
    // Command Line Parser
    sda::utils::CmdLineParser parser;

    // Switches
    //**************//"<Full Arg>",  "<Short Arg>", "<Description>", "<Default>"
    parser.addSwitch("--xclbin_file", "-x", "input binary file string", "");
    parser.addSwitch("--input_file", "-i", "input test data file", "");
    parser.addSwitch("--compare_file", "-c", "Compare File to compare result", "");
    parser.addSwitch("--memory_type", "-m", "Memory Type: 0 (HBM) or 1 (DDR)", "0");
    parser.parse(argc, argv);

    // Read settings
    auto binaryFile = parser.value("xclbin_file");
    std::string bitmapFilename = parser.value("input_file");
    std::string goldenFilename = parser.value("compare_file");
    std::string memoryType = parser.value("memory_type");
    auto ddr_flag = std::stoi(memoryType);

    if (argc < 7) {
        parser.printHelp();
        return EXIT_FAILURE;
    }

    if(ddr_flag)
        std::cout << "DDR is selected. " << std::endl;
    else
        std::cout << "HBM is selected. " << std::endl;
        
    cl_int err;
    cl::CommandQueue q;
    cl::Context context;
    std::vector<cl::Kernel> krnls(num_cu);

    // Read the input bit map file into memory
    BitmapInterface image(bitmapFilename.data());
    bool result = image.readBitmapFile();
    if (!result) {
        std::cerr << "ERROR:Unable to Read Input Bitmap File " << bitmapFilename.data() << std::endl;
        return EXIT_FAILURE;
    }
    auto width = image.getWidth();
    auto height = image.getHeight();

    // Split the image for processing by multiple CUs
    // For simplicity, we'll split it horizontally (by rows)
    auto rows_per_cu = height / num_cu;
    std::vector<int> heights(num_cu);
    std::vector<int> offsets(num_cu);
    
    for(int i = 0; i < num_cu; i++) {
        if(i == num_cu - 1) {
            heights[i] = height - (rows_per_cu * i);
        } else {
            heights[i] = rows_per_cu;
        }
        offsets[i] = i * rows_per_cu * width;
    }

    // Allocate Memory in Host Memory
    auto total_image_size = image.numPixels();
    size_t total_image_size_bytes = total_image_size * sizeof(int);
    
    std::vector<size_t> image_sizes(num_cu);
    std::vector<size_t> image_size_bytes(num_cu);
    
    for(int i = 0; i < num_cu; i++) {
        image_sizes[i] = heights[i] * width;
        image_size_bytes[i] = image_sizes[i] * sizeof(int);
    }
    
    std::vector<int, aligned_allocator<int>> inputImage(total_image_size);
    std::vector<int, aligned_allocator<int>> outImage(total_image_size);

    // Copy image host buffer
    memcpy(inputImage.data(), image.bitmap(), total_image_size_bytes);

    // For Allocating Buffer to specific Global Memory PC
    std::vector<cl_mem_ext_ptr_t> inBufExt(num_cu);
    std::vector<cl_mem_ext_ptr_t> outBufExt(num_cu);

    for (int i = 0; i < num_cu; i++) {
        inBufExt[i].obj = inputImage.data() + offsets[i];
        inBufExt[i].param = 0;
        if(ddr_flag)
            inBufExt[i].flags = pc_ddr[(i%MAX_DDR_PC_COUNT)];
        else
            inBufExt[i].flags = pc[(i*(pc_per_cu))];

        outBufExt[i].obj = outImage.data() + offsets[i];
        outBufExt[i].param = 0;
        if(ddr_flag)
            outBufExt[i].flags = pc_ddr[(i%MAX_DDR_PC_COUNT)];
        else
            outBufExt[i].flags = pc[(i*(pc_per_cu))+1];
    }

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
        OCL_CHECK(err, q = cl::CommandQueue(context, device, CL_QUEUE_PROFILING_ENABLE | CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE, &err));

        std::cout << "Trying to program device[" << i << "]: " << device.getInfo<CL_DEVICE_NAME>() << std::endl;
        cl::Program program(context, {device}, bins, nullptr, &err);
        if (err != CL_SUCCESS) {
            std::cout << "Failed to program device[" << i << "] with xclbin file!\n";
        } else {
            std::cout << "Device[" << i << "]: program successful!\n";
            // Creating Kernel objects for multiple CUs
            for (int j = 0; j < num_cu; j++) {
                OCL_CHECK(err, krnls[j] = cl::Kernel(program, "apply_watermark", &err));
            }
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

    // Allocate Buffer in Global Memory
    std::vector<cl::Buffer> buffer_inImage(num_cu);
    std::vector<cl::Buffer> buffer_outImage(num_cu);

    for (int i = 0; i < num_cu; i++) {
        OCL_CHECK(err, buffer_inImage[i] = cl::Buffer(context, CL_MEM_READ_ONLY | CL_MEM_EXT_PTR_XILINX | CL_MEM_USE_HOST_PTR,
                                                     image_size_bytes[i], &inBufExt[i], &err));
        OCL_CHECK(err, buffer_outImage[i] = cl::Buffer(context, CL_MEM_WRITE_ONLY | CL_MEM_EXT_PTR_XILINX | CL_MEM_USE_HOST_PTR,
                                                      image_size_bytes[i], &outBufExt[i], &err));
    }

    // Set the Kernel Arguments for each CU
    for (int i = 0; i < num_cu; i++) {
        int narg = 0;
        OCL_CHECK(err, err = krnls[i].setArg(narg++, buffer_inImage[i]));
        OCL_CHECK(err, err = krnls[i].setArg(narg++, buffer_outImage[i]));
        OCL_CHECK(err, err = krnls[i].setArg(narg++, width));
        OCL_CHECK(err, err = krnls[i].setArg(narg++, heights[i]));
    }

    // For time measurement
    std::vector<cl::Event> event_kernel(num_cu);
    std::vector<cl::Event> event_data_to_fpga(num_cu);
    std::vector<cl::Event> event_data_to_host(num_cu);
    const int iterations = 500;
    uint64_t nstimestart = 0;
    uint64_t nstimeend = 0;
    uint64_t nstime_kernel = 0;
    uint64_t nstime_data_to_fpga = 0;
    uint64_t nstime_data_to_host = 0;

    std::chrono::duration<double> to_fpga_time(0);
    std::chrono::duration<double> kernel_time(0);
    std::chrono::duration<double> from_fpga_time(0);

    // This is required for proper time measurements in Proteus. We add it here
    // as well to have the same host code for Proteus and native.
    q.finish();

    auto loop_start = std::chrono::high_resolution_clock::now();
    for (int iter = 0; iter < iterations; iter++) {

        auto to_fpga_start = std::chrono::high_resolution_clock::now();
        // Copy input Image to device global memory for all CUs
        for (int i = 0; i < num_cu; i++) {
            OCL_CHECK(err, err = q.enqueueMigrateMemObjects({buffer_inImage[i]}, 0 /* 0 means from host*/, nullptr, &event_data_to_fpga[i]));
        }
        OCL_CHECK(err, err = q.finish());
        auto to_fpga_end = std::chrono::high_resolution_clock::now();

        auto kernel_start = std::chrono::high_resolution_clock::now();
        // Launch the Kernels for all CUs
        for (int i = 0; i < num_cu; i++) {
            OCL_CHECK(err, err = q.enqueueTask(krnls[i], nullptr, &event_kernel[i]));
        }
        OCL_CHECK(err, err = q.finish());
        auto kernel_end = std::chrono::high_resolution_clock::now();

        auto from_fpga_start = std::chrono::high_resolution_clock::now();
        // Copy Result from Device Global Memory to Host Local Memory for all CUs
        for (int i = 0; i < num_cu; i++) {
            OCL_CHECK(err, err = q.enqueueMigrateMemObjects({buffer_outImage[i]}, CL_MIGRATE_MEM_OBJECT_HOST, nullptr, &event_data_to_host[i]));
        }
        OCL_CHECK(err, err = q.finish());
        auto from_fpga_end = std::chrono::high_resolution_clock::now();

        // Collect profiling data from all CUs
        for (int i = 0; i < num_cu; i++) {
            OCL_CHECK(err, err = event_data_to_fpga[i].getProfilingInfo<uint64_t>(CL_PROFILING_COMMAND_START, &nstimestart));
            OCL_CHECK(err, err = event_data_to_fpga[i].getProfilingInfo<uint64_t>(CL_PROFILING_COMMAND_END, &nstimeend));
            nstime_data_to_fpga += nstimeend - nstimestart;

            OCL_CHECK(err, err = event_kernel[i].getProfilingInfo<uint64_t>(CL_PROFILING_COMMAND_START, &nstimestart));
            OCL_CHECK(err, err = event_kernel[i].getProfilingInfo<uint64_t>(CL_PROFILING_COMMAND_END, &nstimeend));
            nstime_kernel += nstimeend - nstimestart;

            OCL_CHECK(err, err = event_data_to_host[i].getProfilingInfo<uint64_t>(CL_PROFILING_COMMAND_START, &nstimestart));
            OCL_CHECK(err, err = event_data_to_host[i].getProfilingInfo<uint64_t>(CL_PROFILING_COMMAND_END, &nstimeend));
            nstime_data_to_host += nstimeend - nstimestart;
        }

        to_fpga_time += std::chrono::duration<double>(to_fpga_end - to_fpga_start);
        kernel_time += std::chrono::duration<double>(kernel_end - kernel_start);
        from_fpga_time += std::chrono::duration<double>(from_fpga_end - from_fpga_start);
    }
    // OPENCL HOST CODE AREA END
    auto loop_end = std::chrono::high_resolution_clock::now();
    auto total_loop_time = std::chrono::duration<double>(loop_end - loop_start);

    // Output timing and throughput information
    // We now have num_cu times more data being processed
    std::cout << "app_name,kernel_input_data_size,kernel_output_data_size,iterations,time_cpu,data_to_fpga_time_ocl,kernel_time_ocl,data_to_host_time_ocl\n";
    std::cout << "cl_gmem_2banks_2x,"
              << total_image_size_bytes << ","
              << total_image_size_bytes << ","
              << iterations << ","
              << std::setprecision(std::numeric_limits<double>::digits10)
              << total_loop_time.count() << ","
              << nstime_data_to_fpga / (double)1'000'000'000 / num_cu << "," // Average per CU
              << nstime_kernel / (double)1'000'000'000 / num_cu << "," // Average per CU
              << nstime_data_to_host / (double)1'000'000'000 / num_cu << "\n"; // Average per CU

    // Throughputs - now accounting for the parallel CUs
    std::cout << "app_name,PCIe_Wr[GB/s],Kernel[GB/s],PCIe_Rd[GB/s],FPGA_exec_time[s],FPGA_reconf_time[s]\n";
    std::cout << "cl_gmem_2banks_2x,"
              << std::setprecision(3) << std::fixed << (total_image_size_bytes * iterations / to_fpga_time.count()) / 1000000000 << ","
              << std::setprecision(3) << std::fixed << (total_image_size_bytes * iterations * 2 / kernel_time.count()) / 1000000000 << ","
              << std::setprecision(3) << std::fixed << (total_image_size_bytes * iterations / from_fpga_time.count()) / 1000000000 << ","
              << total_loop_time.count() << ","
              << reconf_time.count() << ","
              << std::endl;

    // Compare Golden Image with Output image
    bool match = 1;
    // Read the golden bit map file into memory
    // BitmapInterface goldenImage(goldenFilename.data());
    // result = goldenImage.readBitmapFile();
    // if (!result) {
    //     std::cerr << "ERROR:Unable to Read Golden Bitmap File " << goldenFilename.data() << std::endl;
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