#include <CL/opencl.h>
#include <stdio.h>
#include <stdlib.h>
#include <vector>
#include <fstream>
#include <iostream>
#include <chrono>
#include "aocl_utils.h"

#define DATA_SIZE (64 * 1024)

using namespace aocl_utils;

int main(int argc, char** argv) {
    if (argc != 2) {
        printf("Usage: %s <vector_add.aocx>\n", argv[0]);
        return 1;
    }

    std::vector<int> a(DATA_SIZE, 10);
    std::vector<int> b(DATA_SIZE, 32);
    std::vector<int> result(DATA_SIZE, 0);

    cl_int status;

    cl_platform_id platform = findPlatform("Intel(R) FPGA Emulation");
    if (!platform) {
        printf("ERROR: Unable to find Intel FPGA OpenCL platform.\n");
        return 1;
    }

    cl_uint num_devices;
    scoped_array<cl_device_id> devices(getDevices(platform, CL_DEVICE_TYPE_ALL, &num_devices));
    cl_device_id device = devices[0];

    cl_context context = clCreateContext(NULL, 1, &device, &oclContextCallback, NULL, &status);
    checkError(status, "clCreateContext");

    cl_command_queue queue = clCreateCommandQueue(context, device, CL_QUEUE_PROFILING_ENABLE, &status);
    checkError(status, "clCreateCommandQueue");

    std::string binary_file = argv[1];
    cl_program program = createProgramFromBinary(context, binary_file.c_str(), &device, 1);
    status = clBuildProgram(program, 0, NULL, "", NULL, NULL);
    checkError(status, "clBuildProgram");

    cl_kernel kernel = clCreateKernel(program, "vector_add", &status);
    checkError(status, "clCreateKernel");

    cl_mem buf_a = clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(int) * DATA_SIZE, NULL, &status);
    cl_mem buf_b = clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(int) * DATA_SIZE, NULL, &status);
    cl_mem buf_result = clCreateBuffer(context, CL_MEM_WRITE_ONLY, sizeof(int) * DATA_SIZE, NULL, &status);

    clEnqueueWriteBuffer(queue, buf_a, CL_TRUE, 0, sizeof(int) * DATA_SIZE, a.data(), 0, NULL, NULL);
    clEnqueueWriteBuffer(queue, buf_b, CL_TRUE, 0, sizeof(int) * DATA_SIZE, b.data(), 0, NULL, NULL);

    status |= clSetKernelArg(kernel, 0, sizeof(cl_mem), &buf_result);
    status |= clSetKernelArg(kernel, 1, sizeof(cl_mem), &buf_a);
    status |= clSetKernelArg(kernel, 2, sizeof(cl_mem), &buf_b);
    int n = DATA_SIZE;
    status |= clSetKernelArg(kernel, 3, sizeof(int), &n);
    checkError(status, "clSetKernelArg");

    size_t global_size = 1;
    cl_event kernel_event;
    clEnqueueNDRangeKernel(queue, kernel, 1, NULL, &global_size, NULL, 0, NULL, &kernel_event);
    clFinish(queue);

    clEnqueueReadBuffer(queue, buf_result, CL_TRUE, 0, sizeof(int) * DATA_SIZE, result.data(), 0, NULL, NULL);

    int match = 0;
    for (int i = 0; i < DATA_SIZE; ++i) {
        if (result[i] != a[i] + b[i]) {
            printf("Mismatch at %d: %d + %d != %d\n", i, a[i], b[i], result[i]);
            match = 1;
            break;
        }
    }

    printf("TEST %s\n", (match ? "FAILED" : "PASSED"));

    clReleaseMemObject(buf_a);
    clReleaseMemObject(buf_b);
    clReleaseMemObject(buf_result);
    clReleaseKernel(kernel);
    clReleaseProgram(program);
    clReleaseCommandQueue(queue);
    clReleaseContext(context);

    return (match ? 1 : 0);
}
