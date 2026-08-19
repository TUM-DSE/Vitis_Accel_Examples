// Number of work-items per work-group used at kernel-launch time (see
// host_nvidia.cpp).
#define LOCAL_SIZE 256
// using uint16 datatype so vector size is 16
#define VECTOR_SIZE 16

// GPU analog of the FPGA wide-memory-access kernel (vadd.cl). There, a
// single work-item streams the whole vector through on-chip local buffers
// in LOCAL_MEM_SIZE-element chunks of uint16 (512-bit) words, so every
// transaction on the kernel's one AXI memory port moves 16 packed ints at
// once instead of 1.
//
// A GPU widens memory transactions the same way — via a vector load/store
// type — but does it per *thread*, not per pipeline: many work-items each
// load/add/store one uint16 word in parallel, and the memory controller
// coalesces the whole warp's wide accesses into large contiguous
// transactions. This keeps the exact "widen the transfer with uint16" idea
// from the FPGA kernel, just spread across an NDRange instead of staged
// through local memory in a single sequential loop.
__kernel __attribute__((reqd_work_group_size(LOCAL_SIZE, 1, 1))) void vadd(__global const uint16* restrict in1,
                                                                            __global const uint16* restrict in2,
                                                                            __global uint16* restrict out,
                                                                            int size) {
    // Same convention as the FPGA kernel (vadd.cl): the argument is the size in
    // ints and the kernel converts it to a uint16 count itself, so both kernels
    // can be driven by the same host code.
    int size_in16 = (size - 1) / VECTOR_SIZE + 1;

    int gid = get_global_id(0);
    if (gid < size_in16) {
        out[gid] = in1[gid] + in2[gid];
    }
}
