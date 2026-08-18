// Number of work-items per work-group used at kernel-launch time (see
// host_nvidia.cpp).
#define LOCAL_SIZE 256

// GPU analog of the FPGA burst_rw kernel (vadd.cl). There, a single
// work-item streams the whole array through a small on-chip "burstbuffer"
// in BURSTBUFFERSIZE-element chunks, using xcl_pipeline_loop on
// sequential-address read/compute-write loops so the Vitis HLS backend
// infers a hardware AXI burst transaction from that one narrow AXI master
// port.
//
// A GPU has no single, narrow memory port to coax into bursting: with one
// work-item per element, consecutive threads in a warp already issue
// consecutive addresses, and the memory controller coalesces them into wide
// transactions automatically. So this is the plain, un-staged, data-parallel
// form of the same in-place increment — no local buffer or pipeline
// attribute needed to get burst-equivalent behavior.
__kernel __attribute__((reqd_work_group_size(LOCAL_SIZE, 1, 1))) void vadd(__global int* restrict a,
                                                                            int size,
                                                                            int inc_value) {
    int gid = get_global_id(0);
    if (gid < size) {
        a[gid] = a[gid] + inc_value;
    }
}
