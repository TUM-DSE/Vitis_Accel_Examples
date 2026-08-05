// Number of elements each work-group stages on-chip per pass.
// Must match LOCAL_SIZE in host_nvidia.cpp.
#define LOCAL_SIZE 256

// Elementwise vector "increment" kernel — the GPU analog of the FPGA's
// xcl_dataflow adder kernel in adder.cl.
//
// adder.cl is a single work-item task that streams the whole vector
// through three functions — read_input(), compute_add(), write_result() —
// each working against on-chip buffer_in[]/buffer_out[] arrays, with the
// xcl_dataflow attribute pipelining the three stages against each other in
// hardware. A GPU has no equivalent single-task pipeline, but it does have
// on-chip local memory, so here every work-group stages its own
// LOCAL_SIZE-element slice through __local buffer_in[]/buffer_out[]
// arrays via the same three named steps, separated by barriers so the
// whole group finishes one stage before any work-item moves to the next.
// This keeps the same read-into-buffer / compute-into-buffer /
// write-from-buffer structure as the FPGA kernel and the same amount of
// global memory traffic (one read and one write per element) — it is just
// partitioned per work-group instead of over the whole vector in one shot.

// Read one element per work-item from Global Memory into buffer_in.
static void read_input(__global const int* in, __local int* buffer_in, int gid, int lid, int size) {
    buffer_in[lid] = (gid < size) ? in[gid] : 0;
}

// Read buffer_in, add inc, and write the result into buffer_out.
static void compute_add(__local const int* buffer_in, __local int* buffer_out, int lid, int inc) {
    buffer_out[lid] = buffer_in[lid] + inc;
}

// Read buffer_out and write the result to Global Memory.
static void write_result(__global int* out, __local const int* buffer_out, int gid, int lid, int size) {
    if (gid < size) out[gid] = buffer_out[lid];
}

__kernel __attribute__((reqd_work_group_size(LOCAL_SIZE, 1, 1))) void adder(__global const int* restrict in,
                                                                            __global int* restrict out,
                                                                            int inc,
                                                                            int size) {
    __local int buffer_in[LOCAL_SIZE];
    __local int buffer_out[LOCAL_SIZE];

    int lid = get_local_id(0);
    int gid = get_global_id(0);

    read_input(in, buffer_in, gid, lid, size);
    barrier(CLK_LOCAL_MEM_FENCE);

    compute_add(buffer_in, buffer_out, lid, inc);
    barrier(CLK_LOCAL_MEM_FENCE);

    write_result(out, buffer_out, gid, lid, size);
}
