// Number of coefficient components
#define N_COEFF 11

// Naive FIR: one work-item per output sample, reading the sliding window
// directly from global memory. This is the direct GPU analog of fir_naive
// in fir.cl, parallelized across the signal instead of looping serially.
__kernel void fir_naive(__global int* restrict output_r,
                         __global const int* restrict signal_r,
                         __global const int* restrict coeff,
                         int signal_length) {
    int j = get_global_id(0);
    if (j >= signal_length) return;

    int acc = 0;
    int lim = min(j, N_COEFF - 1);
    for (int i = 0; i <= lim; i++) {
        acc += signal_r[j - i] * coeff[i];
    }
    output_r[j] = acc;
}

// FIR using a local-memory staged window.
//
// On the FPGA, fir_shift_register keeps the last N_COEFF samples in a
// completely-partitioned register array so every tap is available in the
// same cycle. A GPU has no equivalent per-lane shift register, so instead
// each work-group cooperatively stages its slice of the signal (plus the
// N_COEFF-1 sample halo needed at the left edge) into fast local memory
// once, and every work-item in the group then reuses that local copy for
// its N_COEFF tap reads instead of hitting global memory N_COEFF times.
// Must match LOCAL_SIZE in host_nvidia.cpp.
#define LOCAL_SIZE 256

__kernel __attribute__((reqd_work_group_size(LOCAL_SIZE, 1, 1))) void fir_shift_register(
    __global int* restrict output_r,
    __global const int* restrict signal_r,
    __global const int* restrict coeff,
    int signal_length) {
    __local int coeff_reg[N_COEFF];
    __local int window[LOCAL_SIZE + N_COEFF - 1];

    int lid = get_local_id(0);
    int group_base = get_group_id(0) * LOCAL_SIZE;
    int j = group_base + lid;

    if (lid < N_COEFF) coeff_reg[lid] = coeff[lid];

    // window[k] holds signal_r[group_base - (N_COEFF - 1) + k]
    int idx = group_base + lid - (N_COEFF - 1);
    window[lid] = (idx >= 0 && idx < signal_length) ? signal_r[idx] : 0;
    if (lid < N_COEFF - 1) {
        int idx2 = group_base + LOCAL_SIZE + lid - (N_COEFF - 1);
        window[LOCAL_SIZE + lid] = (idx2 < signal_length) ? signal_r[idx2] : 0;
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    if (j >= signal_length) return;

    int acc = 0;
    int lim = min(j, N_COEFF - 1);
    for (int i = 0; i <= lim; i++) {
        acc += window[lid + (N_COEFF - 1) - i] * coeff_reg[i];
    }
    output_r[j] = acc;
}
