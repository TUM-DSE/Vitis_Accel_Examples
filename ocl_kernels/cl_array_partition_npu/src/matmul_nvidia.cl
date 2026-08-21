// Matrix multiplication C = A x B, with A, B and C square matrices of
// dimension (size x size), row major.
//
// Must match TILE_SIZE in host_nvidia.cpp.
#define TILE_SIZE 16

// Local-memory tiled implementation.
//
// On the FPGA, matmul_partition partitions B, C and temp_sum completely on
// their second dimension so that a whole row of taps is available in the same
// cycle. A GPU has no equivalent of that partitioning, so instead each
// work-group cooperatively stages one TILE_SIZE x TILE_SIZE tile of A and of B
// into fast local memory, and every work-item of the group then reuses those
// tiles for its TILE_SIZE multiply-accumulates instead of going to global
// memory for each of them.
__kernel __attribute__((reqd_work_group_size(TILE_SIZE, TILE_SIZE, 1))) void matmul_partition(
    const __global int* in1, // Read-Only Matrix 1
    const __global int* in2, // Read-Only Matrix 2
    __global int* out,       // Output Result
    int size) {
    __local int tileA[TILE_SIZE][TILE_SIZE];
    __local int tileB[TILE_SIZE][TILE_SIZE];

    int row = get_global_id(0);
    int col = get_global_id(1);
    int lr = get_local_id(0);
    int lc = get_local_id(1);

    int sum = 0;
    int numTiles = (size + TILE_SIZE - 1) / TILE_SIZE;

    for (int t = 0; t < numTiles; t++) {
        int aCol = t * TILE_SIZE + lc;
        int bRow = t * TILE_SIZE + lr;

        tileA[lr][lc] = (row < size && aCol < size) ? in1[row * size + aCol] : 0;
        tileB[lr][lc] = (bRow < size && col < size) ? in2[bRow * size + col] : 0;

        barrier(CLK_LOCAL_MEM_FENCE);

        for (int k = 0; k < TILE_SIZE; k++) sum += tileA[lr][k] * tileB[k][lc];

        barrier(CLK_LOCAL_MEM_FENCE);
    }

    if (row < size && col < size) out[row * size + col] = sum;
}
