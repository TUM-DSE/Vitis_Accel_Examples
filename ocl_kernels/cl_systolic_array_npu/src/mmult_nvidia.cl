// Matrix multiplication C = A x B, with A (a_row x a_col), B (a_col x b_col)
// and C (a_row x b_col), row major. A and B are int8, C is an int32
// accumulator, matching the RKNPU's INT8_MM_INT8_TO_INT32 matmul mode.
//
// Must match TILE_SIZE in host_nvidia.cpp.
#define TILE_SIZE 16

// Local-memory tiled implementation.
//
// On the FPGA, mmult is a systolic array: every element of C has its own
// multiply-accumulate unit, and each cycle one column of A and one row of B
// flow through the whole array. A GPU has no equivalent of that fixed
// dataflow, so instead each work-group cooperatively stages one
// TILE_SIZE x TILE_SIZE tile of A and of B into fast local memory, and every
// work-item of the group then reuses those tiles for its TILE_SIZE
// multiply-accumulates instead of going to global memory for each of them.
__kernel __attribute__((reqd_work_group_size(TILE_SIZE, TILE_SIZE, 1))) void mmult(
    __global const char* a, // Read-Only Matrix A (int8)
    __global const char* b, // Read-Only Matrix B (int8)
    __global int* c,        // Output Result (int32 accumulator)
    int a_row,
    int a_col,
    int b_col) {
    __local char tileA[TILE_SIZE][TILE_SIZE];
    __local char tileB[TILE_SIZE][TILE_SIZE];

    int row = get_global_id(0);
    int col = get_global_id(1);
    int lr = get_local_id(0);
    int lc = get_local_id(1);

    int sum = 0;
    int numTiles = (a_col + TILE_SIZE - 1) / TILE_SIZE;

    for (int t = 0; t < numTiles; t++) {
        int aCol = t * TILE_SIZE + lc;
        int bRow = t * TILE_SIZE + lr;

        tileA[lr][lc] = (row < a_row && aCol < a_col) ? a[row * a_col + aCol] : 0;
        tileB[lr][lc] = (bRow < a_col && col < b_col) ? b[bRow * b_col + col] : 0;

        barrier(CLK_LOCAL_MEM_FENCE);

        for (int k = 0; k < TILE_SIZE; k++) sum += (int)tileA[lr][k] * (int)tileB[k][lc];

        barrier(CLK_LOCAL_MEM_FENCE);
    }

    if (row < a_row && col < b_col) c[row * b_col + col] = sum;
}
