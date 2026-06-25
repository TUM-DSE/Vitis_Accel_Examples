#define TILE_SIZE 16

__kernel __attribute__((reqd_work_group_size(TILE_SIZE, TILE_SIZE, 1)))
void mmult(__global const int* a,
           __global const int* b,
           __global int* c,
           int a_row,
           int a_col,
           int b_col)
{
    __local int tileA[TILE_SIZE][TILE_SIZE];
    __local int tileB[TILE_SIZE][TILE_SIZE];

    int row = get_global_id(0);
    int col = get_global_id(1);
    int lr  = get_local_id(0);
    int lc  = get_local_id(1);

    int sum = 0;
    int numTiles = (a_col + TILE_SIZE - 1) / TILE_SIZE;

    for (int t = 0; t < numTiles; t++) {
        int aCol = t * TILE_SIZE + lc;
        int bRow = t * TILE_SIZE + lr;

        tileA[lr][lc] = (row < a_row && aCol < a_col) ? a[row * a_col + aCol] : 0;
        tileB[lr][lc] = (bRow < a_col && col < b_col) ? b[bRow * b_col + col] : 0;

        barrier(CLK_LOCAL_MEM_FENCE);

        for (int k = 0; k < TILE_SIZE; k++)
            sum += tileA[lr][k] * tileB[k][lc];

        barrier(CLK_LOCAL_MEM_FENCE);
    }

    if (row < a_row && col < b_col)
        c[row * b_col + col] = sum;
}
