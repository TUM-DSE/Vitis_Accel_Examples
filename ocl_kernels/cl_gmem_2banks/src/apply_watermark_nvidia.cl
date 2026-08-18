// GPU analog of the FPGA watermark kernel (apply_watermark.cl). There, a
// single work-item reads 16 packed pixels at a time as one 512-bit uint16
// vector, processes them with an unrolled inner loop, and the host's
// apply_watermark.cfg routes the input and output buffers to two different
// physical DDR banks so both AXI master ports can transfer concurrently for
// full aggregate DDR bandwidth.
//
// A GPU has neither a single narrow AXI port to widen nor separate physical
// memory banks to route buffers to — it exposes one unified, very-high-
// bandwidth memory space, and one work-item per pixel, reading/writing
// plain (unpacked) ints, already gets full coalesced bandwidth from it. So
// this kernel drops both FPGA-only tricks (vector packing and bank
// routing) and is just the natural one-thread-per-pixel data-parallel form
// of the same per-channel saturating add.

#define WATERMARK_HEIGHT 16
#define WATERMARK_WIDTH 16

__constant int watermark[WATERMARK_HEIGHT][WATERMARK_WIDTH] = {
    {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
    {0, 0x0f0f0f, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0x0f0f0f, 0},
    {0, 0, 0x0f0f0f, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0x0f0f0f, 0, 0},
    {0, 0, 0, 0x0f0f0f, 0, 0, 0, 0, 0, 0, 0, 0, 0x0f0f0f, 0, 0, 0},
    {0, 0, 0, 0, 0x0f0f0f, 0, 0, 0, 0, 0, 0, 0x0f0f0f, 0, 0, 0, 0},
    {0, 0, 0, 0, 0, 0x0f0f0f, 0, 0, 0, 0, 0x0f0f0f, 0, 0, 0, 0, 0},
    {0, 0, 0, 0, 0, 0, 0x0f0f0f, 0, 0, 0x0f0f0f, 0, 0, 0, 0, 0, 0},
    {0, 0, 0, 0, 0, 0, 0, 0x0f0f0f, 0x0f0f0f, 0, 0, 0, 0, 0, 0, 0},
    {0, 0, 0, 0, 0, 0, 0, 0x0f0f0f, 0x0f0f0f, 0, 0, 0, 0, 0, 0, 0},
    {0, 0, 0, 0, 0, 0, 0x0f0f0f, 0, 0, 0x0f0f0f, 0, 0, 0, 0, 0, 0},
    {0, 0, 0, 0, 0, 0x0f0f0f, 0, 0, 0, 0, 0x0f0f0f, 0, 0, 0, 0, 0},
    {0, 0, 0, 0, 0x0f0f0f, 0, 0, 0, 0, 0, 0, 0x0f0f0f, 0, 0, 0, 0},
    {0, 0, 0, 0x0f0f0f, 0, 0, 0, 0, 0, 0, 0, 0, 0x0f0f0f, 0, 0, 0},
    {0, 0, 0x0f0f0f, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0x0f0f0f, 0, 0},
    {0, 0x0f0f0f, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0x0f0f0f, 0},
    {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
};

static int saturated_add(int x, int y) {
    int redX = x & 0xff;
    int redY = y & 0xff;
    int greenX = (x & 0xff00) >> 8;
    int greenY = (y & 0xff00) >> 8;
    int blueX = (x & 0xff0000) >> 16;
    int blueY = (y & 0xff0000) >> 16;

    int red = (redX + redY > 255) ? 255 : redX + redY;
    int green = (greenX + greenY > 255) ? 255 : greenX + greenY;
    int blue = (blueX + blueY > 255) ? 255 : blueX + blueY;

    return red | (green << 8) | (blue << 16);
}

__kernel void apply_watermark(__global const int* restrict input,
                               __global int* restrict output,
                               int width,
                               int height) {
    int x = get_global_id(0);
    int y = get_global_id(1);
    if (x >= width || y >= height) return;

    int idx = y * width + x;
    int w_idx = x % WATERMARK_WIDTH;
    int w_idy = y % WATERMARK_HEIGHT;

    output[idx] = saturated_add(input[idx], watermark[w_idy][w_idx]);
}
