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

/*******************************************************************************
Description:
    This example demonstrate to utilized both DDR and full bandwidth using
    watermark Application. In Watermark application, kernel has to apply a fixed
    watermark (Here is it Xilinx First Character 'X') into a given Image and
    write the output image.
    Inside Host code, input image is placed into Bank0 and Kernel will read
    input Image and write the output image to Bank1.
    To utilized the both Banks fully Kernel code do burst read input image with
    full datawidth of 512 and do the burst write of output image with full
    datawidth of 512.
    As Kernel is accessing Sequentially from both the DDR, so kernel with get
the
    Best memory access bandwidth from both DDRs and will do watermark with good
    performance.
*******************************************************************************/

#define CHANNELS 3
#define WATERMARK_HEIGHT 16
#define WATERMARK_WIDTH 16


#define TYPE uint16
#define DATA_SIZE 16

int saturatedAdd(int x, int y);

__kernel void apply_watermark(__global const TYPE* __restrict input,
                              __global TYPE* __restrict output,
                              int width,
                              int height) {

    int watermark[WATERMARK_HEIGHT][WATERMARK_WIDTH] = {
        {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
        {0, 0x0f0f0f, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0x0f0f0f, 0},
        {0, 0, 0x0f0f0f, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0x0f0f0f, 0, 0},
        {0, 0, 0, 0x0f0f0f, 0, 0, 0, 0, 0, 0, 0, 0, 0x0f0f0f, 0, 0, 0},
        {0, 0, 0, 0, 0x0f0f0f, 0, 0, 0, 0, 0, 0, 0x0f0f0f, 0, 0, 0, 0},
        {0, 0, 0, 0, 0, 0x0f0f0f, 0, 0, 0, 0, 0x0f0f0f, 0, 0, 0, 0, 0},
        {0, 0, 0, 0, 0, 0, 0x0f0f0f, 0, 0x0f0f0f, 0, 0, 0, 0, 0, 0, 0},
        {0, 0, 0, 0, 0, 0, 0, 0x0f0f0f, 0x0f0f0f, 0, 0, 0, 0, 0, 0, 0},
        {0, 0, 0, 0, 0, 0, 0, 0x0f0f0f, 0x0f0f0f, 0, 0, 0, 0, 0, 0, 0},
        {0, 0, 0, 0, 0, 0, 0x0f0f0f, 0, 0x0f0f0f, 0, 0, 0, 0, 0, 0, 0},
        {0, 0, 0, 0, 0, 0x0f0f0f, 0, 0, 0, 0, 0x0f0f0f, 0, 0, 0, 0, 0},
        {0, 0, 0, 0, 0x0f0f0f, 0, 0, 0, 0, 0, 0, 0x0f0f0f, 0, 0, 0, 0},
        {0, 0, 0, 0x0f0f0f, 0, 0, 0, 0, 0, 0, 0, 0, 0x0f0f0f, 0, 0, 0},
        {0, 0, 0x0f0f0f, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0x0f0f0f, 0, 0},
        {0, 0x0f0f0f, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0x0f0f0f, 0},
        {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
    };

    uint imageSize = width * height;
    uint size = ((imageSize - 1) / DATA_SIZE) + 1;

    
    for (uint idx = 0, x = 0, y = 0; idx < size; ++idx, x += DATA_SIZE) {
        TYPE tmp = input[idx];

        if (x >= width) {
            x = x - width;
            ++y;
        }

        #pragma unroll
        for (int i = 0; i < DATA_SIZE; i++) {
            uint tmp_x = x + i;
            uint tmp_y = y;

            if (tmp_x >= width) {
                tmp_x -= width;
                tmp_y += 1;
            }

            uint w_idy = tmp_y % WATERMARK_HEIGHT;
            uint w_idx = tmp_x % WATERMARK_WIDTH;
            tmp[i] = saturatedAdd(tmp[i], watermark[w_idy][w_idx]);
        }

        output[idx] = tmp;
    }
}

int saturatedAdd(int x, int y) {

    // Red Channel
    int redX = x & 0xff;
    int redY = y & 0xff;
    int redOutput;

    // Green Channel
    int greenX = (x & 0xff00) >> 8;
    int greenY = (y & 0xff00) >> 8;
    int greenOutput;

    // Blue Channel
    int blueX = (x & 0xff0000) >> 16;
    int blueY = (y & 0xff0000) >> 16;
    int blueOutput;

    redOutput = (redX + redY > 255) ? 255 : redX + redY;

    greenOutput = (greenX + greenY > 255) ? 255 : greenX + greenY;

    blueOutput = (blueX + blueY > 255) ? 255 : blueX + blueY;

    int combinedOutput = 0;
    combinedOutput |= redOutput;
    combinedOutput |= (greenOutput << 8);
    combinedOutput |= (blueOutput << 16);
    return combinedOutput;
}