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
#include <fcntl.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <sys/mman.h>
#include <unistd.h>
#include <cstring>
#include <filesystem>

#include <iostream>

#include "bitmap.h"

BitmapInterface::BitmapInterface(const char* f) : filename(f) {
    core = nullptr;
    dib = nullptr;
    image = nullptr;

    magicNumber = 0;
    fileSize = 0;
    offsetOfImage = 0;

    sizeOfDIB = 0;
    sizeOfImage = 0;

    height = -1;
    width = -1;
}

BitmapInterface::~BitmapInterface() {
    if (core != nullptr) delete[] core;
    if (dib != nullptr) delete[] dib;
    if (image != nullptr) delete[] image;
}

bool BitmapInterface::readBitmapFile() {
    // First, open the bitmap file
    int fd;
    unsigned int fileSize;

    fd = open(filename, O_RDONLY);
    if (fd < 0) {
        std::cerr << "Cannot read image file " << filename << std::endl;
        return false;
    }

    struct stat st;
    if (fstat(fd, &st) == -1) {
        std::cerr << "fstat failed\n";
        return false;
    }

    char* file = (char*)mmap(NULL, st.st_size, PROT_READ, MAP_PRIVATE, fd, 0);
    if (file == MAP_FAILED) {
        std::cerr << "Failed to mmap file " << filename << "\n";
        return false;
    }

    unsigned int offset = 0;

    core = new char[14];
    std::memcpy(core, file, 14);
    offset+=14;

    magicNumber = (*(unsigned short*)(&(core[0])));
    fileSize = (*(unsigned int*)(&(core[2])));
    offsetOfImage = (*(unsigned int*)(&(core[10])));

    // Just read in the DIB, but don't process it
    sizeOfDIB = offsetOfImage - 14;
    dib = new char[sizeOfDIB];
    std::memcpy(dib, &file[offset], sizeOfDIB);
    offset+=sizeOfDIB;

    width = (*(int*)(&(dib[4])));
    height = (*(int*)(&(dib[8])));

    sizeOfImage = fileSize - 14 - sizeOfDIB;
    int numPixels = sizeOfImage / 3; // RGB

    image = new int[numPixels];

    for (int i = 0; i < numPixels; ++i) {
        // Use an integer for every pixel even though we might not need that
        //  much space (padding 0 bits in the rest of the integer)
        image[i] = 0;
        std::memcpy(&(image[i]), &file[offset], 3);
        offset+=3;
    }

    return true;
}

bool BitmapInterface::writeBitmapFile(int* otherImage) {
    int fd;
    fd = open("output.bmp", O_WRONLY | O_CREAT, 0644);

    if (fd < 0) {
        std::cerr << "Cannot open output.bmp for writing!" << std::endl;
        return false;
    }

    write(fd, core, 14);
    write(fd, dib, sizeOfDIB);

    int numPixels = sizeOfImage / 3;

    int* outputImage = otherImage != nullptr ? otherImage : image;

    for (int i = 0; i < numPixels; ++i) {
        write(fd, &(outputImage[i]), 3);
    }

    return true;
}
