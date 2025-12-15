CUDA Batch Image Histogram Equalizer
Project Description

This project implements a high-performance, GPU-accelerated image processing pipeline using CUDA. It performs Histogram Equalization on a batch of images to enhance contrast and visibility, specifically targeting low-light or low-contrast datasets (e.g., medical X-rays, dark photography).

Unlike standard grayscale equalization, this application maintains color fidelity by converting images to the YCbCr color space, applying equalization only to the Luminance (Y) channel, and reconstructing the RGB image. This prevents the "washed out" look often associated with simple global equalization.

The application is designed for scale:

    It supports batch processing (handling hundreds of images in a single run).

    It utilizes memory reuse optimization (allocating GPU memory once per batch rather than per image) to maximize throughput.

Prerequisites

To build and run this project, you need the following environment:

    OS: Linux (Arch, Ubuntu, etc.) or Windows (with Visual Studio).

    Hardware: NVIDIA GPU with Compute Capability 3.0 or higher.

    Software:

        NVIDIA CUDA Toolkit (v11.0 or higher recommended).

        C++ Compiler supporting C++17 (e.g., g++, cl.exe).

        make (optional, if using a Makefile).

Libraries:

    Standard Libraries: <filesystem> (Requires C++17), <iostream>, <vector>, <cuda_runtime.h>.

    External Libraries: stb_image.h and stb_image_write.h (Included as header-only libraries for image I/O).

Directory Structure

Ensure your project folder is organized as follows before running:

.
├── main.cu                # The main CUDA source code
├── stb_image.h            # Image loading library
├── stb_image_write.h      # Image saving library
├── README.md              # This documentation
├── input_dir/             # Place your input images here (.jpg, .png, .jpeg)
└── output_dir/            # (Created automatically) Processed images appear here

Build Instructions

You can compile the project using nvcc (NVIDIA Cuda Compiler). Since the code uses the C++17 filesystem standard, you must specify the standard version.

Run the following command in your terminal:
Bash

nvcc main.cu -o histogram_eq -std=c++17 -lm

    -o histogram_eq: Names the executable histogram_eq.

    -std=c++17: Enables filesystem support.

    -lm: Links the math library (required for fminf, fmaxf, etc.).

Run Instructions

    Prepare Data: Place your dataset images (JPEG or PNG) into a folder named input_dir in the same directory as your executable.
    Bash

mkdir -p input_dir
# Copy your images into input_dir/

Execute: Run the compiled binary:
Bash

    ./histogram_eq

    View Results: The program will automatically create a folder named output_dir and save the enhanced images there with the prefix eq_.

Implementation Details

The pipeline consists of three main stages, offloading heavy computation to the GPU:

    Preprocessing (Kernel 1):

        Images are loaded as RGB.

        A custom CUDA kernel converts RGB pixels to YCbCr on the fly.

        An Atomic Histogram calculation is performed on the Y (Luminance) channel using atomicAdd to handle parallel thread contention.

    CPU Analysis:

        The histogram (256 integers) is downloaded to the CPU.

        The Cumulative Distribution Function (CDF) is calculated serially. (This step is performed on the CPU as it is computationally trivial for 256 elements and avoids the complexity of a parallel scan algorithm).

    Postprocessing (Kernel 2):

        The CDF map is uploaded back to the GPU.

        A second kernel applies the equalization mapping to the Y channel.

        The image is converted back to RGB using the new Y value and the original Cb/Cr values to preserve color details.

Performance

    Parallelism: Each image is processed by thousands of GPU threads simultaneously (one thread per pixel).

    Memory Management: GPU memory (cudaMalloc) is allocated only once at the start of the program for the maximum supported resolution (4K). This buffer is reused for every image in the batch, eliminating the significant latency overhead of memory allocation during the loop.
