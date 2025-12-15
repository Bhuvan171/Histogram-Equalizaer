# CUDA Histogram Equalization for RGB Images

## Project Description
This project implements a GPU-accelerated Histogram Equalization pipeline for RGB images using CUDA. Unlike standard grayscale equalization, this application preserves color information by converting images to the YCbCr color space, applying equalization only to the Luminance (Y) channel, and reconstructing the RGB image.

The pipeline performs the following steps:
1.  **Load:** Reads RGB images (JPG, PNG) from an input directory using `stb_image`.
2.  **Transfer:** Moves image data to GPU global memory.
3.  **Histogram Calculation (Kernel):** Computes the histogram of the Y channel luminance.
4.  **CDF Calculation (Host):** Calculates the Cumulative Distribution Function (CDF) on the CPU.
5.  **Equalization & Reconstruction (Kernel):** Maps old luminance values to new values using the CDF and converts YCbCr back to RGB in a single pass.
6.  **Save:** Writes the processed images to an output directory.

## Code Organization
The project directory is organized as follows:
* `main.cu`: The main source file containing C++ host code and CUDA kernels.
* `stb_image.h`: Single-header library for loading images.
* `stb_image_write.h`: Single-header library for writing images.
* `input_dir/`: Directory containing input images to be processed (user created).
* `output_dir/`: Directory where processed images will be saved (automatically created).

## Dependencies
To build and run this project, the following are required:
* **NVIDIA CUDA Toolkit**: (v10.0 or higher recommended)
* **C++ Compiler**: Must support C++17 standard (e.g., `g++` 7+, `MSVC` 19.14+, `Clang` 5+) for `std::filesystem`.
* **STB Libraries**: `stb_image.h` and `stb_image_write.h` (included in source).

## Build
Use the NVIDIA CUDA Compiler (`nvcc`) to build the application. You must specify the C++17 standard.

```bash
nvcc main.cu -o histogram_eq -std=c++17
```

## Run
1.  **Prepare Input:** Ensure a folder named `input_dir` exists in the same directory as the executable and contains `.jpg` or `.png` images.
    ```bash
    mkdir -p input_dir
    # Place images inside input_dir/
    ```

2.  **Execute:** Run the compiled binary.
    ```bash
    ./histogram_eq
    ```

3.  **Check Output:** The processed images will be saved in `output_dir/` with the prefix `eq_`.

## Troubleshooting
* **"Folder 'input_dir' does not exist"**: The program expects the input folder to be in the exact working directory where you run the command.
* **"Error writing to file"**: Ensure you have write permissions in the directory.
* **Compile Errors**: If `std::filesystem` is not found, ensure you are adding the `-std=c++17` flag during compilation.
