#include <stdio.h>
#include <string>
#include <vector>
#include <iostream>
#include <algorithm> 
#include <filesystem> 
#include <cuda_runtime.h>

// STB Image Libraries
#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

namespace fs = std::filesystem;


// Case-insensitive check to find .jpg, .JPG, .png, etc.
bool is_image_file(const std::string& filename) {
    std::string lower = filename;
    std::transform(lower.begin(), lower.end(), lower.begin(), ::tolower);
    return (lower.find(".jpg") != std::string::npos || 
            lower.find(".jpeg") != std::string::npos || 
            lower.find(".png") != std::string::npos);
}

//kernels
__device__ unsigned char get_luminance(unsigned char r, unsigned char g, unsigned char b) {
    return (unsigned char)(0.299f * r + 0.587f * g + 0.114f * b);
}

__device__ void ycbcr_to_rgb(unsigned char y, unsigned char cb, unsigned char cr, 
                             unsigned char* r, unsigned char* g, unsigned char* b) {
    float y_val = (float)y;
    float cb_val = (float)cb - 128.0f;
    float cr_val = (float)cr - 128.0f;

    float r_val = y_val + 1.402f * cr_val;
    float g_val = y_val - 0.344136f * cb_val - 0.714136f * cr_val;
    float b_val = y_val + 1.772f * cb_val;

    *r = (unsigned char)fminf(fmaxf(r_val, 0.0f), 255.0f);
    *g = (unsigned char)fminf(fmaxf(g_val, 0.0f), 255.0f);
    *b = (unsigned char)fminf(fmaxf(b_val, 0.0f), 255.0f);
}

__global__ void calculate_y_histogram(unsigned char *image, int *histogram, int width, int height) {
    int idx = (blockIdx.x * blockDim.x + threadIdx.x);
    if (idx < width * height) {
        int rgb_idx = idx * 3;
        unsigned char y = get_luminance(image[rgb_idx], image[rgb_idx+1], image[rgb_idx+2]);
        atomicAdd(&histogram[y], 1);
    }
}

__global__ void equalize_y_and_reconstruct(unsigned char *input, unsigned char *output, int *cdf, int width, int height) {
    int idx = (blockIdx.x * blockDim.x + threadIdx.x);
    if (idx < width * height) {
        int rgb_idx = idx * 3;
        unsigned char r = input[rgb_idx];
        unsigned char g = input[rgb_idx+1];
        unsigned char b = input[rgb_idx+2];

        float r_f = (float)r; float g_f = (float)g; float b_f = (float)b;
        unsigned char y_old = get_luminance(r, g, b);
        unsigned char cb = (unsigned char)((-0.1687f * r_f - 0.3313f * g_f + 0.5f * b_f) + 128.0f);
        unsigned char cr = (unsigned char)((0.5f * r_f - 0.4187f * g_f - 0.0813f * b_f) + 128.0f);

        unsigned char y_new = (unsigned char)cdf[y_old];
        
        unsigned char new_r, new_g, new_b;
        ycbcr_to_rgb(y_new, cb, cr, &new_r, &new_g, &new_b);

        output[rgb_idx] = new_r;
        output[rgb_idx+1] = new_g;
        output[rgb_idx+2] = new_b;
    }
}

//to compute cdf
void calculate_cdf_cpu(int *h_hist, int *h_cdf, int total_pixels) {
    long temp_cdf[256];
    long current_sum = 0;
    for(int i=0; i<256; i++) { current_sum += h_hist[i]; temp_cdf[i] = current_sum; }
    
    long min_val = 0;
    for(int i=0; i<256; i++) { if(temp_cdf[i] > 0) { min_val = temp_cdf[i]; break; } }

    for(int i=0; i<256; i++) {
        float normalized = (float)(temp_cdf[i] - min_val) / (float)(total_pixels - min_val);
        h_cdf[i] = (int)(normalized * 255.0f);
        if(h_cdf[i] < 0) h_cdf[i] = 0; else if(h_cdf[i] > 255) h_cdf[i] = 255;
    }
}

int main() {
    std::string input_dir = "input_dir";
    std::string output_dir = "output_dir";

    //directories from where we read and write files
    if (!fs::exists(input_dir)) {
        printf("CRITICAL ERROR: Folder '%s' does not exist.\n", input_dir.c_str());
        printf("Please create it and put images inside.\n");
        return 1;
    }
    
    if (!fs::exists(output_dir)) {
        fs::create_directory(output_dir);
        printf("Created output directory: %s\n", output_dir.c_str());
    }

    // 2. Allocate GPU Memory Once (Max 4K)
    size_t max_pixels = 4000 * 4000; 
    size_t max_img_size = max_pixels * 3 * sizeof(unsigned char);
    size_t hist_size = 256 * sizeof(int);

    unsigned char *d_img_in, *d_img_out;
    int *d_hist, *d_cdf;

    cudaMalloc((void**)&d_img_in, max_img_size);
    cudaMalloc((void**)&d_img_out, max_img_size);
    cudaMalloc((void**)&d_hist, hist_size);
    cudaMalloc((void**)&d_cdf, hist_size);

    int h_hist[256]; 
    int h_cdf[256];

    printf("Starting processing from '%s' to '%s'...\n", input_dir.c_str(), output_dir.c_str());
    int processed_count = 0;

    //main loop
    for (const auto& entry : fs::directory_iterator(input_dir)) {
        std::string filename = entry.path().filename().string();
        std::string full_path = entry.path().string();


        int width, height, channels;
        // Force RGB (3 channels)
        unsigned char *h_img_in = stbi_load(full_path.c_str(), &width, &height, &channels, 3);
        
        if (!h_img_in) {
            printf("  [!] Failed to load: %s\n", filename.c_str());
            continue;
        }

        

        // --- GPU PIPELINE ---
        cudaMemset(d_hist, 0, hist_size);
        cudaMemcpy(d_img_in, h_img_in, width * height * 3, cudaMemcpyHostToDevice);

        int blockSize = 256;
        int gridSize = (width * height + blockSize - 1) / blockSize;

        calculate_y_histogram<<<gridSize, blockSize>>>(d_img_in, d_hist, width, height);
        
        cudaMemcpy(h_hist, d_hist, hist_size, cudaMemcpyDeviceToHost);
        calculate_cdf_cpu(h_hist, h_cdf, width * height);
        cudaMemcpy(d_cdf, h_cdf, hist_size, cudaMemcpyHostToDevice);

        equalize_y_and_reconstruct<<<gridSize, blockSize>>>(d_img_in, d_img_out, d_cdf, width, height);
        
        unsigned char *h_img_out = (unsigned char*)malloc(width * height * 3);
        cudaMemcpy(h_img_out, d_img_out, width * height * 3, cudaMemcpyDeviceToHost);

        // saving to output directory
        std::string out_path = output_dir + "/eq_" + filename + ".png"; // Force png extension
        
        int success = stbi_write_png(out_path.c_str(), width, height, 3, h_img_out, width * 3);
        
        if (success) {
            printf("  [+] Saved: %s\n", out_path.c_str());
            processed_count++;
        } else {
            printf("  [!] Error writing to file: %s\n", out_path.c_str());
        }

        stbi_image_free(h_img_in);
        free(h_img_out);
    }

    // Cleanup
    cudaFree(d_img_in); cudaFree(d_img_out);
    cudaFree(d_hist); cudaFree(d_cdf);
    
    if (processed_count == 0) {
        printf("\nWARNING: No images were processed. Check if 'input_images' is empty or has valid .jpg/.png files.\n");
    } else {
        printf("\nSuccess! Processed %d images.\n", processed_count);
    }
    
    return 0;
}