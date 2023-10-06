// jet_functions.cu

#include "jet_functions.h"

__device__ void jet_colormap(float gray_value, float &r, float &g, float &b) {
    if (gray_value < 0.125) {
        r = 0.0;
        g = 0.0;
        b = 4.0 * (gray_value + 0.125);
    } else if (gray_value < 0.375) {
        r = 0.0;
        g = 4.0 * (gray_value - 0.125);
        b = 1.0;
    } else if (gray_value < 0.625) {
        r = 4.0 * (gray_value - 0.375);
        g = 1.0;
        b = 1.0 - 4.0 * (gray_value - 0.375);
    } else if (gray_value < 0.875) {
        r = 1.0;
        g = 1.0 - 4.0 * (gray_value - 0.625);
        b = 0.0;
    } else {
        r = 1.0 - 4.0 * (gray_value - 0.875);
        g = 0.0;
        b = 0.0;
    }
}


__global__ void apply_jet_colormap(const float* gray_image, float* rgb_image, int width, int height) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < width && y < height) {
        int gray_idx = y * width + x;
        int rgb_idx = gray_idx * 3;

        float gray_value = gray_image[gray_idx];
        float r, g, b;
        jet_colormap(gray_value, r, g, b);

        rgb_image[rgb_idx + 0] = r;
        rgb_image[rgb_idx + 1] = g;
        rgb_image[rgb_idx + 2] = b;
    }
}

// Your host function to launch the CUDA kernel:
void apply_jet_colormap_host(const float* gray_image, float* rgb_image, int width, int height) {
    dim3 block_dim(16, 16);
    dim3 grid_dim((width + block_dim.x - 1) / block_dim.x, (height + block_dim.y - 1) / block_dim.y);

    apply_jet_colormap<<<grid_dim, block_dim>>>(gray_image, rgb_image, width, height);
    cudaDeviceSynchronize();
}


void apply_jet_colormap_wrapper(const float* h_gray_image, float* h_rgb_image, int width, int height) {
    // Allocate memory for the grayscale image on the device
    float* d_gray_image;
    cudaMalloc(&d_gray_image, width * height * sizeof(float));

    // Copy grayscale image data to the device
    cudaMemcpy(d_gray_image, h_gray_image, width * height * sizeof(float), cudaMemcpyHostToDevice);

    // Allocate memory for the RGB image on the device
    float* d_rgb_image;
    cudaMalloc(&d_rgb_image, width * height * 3 * sizeof(float));

    // Define block and grid dimensions
    dim3 block_dim(16, 16);
    dim3 grid_dim((width + block_dim.x - 1) / block_dim.x, (height + block_dim.y - 1) / block_dim.y);

    // Apply the Jet colormap
    apply_jet_colormap<<<grid_dim, block_dim>>>(d_gray_image, d_rgb_image, width, height);

    // Check for any errors during the kernel launch
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("CUDA Error: %s\n", cudaGetErrorString(err));
    }

    // Synchronize device
    cudaDeviceSynchronize();

    // Copy RGB image data back to the host
    cudaMemcpy(h_rgb_image, d_rgb_image, width * height * 3 * sizeof(float), cudaMemcpyDeviceToHost);

    // Free allocated memory on the device
    cudaFree(d_gray_image);
    cudaFree(d_rgb_image);
}