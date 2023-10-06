/* Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 *  * Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 *  * Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the distribution.
 *  * Neither the name of NVIDIA CORPORATION nor the names of its
 *    contributors may be used to endorse or promote products derived
 *    from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
 * EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
 * PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
 * CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
 * EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
 * PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
 * PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
 * OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

#if defined(WIN32) || defined(_WIN32) || defined(WIN64) || defined(_WIN64)
#define WINDOWS_LEAN_AND_MEAN
#define NOMINMAX
#include <windows.h>
#pragma warning(disable : 4819)
#endif

#include <Exceptions.h>
#include <ImageIO.h>
#include <ImagesCPU.h>
#include <ImagesNPP.h>
#include <string.h>
#include <fstream>
#include <iostream>
#include <cuda_runtime.h>
#include <npp.h>
#include <helper_cuda.h>
#include <helper_string.h>
#include <opencv2/opencv.hpp>

#include "jet_functions.h"

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


// Function to convert npp::Image to a host array
void convertNppImageToHostArray(const npp::ImageCPU_8u_C1 &nppImage, float* h_gray_image) {
    int width = nppImage.width();
    int height = nppImage.height();
    const Npp8u* pSrc = nppImage.data();
    int srcStep = nppImage.pitch();

    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            int srcIdx = y * srcStep + x;
            h_gray_image[y * width + x] = static_cast<float>(pSrc[srcIdx]) / 255.0f;
        }
    }
}

/*
__global__ void calculate_metrics(float* histogram, float* complexity, float* entropy, int histSize) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < histSize) {
        // Assume totalPixels is the total number of pixels in the image
        float probability = histogram[idx] / totalPixels;
        
        // Histogram Complexity: Assume a simple measure like the sum of squared bin values
        atomicAdd(complexity, histogram[idx] * histogram[idx]);
        
        // Entropy
        if (probability > 0)
            atomicAdd(entropy, -probability * log2f(probability));
    }
}

__global__ void normalize_and_average(float* complexity, float* entropy, float* average) {
    // Normalize (assuming maxComplexity and maxEntropy are the maximum possible values)
    *complexity /= maxComplexity;
    *entropy /= maxEntropy;
    
    // Average
    *average = (*complexity + *entropy) / 2.0f;
}
*/

bool printfNPPinfo(int argc, char *argv[]) {
  const NppLibraryVersion *libVer = nppGetLibVersion();

  printf("NPP Library Version %d.%d.%d\n", libVer->major, libVer->minor,
         libVer->build);

  int driverVersion, runtimeVersion;
  cudaDriverGetVersion(&driverVersion);
  cudaRuntimeGetVersion(&runtimeVersion);

  printf("  CUDA Driver  Version: %d.%d\n", driverVersion / 1000,
         (driverVersion % 100) / 10);
  printf("  CUDA Runtime Version: %d.%d\n", runtimeVersion / 1000,
         (runtimeVersion % 100) / 10);

  // Min spec is SM 1.0 devices
  bool bVal = checkCudaCapabilities(1, 0);
  return bVal;
}

int main(int argc, char *argv[]) {
  printf("%s Starting...\n\n", argv[0]);

  try {
    std::string sFilename;
    char *filePath;

    findCudaDevice(argc, (const char **)argv);

    if (printfNPPinfo(argc, argv) == false) {
      exit(EXIT_SUCCESS);
    }

    if (checkCmdLineFlag(argc, (const char **)argv, "input")) {
      getCmdLineArgumentString(argc, (const char **)argv, "input", &filePath);
    } else {
      filePath = sdkFindFilePath("1.1.01.png", argv[0]);
      std::cout << "Filepath: " << filePath << std::endl;
    }


    if (filePath) {
      sFilename = filePath;
    } else {
      sFilename = "teapot512.pgm";
    }

      std::cout << "sFilename: " << sFilename << std::endl;

    // if we specify the filename at the command line, then we only test
    // sFilename[0].
    int file_errors = 0;
    std::ifstream infile(sFilename.data(), std::ifstream::in);

    if (infile.good()) {
      std::cout << "processing opened: <" << sFilename.data()
                << "> successfully!" << std::endl;
      file_errors = 0;
      infile.close();
    } else {
      std::cout << "processing unable to open: <" << sFilename.data() << ">"
                << std::endl;
      file_errors++;
      infile.close();
    }

    if (file_errors > 0) {
      exit(EXIT_FAILURE);
    }

    std::string sResultFilename = sFilename;

    std::string::size_type dot = sResultFilename.rfind('.');

    if (dot != std::string::npos) {
      sResultFilename = sResultFilename.substr(0, dot);
    }

    sResultFilename += "_processing.png";

    if (checkCmdLineFlag(argc, (const char **)argv, "output")) {
      char *outputFilePath;
      getCmdLineArgumentString(argc, (const char **)argv, "output",
                               &outputFilePath);
      sResultFilename = outputFilePath;
    }

    // declare a host image object for an 8-bit grayscale image
    npp::ImageCPU_8u_C1 oHostSrc;
    // load gray-scale image from disk
    npp::loadImage(sFilename, oHostSrc);
    // declare a device image and copy construct from the host image,
    // i.e. upload host to device
    npp::ImageNPP_8u_C1 oDeviceSrc(oHostSrc);

    // create struct with box-filter mask size
    NppiSize oMaskSize = {5, 5};

    NppiSize oSrcSize = {(int)oDeviceSrc.width(), (int)oDeviceSrc.height()};
    NppiPoint oSrcOffset = {0, 0};

    // create struct with ROI size
    NppiSize oSizeROI = {(int)oDeviceSrc.width(), (int)oDeviceSrc.height()};
    // allocate device image of appropriately reduced size
    npp::ImageNPP_8u_C1 oDeviceDst(oSizeROI.width, oSizeROI.height);
    // set anchor point inside the mask to (oMaskSize.width / 2,
    // oMaskSize.height / 2) It should round down when odd
    NppiPoint oAnchor = {oMaskSize.width / 2, oMaskSize.height / 2};

    // run box filter
    NPP_CHECK_NPP(nppiFilterBoxBorder_8u_C1R(
        oDeviceSrc.data(), oDeviceSrc.pitch(), oSrcSize, oSrcOffset,
        oDeviceDst.data(), oDeviceDst.pitch(), oSizeROI, oMaskSize, oAnchor,
        NPP_BORDER_REPLICATE));

    // declare a host image for the result
    npp::ImageCPU_8u_C1 oHostDst(oDeviceDst.size());
    // and copy the device result data into it
    oDeviceDst.copyTo(oHostDst.data(), oHostDst.pitch());

    // Assume histogram is already calculated and is in device memory
    float* d_histogram;
    float *d_complexity, *d_entropy, *d_average;
    
    // Allocate memory for metrics
    cudaMalloc(&d_complexity, sizeof(float));
    cudaMalloc(&d_entropy, sizeof(float));
    cudaMalloc(&d_average, sizeof(float));
    
    // Initialize metrics to 0
    cudaMemset(d_complexity, 0, sizeof(float));
    cudaMemset(d_entropy, 0, sizeof(float));
    
    // Launch kernels
    /*
    int histSize = 256;
    int blocks = (histSize + 255) / 256;
    calculate_metrics<<<blocks, 256>>>(d_histogram, d_complexity, d_entropy, histSize);
    normalize_and_average<<<1, 1>>>(d_complexity, d_entropy, d_average);
    */

    int width = oHostDst.width();
    int height = oHostDst.height();

    // Allocate memory for the grayscale image on the host
    float* h_gray_image = new float[width * height];
    // Allocate memory for the RGB image on the host
    float* h_rgb_image = new float[width * height * 3];


   // Convert npp::Image to host array
    convertNppImageToHostArray(oHostDst, h_gray_image);

    apply_jet_colormap_wrapper(h_gray_image, h_rgb_image, width, height);

    cv::Mat cvImage(height, width, CV_32FC3, h_rgb_image);

    // Optional: Convert float values to 8-bit unsigned integer values
    cvImage.convertTo(cvImage, CV_8UC3, 255.0);

    // Save or display the image
    cv::imwrite(sResultFilename, cvImage);

    delete [] h_rgb_image;
    delete [] h_gray_image;

    // cv::Mat cvImage(height, width, CV_8UC1, oHostDst.data(), oHostDst.pitch());
    // cv::imwrite(sResultFilename, cvImage);


    std::cout << "Saved processed image: " << sResultFilename << std::endl;

    nppiFree(oDeviceSrc.data());
    nppiFree(oDeviceDst.data());

    exit(EXIT_SUCCESS);
  } catch (npp::Exception &rException) {
    std::cerr << "Program error! The following exception occurred: \n";
    std::cerr << rException << std::endl;
    std::cerr << "Aborting." << std::endl;

    exit(EXIT_FAILURE);
  } catch (...) {
    std::cerr << "Program error! An unknow type of exception occurred. \n";
    std::cerr << "Aborting." << std::endl;

    exit(EXIT_FAILURE);
    return -1;
  }

  return 0;
}
