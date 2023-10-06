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
#include <filesystem>

#include "jet_functions.h"


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

void processImage(const std::string& inputFilePath, const std::string& outputFilePath) {
    // Declare a host image object for an 8-bit grayscale image
    npp::ImageCPU_8u_C1 oHostSrc;
    // Load gray-scale image from disk
    npp::loadImage(inputFilePath, oHostSrc);
    // Declare a device image and copy construct from the host image,
    // i.e. upload host to device
    npp::ImageNPP_8u_C1 oDeviceSrc(oHostSrc);

    // Create struct with box-filter mask size
    NppiSize oMaskSize = {5, 5};
    NppiSize oSrcSize = {(int)oDeviceSrc.width(), (int)oDeviceSrc.height()};
    NppiPoint oSrcOffset = {0, 0};
    // Create struct with ROI size
    NppiSize oSizeROI = {(int)oDeviceSrc.width(), (int)oDeviceSrc.height()};
    // Allocate device image of appropriately reduced size
    npp::ImageNPP_8u_C1 oDeviceDst(oSizeROI.width, oSizeROI.height);
    // Set anchor point inside the mask to (oMaskSize.width / 2,
    // oMaskSize.height / 2) It should round down when odd
    NppiPoint oAnchor = {oMaskSize.width / 2, oMaskSize.height / 2};

    // Run box filter
    // NPP_CHECK_NPP(nppiFilterBoxBorder_8u_C1R(
    //    oDeviceSrc.data(), oDeviceSrc.pitch(), oSrcSize, oSrcOffset,
    //    oDeviceDst.data(), oDeviceDst.pitch(), oSizeROI, oMaskSize, oAnchor,
    //    NPP_BORDER_REPLICATE));

    // Declare a host image for the result
    // npp::ImageCPU_8u_C1 oHostDst(oDeviceDst.size());
    // And copy the device result data into it
    // oDeviceDst.copyTo(oHostDst.data(), oHostDst.pitch());

    int width = oHostSrc.width();
    int height = oHostSrc.height();

    // Allocate memory for the grayscale image on the host
    float* h_gray_image = new float[width * height];
    // Allocate memory for the RGB image on the host
    float* h_rgb_image = new float[width * height * 3];

    // Convert npp::Image to host array
    convertNppImageToHostArray(oHostSrc, h_gray_image);

    // Apply the Jet colormap
    apply_jet_colormap_wrapper(h_gray_image, h_rgb_image, width, height);

    cv::Mat cvImage(height, width, CV_32FC3, h_rgb_image);

    // Optional: Convert float values to 8-bit unsigned integer values
    cvImage.convertTo(cvImage, CV_8UC3, 255.0);

    // Save or display the image
    cv::imwrite(outputFilePath, cvImage);

    delete[] h_rgb_image;
    delete[] h_gray_image;

    nppiFree(oDeviceSrc.data());
    // nppiFree(oDeviceDst.data());
}

void processFile(const std::string& inputFilePath, const std::string& outputFilePath) {
    try {
        processImage(inputFilePath, outputFilePath);  // Call to processImage with file paths
        std::cout << "Saved processed image: " << outputFilePath << std::endl;
    } catch (npp::Exception &rException) {
        std::cerr << "Program error! The following exception occurred: \n";
        std::cerr << rException << std::endl;
        std::cerr << "Aborting." << std::endl;
        exit(EXIT_FAILURE);
    } catch (...) {
        std::cerr << "Program error! An unknown type of exception occurred. \n";
        std::cerr << "Aborting." << std::endl;
        exit(EXIT_FAILURE);
    }
}


int main(int argc, char *argv[]) {
    printf("%s Starting...\n\n", argv[0]);

    findCudaDevice(argc, (const char **)argv);

    if (printfNPPinfo(argc, argv) == false) {
        exit(EXIT_SUCCESS);
    }

    for (const auto& entry : std::filesystem::directory_iterator("./data")) {
        if (entry.is_regular_file()) {
            std::string inputFilePath = entry.path().string();
            std::string outputFilePath = inputFilePath.substr(0, inputFilePath.find_last_of('.')) + "_processed.png";
            processFile(inputFilePath, outputFilePath);
        }
    }

    return 0;
}

