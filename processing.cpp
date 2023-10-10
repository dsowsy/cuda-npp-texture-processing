/* /* Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.
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
#include <cuda_runtime.h>
#include <helper_cuda.h>
#include <helper_string.h>
#include <npp.h>
#include <string.h>

#include <filesystem>
#include <fstream>
#include <functional>
#include <iostream>
#include <opencv2/opencv.hpp>
#include <unordered_set>

#include "jet_functions.h"

// Function to convert npp::Image to a host array
void convertNppImageToHostArray(const npp::ImageCPU_8u_C1 &nppImage,
                                float *h_gray_image) {
  int width = nppImage.width();
  int height = nppImage.height();
  const Npp8u *pSrc = nppImage.data();
  int srcStep = nppImage.pitch();

  for (int y = 0; y < height; ++y) {
    for (int x = 0; x < width; ++x) {
      int srcIdx = y * srcStep + x;
      h_gray_image[y * width + x] = static_cast<float>(pSrc[srcIdx]) / 255.0f;
    }
  }
}
void convertMatToImageCPU(const cv::Mat &mat, npp::ImageCPU_8u_C1 &nppImage) {
  // Check if the input Mat is single channel and 8-bit
  if (mat.channels() != 1 || mat.depth() != CV_8U) {
    throw std::runtime_error(
        "Input Mat must be a single channel, 8-bit image.");
  }

  // Create an ImageCPU_8u_C1 object with the same dimensions as the input Mat
  npp::ImageCPU_8u_C1 tempImage(mat.cols, mat.rows);

  // Get the data pointer and pitch (step) of the input Mat
  const Npp8u *matData = mat.data;
  Npp32s matPitch = static_cast<Npp32s>(mat.step);

  // Get the data pointer and pitch of the ImageCPU_8u_C1 object
  Npp8u *nppData = tempImage.data();
  Npp32s nppPitch = static_cast<Npp32s>(tempImage.pitch());

  // Copy data from the Mat to the ImageCPU_8u_C1 object, line by line
  for (int y = 0; y < mat.rows; ++y) {
    memcpy(nppData + y * nppPitch, matData + y * matPitch,
           mat.cols * sizeof(Npp8u));
  }

  // Swap the temporary ImageCPU_8u_C1 object with the output object
  nppImage.swap(tempImage);
}

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

void processImage(const std::string &inputFilePath,
                  const std::string &outputFilePath) {
  // Declare a host image object for an 8-bit grayscale image
  npp::ImageCPU_8u_C1 oHostSrc;
  cv::Mat grayImage = cv::imread(inputFilePath, cv::IMREAD_GRAYSCALE);

  if (grayImage.empty()) {
    std::cerr << "Failed to load image: " << inputFilePath << std::endl;
    return;
  }

  // Declare a device image and copy construct from the host image,
  // i.e. upload host to device
  convertMatToImageCPU(grayImage, oHostSrc);

  int width = oHostSrc.width();
  int height = oHostSrc.height();

  float *h_gray_image = new float[width * height * 3];
  float *h_rgb_image = new float[width * height * 3];

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

  cvImage.deallocate();
  grayImage.deallocate();
}

void processFile(const std::filesystem::directory_entry &entry) {
  std::string inputFilePath = entry.path().string();
  std::string outputFilePath =
      inputFilePath.substr(0, inputFilePath.find_last_of('.')) +
      "_processed.png";

  try {
    processImage(inputFilePath,
                 outputFilePath);  // Call to processImage with file paths
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

void forEachFile(const std::string &dirPath,
                 std::function<void(const std::filesystem::directory_entry &)>
                     fileProcessor) {
  // Capture the directory listing once
  std::vector<std::filesystem::directory_entry> dirListing;
  for (const auto &entry : std::filesystem::directory_iterator(dirPath)) {
    if (entry.is_regular_file() && entry.path().extension() == ".png") {
      dirListing.push_back(entry);
    }
  }
  // Now work off the static list
  for (const auto &entry : dirListing) {
    fileProcessor(entry);
  }
}

int main(int argc, char *argv[]) {
  printf("%s Starting...\n\n", argv[0]);

  findCudaDevice(argc, (const char **)argv);

  if (printfNPPinfo(argc, argv) == false) {
    exit(EXIT_SUCCESS);
  }

  forEachFile("./data", processFile);
  return 0;
}
