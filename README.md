# cuda-npp-texture-processing
Image texture processing in CUDA with NPP - Creating a kernel for OpenCV "jet" style pseucolor in CUDA

  Before                   |   After                   
:-------------------------:|:-------------------------: 
![1 1 13](https://github.com/dsowsy/cuda-npp-texture-processing/assets/978118/c04d246d-e997-4d84-9b51-d42027072e49) |  ![Greenshot 2023-10-09 23 34 13](https://github.com/dsowsy/cuda-npp-texture-processing/assets/978118/c664827b-d2c8-439d-a0a7-fd2e5d694e43)

# Project description
This project leverages the USC Viterbi Volume 1: Textures image collection from the provided images.zip file. 

The images, originally in TIFF format, have been converted to PNG for easier handling.
More details about the image collection can be found [here](https://sipi.usc.edu/database/database.php?volume=textures).

1. Initially, all 66 images are loaded into the system as grayscale images using OpenCV's image loading functionality.
2. The project replicates OpenCV's jet pseudocolor technique through a custom CUDA kernel.
3. Subsequently, all loaded images are transformed into NPP image objects.
4. The custom kernel is executed on the GPU, processing all the images.
5. The pseudocolored images are then transferred back to the host system.
6. Finally the processed images are saved back to PNG format using the host's CPU resources in OpenCV.

# Problems encountered

During this project, I ran into a bunch of challenges that really tested my problem-solving skills. One big headache came from using the FreeImage library, which kept throwing exceptions and didn't allow for freeing up memory that was allocated internally. I realized I could just use OpenCV's Mat's as an allocator and then convert the image from OpenCV. I also had to spend some time tweaking the block size parameters to get things right.

On top of that, there wasn't any filtering in place for non-image files, so I ended up re-processing some files that were already done before, which was a bit of a time-waster. At certain points, I bumped into some CUDA errors and had to find ways to work around them to keep things moving.

I also dived into a bit of an IDE adventure, trying to find a workable setup for profiler and debugging. The batteries included approach with Eclipse was abandoned by NVIDIA; and after realizing this I gave  Visual Studio, Visual Studio Code, and even Ubuntu under Windows Subsystem for Linux (WSL) a shot. It was a bit of a time sink. Nonetheless even with all these hurdles I debugged the problems and got everything working as I have planned.

Additionally, being unable to get the result images out of the sandbox because of the limitations of Coursera. The support team was unwilling/unable to sent me along the resultant images that I generated
but I screen capped an example from the browser. 
