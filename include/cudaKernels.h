#include <cuda_runtime.h>
#include <cuda.h>


extern "C" void cuda_grayscale(unsigned char* imageInput, 
    unsigned char* imageOutput, 
    int width,
    int height,
    int colorWidthStep,
    int grayWidthStep, 
    dim3 grid, 
    dim3 block_size);

extern "C" void cuda_Sobel(unsigned char *imageInput,
    unsigned char *imageOutput, 
    int width, 
    int height,
    unsigned int maskWidth, 
    dim3 grid, 
    dim3 block_size);

extern "C" void cuda_Gaussian(unsigned char *imageInput,
    unsigned char *imageOutput, 
    int width, 
    int height,
    unsigned int maskWidth, 
    dim3 grid, 
    dim3 block_size);