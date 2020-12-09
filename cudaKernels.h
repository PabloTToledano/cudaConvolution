#include <cuda_runtime.h>
#include <cuda.h>


extern "C" void cuda_grayscale(unsigned char* input, 
    unsigned char* output, 
    int width,
    int height,
    int colorWidthStep,
    int grayWidthStep, 
    dim3 blocks, 
    dim3 block_size);

extern "C" void cuda_Sobel(unsigned char *input,
    unsigned char *output, 
    int width, 
    int height,
    unsigned int maskWidth, 
    dim3 blocks, 
    dim3 block_size);