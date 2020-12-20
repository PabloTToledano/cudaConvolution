#include "include/cudaKernels.h"
#include<iostream>
#include<cstdio>
#include<opencv2/core/core.hpp>
#include<opencv2/highgui/highgui.hpp>
#include <opencv2/opencv.hpp>
#include <cuda_runtime.h>
#include <cuda.h>

static inline void _safe_cuda_call(cudaError err, const char* msg, const char* file_name, const int line_number)
{
	if(err!=cudaSuccess)
	{
		std::cerr << msg << "\nFile: " << file_name << "\nLine: " << line_number << "\nError: " << cudaGetErrorString(err);
		std::cin.get();
		exit(EXIT_FAILURE);
	}
}

#define SAFE_CALL(call,msg) _safe_cuda_call((call),(msg),__FILE__,__LINE__)


void convertToGray(const cv::Mat& input, cv::Mat& output){

	const int inputBytes = input.step * input.rows;
	const int outputBytes = output.step * output.rows;

	unsigned char *deviceInput, *deviceOutput;

	SAFE_CALL(cudaMalloc<unsigned char>(&deviceInput,inputBytes),"CUDA Malloc Error");
	SAFE_CALL(cudaMalloc<unsigned char>(&deviceOutput,outputBytes),"CUDA Malloc Error");

	SAFE_CALL(cudaMemcpy(deviceInput,input.ptr(),inputBytes,cudaMemcpyHostToDevice),"CUDA Memcpy Host To Device Error");

	const dim3 block(16,16);
	const dim3 grid((input.cols + block.x - 1)/block.x, (input.rows + block.y - 1)/block.y);

	cuda_grayscale(deviceInput,deviceOutput,input.cols,input.rows,input.step,output.step,grid,block);

	SAFE_CALL(cudaDeviceSynchronize(),"Kernel Launch Error");

	SAFE_CALL(cudaMemcpy(output.ptr(),deviceOutput,outputBytes,cudaMemcpyDeviceToHost),"CUDA Memcpy Host To Device Error");

	SAFE_CALL(cudaFree(deviceInput),"CUDA Free Error");
	SAFE_CALL(cudaFree(deviceOutput),"CUDA Free Error");
}

void sobelFilter(const cv::Mat& input, cv::Mat& output){
	const int inputBytes = input.step * input.rows;
	const int outputBytes = output.step * output.rows;

	unsigned char *deviceInput, *deviceOutput;

	SAFE_CALL(cudaMalloc<unsigned char>(&deviceInput,inputBytes),"CUDA Malloc Error");
	SAFE_CALL(cudaMalloc<unsigned char>(&deviceOutput,outputBytes),"CUDA Malloc Error");

	SAFE_CALL(cudaMemcpy(deviceInput,input.ptr(),inputBytes,cudaMemcpyHostToDevice),"CUDA Memcpy Host To Device Error");

	const dim3 block(16,16);
	const dim3 grid((input.cols + block.x - 1)/block.x, (input.rows + block.y - 1)/block.y);

	cuda_Sobel(deviceInput,deviceOutput,input.cols,input.rows,3,grid,block);

	SAFE_CALL(cudaDeviceSynchronize(),"Kernel Launch Error");

	SAFE_CALL(cudaMemcpy(output.ptr(),deviceOutput,outputBytes,cudaMemcpyDeviceToHost),"CUDA Memcpy Host To Device Error");

	SAFE_CALL(cudaFree(deviceInput),"CUDA Free Error");
	SAFE_CALL(cudaFree(deviceOutput),"CUDA Free Error");
}

void gaussianFilter(const cv::Mat& input, cv::Mat& output){
	const int inputBytes = input.step * input.rows;
	const int outputBytes = output.step * output.rows;

	unsigned char *deviceInput, *deviceOutput;

	SAFE_CALL(cudaMalloc<unsigned char>(&deviceInput,inputBytes),"CUDA Malloc Error");
	SAFE_CALL(cudaMalloc<unsigned char>(&deviceOutput,outputBytes),"CUDA Malloc Error");

	SAFE_CALL(cudaMemcpy(deviceInput,input.ptr(),inputBytes,cudaMemcpyHostToDevice),"CUDA Memcpy Host To Device Error");

	const dim3 block(16,16);
	const dim3 grid((input.cols + block.x - 1)/block.x, (input.rows + block.y - 1)/block.y);

	cuda_Gaussian(deviceInput,deviceOutput,input.cols,input.rows,5,grid,block);

	SAFE_CALL(cudaDeviceSynchronize(),"Kernel Launch Error");

	SAFE_CALL(cudaMemcpy(output.ptr(),deviceOutput,outputBytes,cudaMemcpyDeviceToHost),"CUDA Memcpy Host To Device Error");

	SAFE_CALL(cudaFree(deviceInput),"CUDA Free Error");
	SAFE_CALL(cudaFree(deviceOutput),"CUDA Free Error");
}
