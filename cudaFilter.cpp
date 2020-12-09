#include "cudaKernels.h"
#include "cudaFilter.h"
#include "cxxopts.hpp"
#include<iostream>
#include<cstdio>
#include <chrono>
#include<opencv2/core/core.hpp>
#include<opencv2/highgui/highgui.hpp>
#include <opencv2/opencv.hpp>
#include <cuda_runtime.h>
#include <cuda.h>


static inline void _safe_cuda_call(cudaError err, const char* msg, const char* file_name, const int line_number)
{
	if(err!=cudaSuccess)
	{
		fprintf(stderr,"%s\n\nFile: %s\n\nLine Number: %d\n\nReason: %s\n",msg,file_name,line_number,cudaGetErrorString(err));
		std::cin.get();
		exit(EXIT_FAILURE);
	}
}

#define SAFE_CALL(call,msg) _safe_cuda_call((call),(msg),__FILE__,__LINE__)



class CudaFilter{
	public:
		dim3 block;
		dim3 grid;

    CudaFilter(const cv::Mat& frame){
		block = (16,16);
		grid = ((frame.cols + block.x - 1)/block.x, (frame.rows + block.y - 1)/block.y);
	}

	void convertToGray(const cv::Mat& input, cv::Mat& output){
		int frameByte = input.step * input.rows;; //total bytes of frame
		printf("Mi tamaño de cosa es %d\n",frameByte);
		unsigned char *d_input, *d_output;

		//Allocate device memory
		SAFE_CALL(cudaMalloc<unsigned char>(&d_input,frameByte),"CUDA Malloc Failed");
		SAFE_CALL(cudaMalloc<unsigned char>(&d_output,frameByte),"CUDA Malloc Failed");

		//Copy data from OpenCV input image to device memory
		SAFE_CALL(cudaMemcpy(d_input,input.ptr(),frameByte,cudaMemcpyHostToDevice),"CUDA Memcpy Host To Device Failed");

		//Launch the color conversion kernel
		//rgb_to_grayscale<<<grid,block>>>(d_input,d_output,input.cols,input.rows,input.step,output.step);
		cuda_grayscale(d_input,d_output,input.cols,input.rows,input.step,output.step,grid,block);
		//Synchronize to check for any kernel launch errors
		SAFE_CALL(cudaDeviceSynchronize(),"Kernel Launch Failed");

		//Copy back data from destination device meory to OpenCV output image
		SAFE_CALL(cudaMemcpy(output.ptr(),d_output,frameByte,cudaMemcpyDeviceToHost),"CUDA Memcpy Host To Device Failed");

		//Free the device memory
		SAFE_CALL(cudaFree(d_input),"CUDA Free Failed");
		SAFE_CALL(cudaFree(d_output),"CUDA Free Failed");
	}
    void sobelFilter(const cv::Mat& input, cv::Mat& output){
		int frameByte = input.step * input.rows;; //total bytes of frame
		unsigned char *d_input, *d_output;


		//Allocate device memory
		SAFE_CALL(cudaMalloc<unsigned char>(&d_input,frameByte),"CUDA Malloc Failed");
		SAFE_CALL(cudaMalloc<unsigned char>(&d_output,frameByte),"CUDA Malloc Failed");

		//Copy data from OpenCV input image to device memory
		SAFE_CALL(cudaMemcpy(d_input,input.ptr(),frameByte,cudaMemcpyHostToDevice),"CUDA Memcpy Host To Device Failed");

		//Launch the color conversion kernel
		cuda_Sobel(d_input,d_output,input.cols,input.rows,3,grid,block);
		//Synchronize to check for any kernel launch errors
		SAFE_CALL(cudaDeviceSynchronize(),"Kernel Launch Failed");

		//Copy back data from destination device meory to OpenCV output image
		SAFE_CALL(cudaMemcpy(output.ptr(),d_output,frameByte,cudaMemcpyDeviceToHost),"CUDA Memcpy Host To Device Failed");

		//Free the device memory
		SAFE_CALL(cudaFree(d_input),"CUDA Free Failed");
		SAFE_CALL(cudaFree(d_output),"CUDA Free Failed");
	}

};
