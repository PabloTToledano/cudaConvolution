#include "cudaKernels.h"
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


void convertToGray(const cv::Mat& input, cv::Mat& output){
	//Calculate total number of bytes of input and output image
	const int colorBytes = input.step * input.rows;
	const int grayBytes = output.step * output.rows;

	unsigned char *d_input, *d_output;

	//Allocate device memory
	SAFE_CALL(cudaMalloc<unsigned char>(&d_input,colorBytes),"CUDA Malloc Failed");
	SAFE_CALL(cudaMalloc<unsigned char>(&d_output,grayBytes),"CUDA Malloc Failed");

	//Copy data from OpenCV input image to device memory
	SAFE_CALL(cudaMemcpy(d_input,input.ptr(),colorBytes,cudaMemcpyHostToDevice),"CUDA Memcpy Host To Device Failed");

	//Specify a reasonable block size
	const dim3 block(16,16);

	//Calculate grid size to cover the whole image
	const dim3 grid((input.cols + block.x - 1)/block.x, (input.rows + block.y - 1)/block.y);

	//Launch the color conversion kernel
	cuda_grayscale(d_input,d_output,input.cols,input.rows,input.step,output.step,grid,block);

	//Synchronize to check for any kernel launch errors
	SAFE_CALL(cudaDeviceSynchronize(),"Kernel Launch Failed");

	//Copy back data from destination device meory to OpenCV output image
	SAFE_CALL(cudaMemcpy(output.ptr(),d_output,grayBytes,cudaMemcpyDeviceToHost),"CUDA Memcpy Host To Device Failed");

	//Free the device memory
	SAFE_CALL(cudaFree(d_input),"CUDA Free Failed");
	SAFE_CALL(cudaFree(d_output),"CUDA Free Failed");
	}

void sobelFilter(const cv::Mat& input, cv::Mat& output){
			//Calculate total number of bytes of input and output image
	const int colorBytes = input.step * input.rows;
	const int grayBytes = output.step * output.rows;

	unsigned char *d_input, *d_output;

	//Allocate device memory
	SAFE_CALL(cudaMalloc<unsigned char>(&d_input,colorBytes),"CUDA Malloc Failed");
	SAFE_CALL(cudaMalloc<unsigned char>(&d_output,grayBytes),"CUDA Malloc Failed");

	//Copy data from OpenCV input image to device memory
	SAFE_CALL(cudaMemcpy(d_input,input.ptr(),colorBytes,cudaMemcpyHostToDevice),"CUDA Memcpy Host To Device Failed");

	//Specify a reasonable block size
	const dim3 block(16,16);

	//Calculate grid size to cover the whole image
	const dim3 grid((input.cols + block.x - 1)/block.x, (input.rows + block.y - 1)/block.y);

	//Launch the color conversion kernel
	cuda_Sobel(d_input,d_output,input.cols,input.rows,3,grid,block);

	//Synchronize to check for any kernel launch errors
	SAFE_CALL(cudaDeviceSynchronize(),"Kernel Launch Failed");

	//Copy back data from destination device meory to OpenCV output image
	SAFE_CALL(cudaMemcpy(output.ptr(),d_output,grayBytes,cudaMemcpyDeviceToHost),"CUDA Memcpy Host To Device Failed");

	//Free the device memory
	SAFE_CALL(cudaFree(d_input),"CUDA Free Failed");
	SAFE_CALL(cudaFree(d_output),"CUDA Free Failed");
	}

void showMode(){
	cv::VideoCapture cap;
	if(!cap.open(0))
		exit(-1);
	cv::Mat frame;
		while(true)
		{
				cap >> frame;
				cv::Mat outputFrame(frame.rows,frame.cols,CV_8UC1);
				cv::Mat outputFrameSobel(frame.rows,frame.cols,CV_8UC1);
				if( frame.empty() ) break; // end of video stream
				convertToGray(frame,outputFrame);
				//cv::imshow("Say hello to this desgraciado in grayscale :)", outputFrame);
				sobelFilter(outputFrame,outputFrameSobel);
				cv::imshow("Say hello to this desgraciado in line mode :)", outputFrameSobel);

				if( cv::waitKey(10) == 27 ) break; // close winddow by pressing ESC 
		}
		cap.release();
}

void benchMode(int nFrames){
	cv::VideoCapture cap;
	if(!cap.open(0))
		exit(-1);
	cv::Mat frame;
	
		//GPU 
		auto beginGPU = std::chrono::high_resolution_clock::now();
		for(int i=0;i<nFrames; i++)
		{
				cap >> frame;
				cv::Mat outputFrame(frame.rows,frame.cols,CV_8UC1);
				cv::Mat outputFrameSobel(frame.rows,frame.cols,CV_8UC1);
				if( frame.empty() ) break; // end of video stream
				convertToGray(frame,outputFrame);
				sobelFilter(outputFrame,outputFrameSobel);
				//cv::imshow("Say hello to this desgraciado in line mode :)", outputFrameSobel);
				//if( cv::waitKey(10) == 27 ) break; // close winddow by pressing ESC 
		}
		auto endGPU = std::chrono::high_resolution_clock::now();
		//CPU
		auto beginCPU = std::chrono::high_resolution_clock::now();
		for(int i=0;i<nFrames; i++)
		{
				cap >> frame;
				cv::Mat outputGrayFrame(frame.rows,frame.cols,CV_8UC1);
				cv::Mat outputSobelFrameX(frame.rows,frame.cols,CV_8UC1);
				cv::Mat outputSobelFrameY(frame.rows,frame.cols,CV_8UC1);
				cv::Mat outputSobelFrame(frame.rows,frame.cols,CV_8UC1);
				if( frame.empty() ) break; // end of video stream
				cv::cvtColor(frame,outputGrayFrame,cv::COLOR_RGB2GRAY); // == convertToGray(frame,outputFrame) in opencv
				// == sobelFilter(outputFrame,outputSobelFrame)
				cv::Sobel(outputGrayFrame, outputSobelFrameX, CV_8UC1, 1, 0, 1, 1, 0, cv::BORDER_DEFAULT); 
    			cv::Sobel(outputGrayFrame, outputSobelFrameY, CV_8UC1, 0, 1, 1, 1, 0, cv::BORDER_DEFAULT);
				cv::addWeighted(outputSobelFrameX, 0.5, outputSobelFrameY, 0.5, 0, outputSobelFrame);
		}
		auto endCPU = std::chrono::high_resolution_clock::now();
		cap.release();
		auto elapsedGPU = std::chrono::duration_cast<std::chrono::nanoseconds>(endGPU - beginGPU);
		auto elapsedCPU = std::chrono::duration_cast<std::chrono::nanoseconds>(endCPU - beginCPU);
		printf("Time for %d frames\nGPU time: %.3f seconds.\nCPU time: %.3f seconds.\n",nFrames ,elapsedGPU.count()*1e-9,elapsedCPU.count()*1e-9);

}

int main(int argc, char** argv)
{
	cxxopts::Options options("Sobel Test", "Computadores Avanzados final commit \n Choose either s or t modes");
	options.add_options()
        ("s,show", "Realtime visualization mode", cxxopts::value<bool>()->default_value("true"))
        ("t,time", "Show dif between cpu and cuda time for given number of frames, ex: 30", cxxopts::value<int>())
        ("h,help", "Print usage")
    ;
	try{
		auto result = options.parse(argc, argv);

		if (result.count("help")){
		std::cout << options.help() << std::endl;
		exit(0);
		}

		if (result.count("show")){
			showMode();
		}

		if (result.count("time")){
			benchMode(result["t"].as<int>());
		}

	}catch(cxxopts::OptionException e){
		std::cerr << e.what() << std::endl;
		std::cout << options.help() << std::endl;
		exit(-1);
	}
	return 0;
}