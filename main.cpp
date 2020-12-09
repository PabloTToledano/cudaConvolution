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

void showMode(std::string videoStream){
	cv::VideoCapture cap;
	std::string cameraString = "0";
	double delayPerFrame = 33.33; //default 33.33ms or 30 fps
	//Open webcam or video file
	if(cameraString.compare(videoStream) == 0){
			if(!cap.open(0))
			exit(-1);
	}else{
			if(!cap.open(videoStream))
			exit(-1);
			//calculate delay appropiated for the video
			delayPerFrame = 1000/cap.get(cv::CAP_PROP_FPS);
			std::cout << "Video delay per frame: " << std::to_string(delayPerFrame) <<  " FPS: "<< std::to_string(cap.get(cv::CAP_PROP_FPS)) << "\n";
	}

	cv::Mat frame;
		while(true)
		{
				cap >> frame;
				if( frame.empty() ) break; // end of video stream
				cv::Mat outputFrame(frame.rows,frame.cols,CV_8UC1);
				cv::Mat outputFrameSobel(frame.rows,frame.cols,CV_8UC1);
				convertToGray(frame,outputFrame);
				//cv::imshow("Say hello to this desgraciado in grayscale :)", outputFrame);
				sobelFilter(outputFrame,outputFrameSobel);
				cv::imshow("Say hello to this desgraciado in line mode :)", outputFrameSobel);
				if( cv::waitKey(delayPerFrame) == 27 ) break; // close winddow by pressing ESC 
		}
		cap.release();
}

void benchMode(std::string videoPath){
	cv::VideoCapture cap;
	if(!cap.open(videoPath))
		exit(-1);
	cv::Mat frame;
	int totalFrames = 0;
	//GPU 
	auto beginGPU = std::chrono::high_resolution_clock::now();
	while(1)
	{
		cap >> frame;
		cv::Mat outputFrame(frame.rows,frame.cols,CV_8UC1);
		cv::Mat outputFrameSobel(frame.rows,frame.cols,CV_8UC1);
		if( frame.empty() ) break; // end of video stream
		convertToGray(frame,outputFrame);
		sobelFilter(outputFrame,outputFrameSobel);
		//cv::imshow("Say hello to this desgraciado in line mode :)", outputFrameSobel);
		//if( cv::waitKey(10) == 27 ) break; // close winddow by pressing ESC 
		totalFrames++; //total number of frames for fps calc
		}
	auto endGPU = std::chrono::high_resolution_clock::now();

	//CPU
	auto beginCPU = std::chrono::high_resolution_clock::now();
	cap.release();
	if(!cap.open(videoPath)) //open video again for cpu test
	exit(-1);
	while(1)
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
	double fpsGPU = totalFrames/ (elapsedGPU.count()*1e-9);
	double fpsCPU = totalFrames/ (elapsedCPU.count()*1e-9);
	std::cout << "GPU Time: " << std::to_string(elapsedGPU.count()*1e-9) << " FPS: " << std::to_string(fpsGPU) << "\n";
	std::cout << "CPU Time: " << std::to_string(elapsedCPU.count()*1e-9) << " FPS: " << std::to_string(fpsCPU) << "\n";

}

int main(int argc, char** argv)
{
	cxxopts::Options options("Sobel Test", "Computadores Avanzados final assessment \n Choose either s or t modes");
	options.add_options()
        ("s,show", "Realtime visualization mode, ex: --show=video.mp4", cxxopts::value<std::string>()->implicit_value("0"))
        ("t,time", "Show time dif between cpu and cuda time for given video file, ex: testvideo.mp4", cxxopts::value<std::string>())
        ("h,help", "Print usage")
    ;
	try{
		auto result = options.parse(argc, argv);

		if (result.count("help")){
		std::cout << options.help() << std::endl;
		exit(0);
		}

		if (result.count("show")){
			showMode(result["s"].as<std::string>());
		}

		if (result.count("time")){
			benchMode(result["t"].as<std::string>());
		}

	}catch(cxxopts::OptionException e){
		std::cerr << e.what() << std::endl;
		std::cout << options.help() << std::endl;
		exit(-1);
	}
	return 0;
}