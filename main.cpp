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

void applyFilterVideoGPU(std::string videoPath){
	cv::VideoCapture cap;
	if(!cap.open(videoPath))
		exit(-1);
	cv::Mat frame;
	while(1)
	{
		auto begin = std::chrono::high_resolution_clock::now();
		cap >> frame;
		if( frame.empty() ) break; // end of video stream

		cv::Mat outputFrame(frame.rows,frame.cols,CV_8UC1);
		cv::Mat outputFrameSobel(frame.rows,frame.cols,CV_8UC1);

		convertToGray(frame,outputFrame);
		sobelFilter(outputFrame,outputFrameSobel);

		auto end = std::chrono::high_resolution_clock::now();
		auto elapsed = std::chrono::duration_cast<std::chrono::nanoseconds>(end - begin);
		float fps = 1/(elapsed.count()*1e-9);
		std::cout << "[GPU FPS]: "<< std::to_string(fps) << '\r' << std::flush;
		}
		std::cout << "[GPU FPS]: "<< std::to_string(fps) << '\n'
	cap.release();
}

void applyFilterVideoCPU(std::string videoPath){
	cv::VideoCapture cap;
	if(!cap.open(videoPath)) 
	exit(-1);
	cv::Mat frame;
	while(1)
	{
		auto begin = std::chrono::high_resolution_clock::now();
		cap >> frame;
		if( frame.empty() ) break; // end of video stream

		cv::Mat outputGrayFrame(frame.rows,frame.cols,CV_8UC1);
		cv::Mat outputSobelFrameX(frame.rows,frame.cols,CV_8UC1);
		cv::Mat outputSobelFrameY(frame.rows,frame.cols,CV_8UC1);
		cv::Mat outputSobelFrame(frame.rows,frame.cols,CV_8UC1);

		cv::cvtColor(frame,outputGrayFrame,cv::COLOR_RGB2GRAY); // == convertToGray(frame,outputFrame) in opencv

		// == sobelFilter(outputFrame,outputSobelFrame)
		cv::Sobel(outputGrayFrame, outputSobelFrameX, CV_8UC1, 1, 0, 1, 1, 0, cv::BORDER_DEFAULT); 
    	cv::Sobel(outputGrayFrame, outputSobelFrameY, CV_8UC1, 0, 1, 1, 1, 0, cv::BORDER_DEFAULT);
		cv::addWeighted(outputSobelFrameX, 0.5, outputSobelFrameY, 0.5, 0, outputSobelFrame);

		auto end = std::chrono::high_resolution_clock::now();
		auto elapsed = std::chrono::duration_cast<std::chrono::nanoseconds>(end - begin);
		float fps = 1/(elapsed.count()*1e-9);
		std::cout << "[CPU FPS]: "<< std::to_string(fps) << '\r' << std::flush;
	}
	std::cout << "[CPU FPS]: "<< std::to_string(fps) << '\n' 	
	cap.release();
}

void photoMode(std::string photoPath){
	cv::Mat photo = cv::imread(photoPath);
	cv::Mat outputGray(photo.rows,photo.cols,CV_8UC1);
	cv::Mat outputSobel(photo.rows,photo.cols,CV_8UC1);

	auto beginGPU = std::chrono::high_resolution_clock::now();
	convertToGray(photo,outputGray);
	sobelFilter(outputGray,outputSobel);
	auto endGPU = std::chrono::high_resolution_clock::now();
	auto elapsedGPU = std::chrono::duration_cast<std::chrono::nanoseconds>(endGPU - beginGPU);
	std::cout << "GPU Time: " << std::to_string(elapsedGPU.count()*1e-9) << "\n";
	cv::imshow("Say hello to this desgraciado in line mode :)", outputSobel);
	cv::waitKey(0) == 27;  // close winddow by pressing ESC 

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

	auto beginGPU = std::chrono::high_resolution_clock::now();
	applyFilterVideoGPU(videoPath);
	auto endGPU = std::chrono::high_resolution_clock::now();

	auto beginCPU = std::chrono::high_resolution_clock::now();
	applyFilterVideoCPU(videoPath);
	auto endCPU = std::chrono::high_resolution_clock::now();

	auto elapsedGPU = std::chrono::duration_cast<std::chrono::nanoseconds>(endGPU - beginGPU);
	auto elapsedCPU = std::chrono::duration_cast<std::chrono::nanoseconds>(endCPU - beginCPU);
	cv::VideoCapture cap;
	cap.open(videoPath);
	int totalFrames = cap.get(cv::CAP_PROP_FRAME_COUNT);


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
        ("p,photo", "Proccess a photo file, ex: photo.jpg", cxxopts::value<std::string>())
		("t,time", "Show time dif between cpu and cuda time for given video file, ex: testvideo.mp4", cxxopts::value<std::string>())
        ("h,help", "Print usage")
    ;
	try{
		auto result = options.parse(argc, argv);

		if (result.count("help")){
		std::cout << options.help() << std::endl;
		exit(0);
		}

		if (result.count("photo")){
			photoMode(result["p"].as<std::string>());
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