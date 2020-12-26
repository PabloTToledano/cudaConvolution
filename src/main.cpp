#include "include/cudaKernels.h"
#include "include/cudaFilter.h"
#include "libs/cxxopts.hpp"
#include<iostream>
#include<cstdio>
#include <chrono>
#include<opencv2/core/core.hpp>
#include<opencv2/highgui/highgui.hpp>
#include <opencv2/opencv.hpp>


inline const char * const BoolToString(bool b)
{
  return b ? "true" : "false";
}

int openVideo(cv::VideoCapture& cap,std::string videoPath){
	const std::string cameraString = "0";
	int videoFPS = 30;
	//Open webcam or video file
	if(cameraString.compare(videoPath) == 0){
			if(!cap.open(0))
			exit(-1);
	}else{
			if(!cap.open(videoPath))
			exit(-1);
			videoFPS =cap.get(cv::CAP_PROP_FPS);
	}
	return videoFPS;
}

void applyFilterVideoGPU(std::string videoPath, bool show, bool blur){
	cv::VideoCapture cap;
	int videoFPS = openVideo(cap,videoPath);

	double delayPerFrame = 1000/videoFPS;
	if(show)
	std::cout << "Video delay per frame: " << std::to_string(delayPerFrame) <<  " FPS: "<< std::to_string(videoFPS) << "\n";
	cv::Mat frame;
	while(1)
	{
		cap >> frame;
		if( frame.empty() ) break; // end of video stream

		cv::Mat outputFrameGray(frame.rows,frame.cols,CV_8UC1);
		cv::Mat outputFrameSobel(frame.rows,frame.cols,CV_8UC1);

		convertToGray(frame,outputFrameGray);
		if(blur){
			cv::Mat outputFrameGaus(frame.rows,frame.cols,CV_8UC1);
			gaussianFilter(outputFrameGray,outputFrameGaus);
			sobelFilter(outputFrameGaus,outputFrameSobel);
		}else{
			sobelFilter(outputFrameGray,outputFrameSobel);
		}

		if(show){
			//cv::resize(outputFrameSobel,outputFrameSobel,cv::Size(1280,720));
			cv::imshow("Say hello to this desgraciado in line mode :)", outputFrameSobel);
			if( cv::waitKey(delayPerFrame) == 27 ) break; // close winddow by pressing ESC 
		}
	}

	cap.release();
}

void applyFilterVideoCPU(std::string videoPath, bool show, bool blur ){
	cv::VideoCapture cap;
	openVideo(cap,videoPath);
	double delayPerFrame = 1000/cap.get(cv::CAP_PROP_FPS);
	cv::Mat frame;
	while(1)
	{
		cap >> frame;
		if( frame.empty() ) break; // end of video stream

		cv::Mat outputGrayFrame(frame.rows,frame.cols,CV_8UC1);
		cv::Mat outputFrameSobelX(frame.rows,frame.cols,CV_8UC1);
		cv::Mat outputFrameSobelY(frame.rows,frame.cols,CV_8UC1);
		cv::Mat outputFrameSobel(frame.rows,frame.cols,CV_8UC1);

		cv::cvtColor(frame,outputGrayFrame,cv::COLOR_RGB2GRAY); // == convertToGray(frame,outputFrame) in opencv

		if(blur){
			cv::Mat outputFrameGaus(frame.rows,frame.cols,CV_8UC1);
			cv::GaussianBlur(outputGrayFrame,outputFrameGaus,cv::Size(5,5),0); // == goussianFilter in opencv
			// == sobelFilter(outputFrame,outputFrameSobel)
			cv::Sobel(outputFrameGaus, outputFrameSobelX, CV_8UC1, 1, 0, 1, 1, 0, cv::BORDER_DEFAULT); 
    		cv::Sobel(outputFrameGaus, outputFrameSobelY, CV_8UC1, 0, 1, 1, 1, 0, cv::BORDER_DEFAULT);
		}else{
			cv::Sobel(outputGrayFrame, outputFrameSobelX, CV_8UC1, 1, 0, 1, 1, 0, cv::BORDER_DEFAULT); 
    		cv::Sobel(outputGrayFrame, outputFrameSobelY, CV_8UC1, 0, 1, 1, 1, 0, cv::BORDER_DEFAULT);
		}

		cv::addWeighted(outputFrameSobelX, 0.5, outputFrameSobelY, 0.5, 0, outputFrameSobel);
		if(show){
			cv::resize(outputFrameSobel,outputFrameSobel,cv::Size(1280,720));
			cv::imshow("Say hello to this desgraciado in line mode :)", outputFrameSobel);
			if( cv::waitKey(delayPerFrame) == 27 ) break; // close winddow by pressing ESC 
		}

	} 	
	cap.release();
}

cv::Mat applyFilterPhotoGPU(std::string photoPath,bool blur){
	cv::Mat photo = cv::imread(photoPath);
	std::cout << "[GPU] Processing " << photoPath << "\n Size: " << std::to_string((int)photo.rows) << "x" << std::to_string((int)photo.cols) << "\n";
	cv::Mat outputGray(photo.rows,photo.cols,CV_8UC1);
	cv::Mat outputSobel(photo.rows,photo.cols,CV_8UC1);

	convertToGray(photo,outputGray);
	if(blur){
		cv::Mat outputGaus(photo.rows,photo.cols,CV_8UC1);
		gaussianFilter(outputGray,outputGaus);
		sobelFilter(outputGaus,outputSobel);
	}else{
		sobelFilter(outputGray,outputSobel);
	}
	return outputSobel;
}

cv::Mat applyFilterPhotoCPU(std::string photoPath,bool blur){
	cv::Mat photo = cv::imread(photoPath);
	std::cout << "[CPU] Processing " << photoPath << "\n Size: " << std::to_string((int)photo.rows) << "x" << std::to_string((int)photo.cols) << "\n";
	cv::Mat outputGray(photo.rows,photo.cols,CV_8UC1);
	cv::Mat outputGaus(photo.rows,photo.cols,CV_8UC1);
	cv::Mat outputSobelX(photo.rows,photo.cols,CV_8UC1);
	cv::Mat outputSobelY(photo.rows,photo.cols,CV_8UC1);
	cv::Mat outputSobel(photo.rows,photo.cols,CV_8UC1);

	cv::cvtColor(photo,outputGray,cv::COLOR_RGB2GRAY); // == convertToGray(frame,outputFrame) in opencv
	if(blur){
		cv::GaussianBlur(outputGray,outputGaus,cv::Size(5,5),0); // == goussianFilter in opencv
		// == sobelFilter(outputFrame,outputFrameSobel)
		cv::Sobel(outputGaus, outputSobelX, CV_8UC1, 1, 0, 1, 1, 0, cv::BORDER_DEFAULT); 
		cv::Sobel(outputGaus, outputSobelY, CV_8UC1, 0, 1, 1, 1, 0, cv::BORDER_DEFAULT);
	}else{
		cv::Sobel(outputGray, outputSobelX, CV_8UC1, 1, 0, 1, 1, 0, cv::BORDER_DEFAULT); 
		cv::Sobel(outputGray, outputSobelY, CV_8UC1, 0, 1, 1, 1, 0, cv::BORDER_DEFAULT);
	}
	cv::addWeighted(outputSobelX, 0.5, outputSobelY, 0.5, 0, outputSobel);
	return outputSobel;
}

void photoMode(std::string photoPath,std::string outPath, bool blur){
	
	std::cout << "Gaussian Blur: " << BoolToString(blur) << "\n";
	auto beginGPU = std::chrono::high_resolution_clock::now();
	cv::Mat outputGPU = applyFilterPhotoGPU(photoPath,blur);
	auto endGPU = std::chrono::high_resolution_clock::now();

	auto beginCPU = std::chrono::high_resolution_clock::now();
	cv::Mat outputCPU = applyFilterPhotoCPU(photoPath,blur);
	auto endCPU = std::chrono::high_resolution_clock::now();

	cv::imwrite(outPath, outputGPU);

	auto elapsedGPU = std::chrono::duration_cast<std::chrono::nanoseconds>(endGPU - beginGPU);
	auto elapsedCPU = std::chrono::duration_cast<std::chrono::nanoseconds>(endCPU - beginCPU);
	std::cout << "[GPU] Time: " << std::to_string(elapsedGPU.count()*1e-9) << "\n";
	std::cout << "[CPU] Time: " << std::to_string(elapsedCPU.count()*1e-9) << "\n";
	std::cout << "Performance gain: " << std::to_string((elapsedCPU.count()*1e-9)/(elapsedGPU.count()*1e-9)) << "\n";

}

void benchVideoMode(std::string videoPath,bool blur){
	std::cout << "Gaussian Blur: " << BoolToString(blur) << "\n";
	auto beginGPU = std::chrono::high_resolution_clock::now();
	applyFilterVideoGPU(videoPath,false,blur);
	auto endGPU = std::chrono::high_resolution_clock::now();

	auto beginCPU = std::chrono::high_resolution_clock::now();
	applyFilterVideoCPU(videoPath,false,blur);
	auto endCPU = std::chrono::high_resolution_clock::now();

	auto elapsedGPU = std::chrono::duration_cast<std::chrono::nanoseconds>(endGPU - beginGPU);
	auto elapsedCPU = std::chrono::duration_cast<std::chrono::nanoseconds>(endCPU - beginCPU);
	cv::VideoCapture cap;
	cap.open(videoPath);
	int totalFrames = cap.get(cv::CAP_PROP_FRAME_COUNT);


	double fpsGPU = totalFrames/ (elapsedGPU.count()*1e-9);
	double fpsCPU = totalFrames/ (elapsedCPU.count()*1e-9);

	std::cout << "[GPU]"<< "\n" << "\tTime: " << std::to_string(elapsedGPU.count()*1e-9) << "\n\tFPS: " << std::to_string(fpsGPU) << "\n";
	std::cout << "[CPU]"<< "\n" << "\tTime: " << std::to_string(elapsedCPU.count()*1e-9) << "\n\tFPS: " << std::to_string(fpsCPU) << "\n";
	std::cout << "Performance gain: " << std::to_string((elapsedCPU.count()*1e-9)/(elapsedGPU.count()*1e-9)) << "\n";

}

int main(int argc, char** argv)
{
	cxxopts::Options options("Sobel Test", "Computadores Avanzados final assessment \n");
	options.add_options()
        ("s,show", "Realtime video visualization mode, ex: --show=video.mp4", cxxopts::value<std::string>()->implicit_value("0"))
        ("p,photo", "Proccess a photo file, ex: photo.jpg", cxxopts::value<std::string>())
		("v,video", "Shows time dif between cpu and cuda time for a given video file, ex: testvideo.mp4", cxxopts::value<std::string>())
		("o,out", "Output name for photo file, ex: out.jpg", cxxopts::value<std::string>()->implicit_value("o.jpg"))
        ("g,gaus", "Enables Gaussian Blur filter in the filter pipeline")
		("h,help", "Print usage")
    ;
	try{
		auto result = options.parse(argc, argv);

		if (result.count("help")){
			std::cout << options.help() << std::endl;
			exit(0);
		}

		if (result.count("photo")){
			if (result.count("out")){
				photoMode(result["p"].as<std::string>(),result["o"].as<std::string>(),result.count("gaus"));
			}else{
				photoMode(result["p"].as<std::string>(),"o.jpg",result.count("gaus"));
			}

			exit(0);
		}

		if (result.count("show")){
			applyFilterVideoGPU(result["s"].as<std::string>(),true,result.count("gaus"));
			exit(0);
		}

		if (result.count("video")){
			benchVideoMode(result["v"].as<std::string>(),result.count("gaus"));
			exit(0);
		}

	}catch(cxxopts::OptionException e){
		std::cerr << e.what() << std::endl;
		std::cout << options.help() << std::endl;
		exit(-1);
	}
	return 0;
}