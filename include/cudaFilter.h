#include<iostream>
#include<cstdio>
#include<opencv2/core/core.hpp>
#include<opencv2/highgui/highgui.hpp>
#include <opencv2/opencv.hpp>
#include <cuda_runtime.h>
#include <cuda.h>

void convertToGray(const cv::Mat& input, cv::Mat& output);
void sobelFilter(const cv::Mat& input, cv::Mat& output);
void gaussianFilter(const cv::Mat& input, cv::Mat& output);
