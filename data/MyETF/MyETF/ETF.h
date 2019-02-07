#pragma once
#include <stdio.h>
#include <opencv2/opencv.hpp>
#include <cmath>

using namespace std;
const int img_size = 512;

class ETF {
public:
	ETF();
	ETF(cv::Size);
	void Init(cv::Size);
	void initial_ETF(string, cv::Size);
	void refine_ETF(int kernel);
	void rotateFlow(cv::Mat& src, cv::Mat& dst, float theta);
	void getAngle();

	cv::Mat gradientMag; // Normalized gradient magnitude
	cv::Mat flowField;   // edge tangent flow
	cv::Mat refinedETF;  // ETF after refinement 
	float angles[img_size][img_size];

private:
	void resizeMat(cv::Size);
	void computeNewVector(int x, int y, const int kernel);
	float computePhi(cv::Vec3f x, cv::Vec3f y);
	float computeWs(cv::Point2f x, cv::Point2f y, int r);
	float computeWm(float gradmag_x, float gradmag_y);
	float computeWd(cv::Vec3f x, cv::Vec3f y);
};
