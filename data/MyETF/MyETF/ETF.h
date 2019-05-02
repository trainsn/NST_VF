#pragma once
#include <stdio.h>
#include <iostream>
#include <opencv2/opencv.hpp>
#include <cmath>
#include <queue>

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
	void getVector();

	cv::Mat gradientMag; // Normalized gradient magnitude
	cv::Mat flowField;   // edge tangent flow
	cv::Mat refinedETF;  // ETF after refinement 
	float angles[img_size][img_size];
	float vectors[img_size][img_size][2];
	const int dx[4] = { -1, 0, 1, 0 };
	const int dy[4] = { 0, -1, 0, 1 };

private:
	void resizeMat(cv::Size);
	void computeNewVector(int x, int y, const int kernel);
	float computePhi(cv::Vec3f x, cv::Vec3f y);
	float computeWs(cv::Point2f x, cv::Point2f y, int r);
	float computeWm(float gradmag_x, float gradmag_y);
	float computeWd(cv::Vec3f x, cv::Vec3f y);
};
