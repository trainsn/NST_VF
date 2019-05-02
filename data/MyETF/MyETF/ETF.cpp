#include "ETF.h"
# define M_PI 3.14159265358979323846

using namespace cv;
using namespace std;
const float eps = 1e-10;
const float PI = 3.1415926;


ETF::ETF() {
	Size s(512, 512);

	Init(s);
}

ETF::ETF(Size s) {
	Init(s);
}

void ETF::Init(Size s) {
	flowField = Mat::zeros(s, CV_32FC3);
	refinedETF = Mat::zeros(s, CV_32FC3);
	gradientMag = Mat::zeros(s, CV_32FC3);
}

/**
 * Generate initial ETF
 * by taking perpendicular vectors(counter-clockwise) from gradient map
 */
void ETF::initial_ETF(string file, Size s) {
	resizeMat(s);

	Mat src = imread(file, 1);
	Mat src_n;
	Mat grad;
	normalize(src, src_n, 0.0, 1.0, NORM_MINMAX, CV_32FC1);

	freopen("src.csv", "w", stdout);
	for (int i = 0; i < src.rows; i++) {
		for (int j = 0; j < src.cols; j++) {
			Vec3f p = src_n.at<Vec3f>(i, j);
			cout << p.val[0] << ","; 
		}
		cout << endl;
	}
	// GaussianBlur(src_n, src_n, Size(101, 101), 0, 0);

	// Generate grad_x and grad_y
	Mat grad_x, grad_y, abs_grad_x, abs_grad_y;
	Sobel(src_n, grad_x, CV_32FC1, 1, 0, 5);
	Sobel(src_n, grad_y, CV_32FC1, 0, 1, 5);

	freopen("grad_x.csv", "w", stdout);
	for (int i = 0; i < src.rows; i++) {
		for (int j = 0; j < src.cols; j++) {
			Vec3f u = grad_x.at<Vec3f>(i, j);
			cout << u.val[0] << ",";
		}
		cout << endl;
	}

	freopen("grad_y.csv", "w", stdout);
	for (int i = 0; i < src.rows; i++) {
		for (int j = 0; j < src.cols; j++) {
			Vec3f v = grad_y.at<Vec3f>(i, j);
			cout << v.val[0] << ",";
		}
		cout << endl;
	}

	//Compute gradient
	magnitude(grad_x, grad_y, gradientMag);
	normalize(gradientMag, gradientMag, 0.0, 1.0, NORM_MINMAX);

	freopen("gradientMag.csv", "w", stdout);
	for (int i = 0; i < src.rows; i++) {
		for (int j = 0; j < src.cols; j++) {
			Vec3f v = gradientMag.at<Vec3f>(i, j);
			cout << v.val[0] << ",";
		}
		cout << endl;
	}

	Mat refined_grad_x = grad_x.clone();
	Mat refined_grad_y = grad_y.clone();
	Mat refined_gradientMag = gradientMag.clone();
#pragma omp parallel for
	for (int y = 0; y < src.rows; y++) {
		for (int x = 0; x < src.cols; x++) {
			Vec3f v = gradientMag.at<Vec3f>(y, x);
	
			if (v.val[0] == 0) {
				vector<vector<bool> > visited(src.rows, vector<bool>(src.cols, false));
				queue<Point2f> q1;
				q1.push(Point2f(y, x));
				visited[y][x] = true;
				float min_dist = 1000000;
				int miny = -1, minx = -1;
				while (!q1.empty()) {
					Point2f cur = q1.front();
					q1.pop();
					for (int k = 0; k < 4; k++) {
						int ny = cur.x + dy[k];
						int nx = cur.y + dx[k];
						if (ny < 0 || ny >= refinedETF.rows ||  nx < 0 || nx >= refinedETF.cols)
							continue;
							
						Vec3f nv = gradientMag.at<Vec3f>(ny, nx);
						if (!visited[ny][nx]) {
							if (norm(Point2f(y, x) - Point2f(ny, nx)) < min_dist) {
								if (nv.val[0]) {
									min_dist = norm(Point2f(y, x) - Point2f(ny, nx));
									miny = ny;
									minx = nx;
								}
								q1.push(Point2f(ny, nx));
								visited[ny][nx] = true;	
							}
						}
					}	
				}
				refined_grad_x.at<Vec3f>(y, x) = grad_x.at<Vec3f>(miny, minx);
				refined_grad_y.at<Vec3f>(y, x) = grad_y.at<Vec3f>(miny, minx);
				refined_gradientMag.at<Vec3f>(y, x) = gradientMag.at<Vec3f>(miny, minx);
			}
		}
	}
	grad_x = refined_grad_x.clone();
	grad_y = refined_grad_y.clone();
	gradientMag = refined_gradientMag.clone();

	flowField = Mat::zeros(src.size(), CV_32FC3);

#pragma omp parallel for
	for (int i = 0; i < src.rows; i++) {
		for (int j = 0; j < src.cols; j++) {
			Vec3f u = grad_x.at<Vec3f>(i, j);
			Vec3f v = grad_y.at<Vec3f>(i, j);

			flowField.at<Vec3f>(i, j) = normalize(Vec3f(v.val[0], u.val[0], 0));
		}
	}

	rotateFlow(flowField, flowField, 90);
}


void ETF::refine_ETF(int kernel) {
#pragma omp parallel for
	for (int r = 0; r < flowField.rows; r++) {
		for (int c = 0; c < flowField.cols; c++) {
			computeNewVector(c, r, kernel);
		}
	}

	flowField = refinedETF.clone();
}

// add by trainsn
void ETF::getAngle() {
#pragma omp parallel for
	for (int r = 0; r < flowField.rows; r++) {
		for (int c = 0; c < flowField.cols; c++) {
			const Vec3f t_cur_y = flowField.at<Vec3f>(r, c);
			float angle = 0;
			if (t_cur_y.val[1] > eps) 
				angle = atan(t_cur_y.val[0] / t_cur_y.val[1]);
			angles[r][c] = angle * 180 / PI;
			//float rr = r - img_size/2;
			//float cc = c - img_size/2;
			//flowField.at<Vec3f>(r, c) = normalize(Vec3f(cc, -rr, 0));
		}
	}
}

void ETF::getVector() {
#pragma omp parallel for
	for (int r = 0; r < flowField.rows; r++) {
		for (int c = 0; c < flowField.cols; c++) {
			const Vec3f t_cur_y = flowField.at<Vec3f>(r, c);
			vectors[r][c][0] = t_cur_y.val[1];
			vectors[r][c][1] = t_cur_y.val[0];
		}
	}
}


/*
 * Paper's Eq(1)
 */
void ETF::computeNewVector(int x, int y, const int kernel) {
	const Vec3f t_cur_x = flowField.at<Vec3f>(y, x);
	Vec3f t_new = Vec3f(0, 0, 0);

	for (int r = y - kernel; r <= y + kernel; r++) {
		for (int c = x - kernel; c <= x + kernel; c++) {
			if (r < 0 || r >= refinedETF.rows || c < 0 || c >= refinedETF.cols) continue;

			const Vec3f t_cur_y = flowField.at<Vec3f>(r, c);
			float phi = computePhi(t_cur_x, t_cur_y);
			float w_s = computeWs(Point2f(y, x), Point2f(r, c), kernel);
			float w_m = computeWm(gradientMag.at<Vec3f>(y, x).val[0], gradientMag.at<Vec3f>(r, c).val[0]);
			/*std::cout << norm(gradientMag.at<Vec3f>(y, x)) << " " << norm(gradientMag.at<Vec3f>(r, c)) << 
				" " << gradientMag.at<Vec3f>(y, x).val[0] << " " << gradientMag.at<Vec3f>(r, c).val[0] <<
				" " << w_m << std::endl;*/
			float w_d = computeWd(t_cur_x, t_cur_y);
			//float w_d = norm(t_cur_x) != 0  ?  tmp_w_d : 0.5;
			t_new += phi * t_cur_y*w_s*w_m*w_d;
		}
	}
	refinedETF.at<Vec3f>(y, x) = normalize(t_new);
}


/*
 * Paper's Eq(5)
 */
float ETF::computePhi(cv::Vec3f x, cv::Vec3f y) {
	return x.dot(y) >= 0 ? 1 : -1;
}

/*
 * Paper's Eq(2)
 */
float ETF::computeWs(cv::Point2f x, cv::Point2f y, int r) {
	return norm(x - y) < r ? 1 : 0;
}

/*
 * Paper's Eq(3)
 */
float ETF::computeWm(float gradmag_x, float gradmag_y) {
	float wm = (1 + tanh(1 * (gradmag_y - gradmag_x))) / 2;
	return wm;
}

/*
 * Paper's Eq(4)
 */
float ETF::computeWd(cv::Vec3f x, cv::Vec3f y) {
	return abs(x.dot(y));
}

void ETF::rotateFlow(Mat& src, Mat& dst, float theta) {
	theta = theta / 180.0 * M_PI;

	for (int i = 0; i < src.rows; i++) {
		for (int j = 0; j < src.cols; j++) {
			Vec3f v = src.at<cv::Vec3f>(i, j);
			float rx = v[0] * cos(theta) - v[1] * sin(theta);
			float ry = v[1] * cos(theta) + v[0] * sin(theta);
			dst.at<cv::Vec3f>(i, j) = Vec3f(rx, ry, 0.0);
		}
	}

}

void ETF::resizeMat(Size s) {
	resize(flowField, flowField, s, 0, 0, CV_INTER_LINEAR);
	resize(refinedETF, refinedETF, s, 0, 0, CV_INTER_LINEAR);
	resize(gradientMag, gradientMag, s, 0, 0, CV_INTER_LINEAR);
}


