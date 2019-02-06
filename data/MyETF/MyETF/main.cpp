#include "ETF.h"
#include "postProcessing.h"

using namespace cv;

ETF etf;
PP processing;

int main(int argc, char *argv[]) {
	argc--;
	argv++;
	if (argc < 2) {
		fprintf(stderr, "main in.jpg out.jpg\n"); exit(1);
	}

	string file = argv[0];
	Mat originalImg;
	originalImg = imread(file, CV_LOAD_IMAGE_GRAYSCALE);
	etf.initial_ETF(file, originalImg.size());
	int ETF_kernel = 7;
	int times = 3;
	for (int i = 0; i < times; i++) {
		etf.refine_ETF(ETF_kernel);

		Mat dis;
		dis = originalImg.clone();
		cvtColor(dis, dis, CV_GRAY2BGR);
		processing.ETF(etf.flowField, dis);
		dis.convertTo(dis, CV_8UC1, 255);
		cv::cvtColor(dis, dis, CV_GRAY2BGR);

		cv::imwrite(argv[i+1], dis);
	}
		
	return 0;
}