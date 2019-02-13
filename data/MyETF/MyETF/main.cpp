#include "ETF.h"
#include "postProcessing.h"
#include "hdf5.h"

#define DATASET         "vector_field"
#define DIM0            512
#define DIM1            512

using namespace cv;

ETF etf;
PP processing;

int main(int argc, char *argv[]) {
	argc--;
	argv++;
	if (argc < 1) {
		fprintf(stderr, "main rootDir\n"); exit(1);
	}

	string rootDir = argv[0];
	string fname = rootDir + "\\img_names.txt";
	string inputDir = rootDir + "\\train_gray\\";
	string outputDir = rootDir + "\\vector_fields\\";
	string licDir = rootDir + "\\lic\\";

	FILE* fp = fopen(fname.c_str(), "r");
	char name[260];
	
	while (fscanf(fp, "%s", name) != EOF) {
		string nameStr(name);
		int pos = nameStr.find_last_of(".");
		string name_sub = nameStr.substr(0, pos);
		string file = inputDir + nameStr;
		string vf_file = outputDir + name_sub + ".h5";
		string lic_file = licDir + nameStr;

		Mat originalImg;
		originalImg = imread(file, CV_LOAD_IMAGE_GRAYSCALE);
		etf.initial_ETF(file, originalImg.size());
		int ETF_kernel = 7;
		int times = 3;
		for (int i = 0; i < times; i++) {
			etf.refine_ETF(ETF_kernel);
		}
		etf.getAngle();

		hid_t       hdf5_file, space, dset;          /* Handles */
		herr_t      status;
		hsize_t     dims[2] = { DIM0, DIM1 };

		/*
		* Create a new file using the default properties.
		*/
		hdf5_file = H5Fcreate(vf_file.c_str(), H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT);

		/*
		* Create dataspace.  Setting maximum size to NULL sets the maximum
		* size to be the current size.
		*/
		space = H5Screate_simple(2, dims, NULL);

		/*
		* Create the dataset.  We will use all default properties for this
		* example.
		*/
		dset = H5Dcreate(hdf5_file, DATASET, H5T_NATIVE_FLOAT, space, H5P_DEFAULT,
			H5P_DEFAULT, H5P_DEFAULT);

		/*
		* Write the data to the dataset.
		*/
		status = H5Dwrite(dset, H5T_NATIVE_FLOAT, H5S_ALL, H5S_ALL, H5P_DEFAULT,
			etf.angles);

		/*
		* Close and release resources.
		*/
		status = H5Dclose(dset);
		status = H5Sclose(space);
		status = H5Fclose(hdf5_file);

		Mat dis;
		dis = originalImg.clone();
		cvtColor(dis, dis, CV_GRAY2BGR);
		processing.ETF(etf.flowField, dis);
		dis.convertTo(dis, CV_8UC1, 255);
		cv::cvtColor(dis, dis, CV_GRAY2BGR);

		cv::imwrite(lic_file, dis);
		printf("Info: generate edge tangent flow for %s\n", name);
	}
		
	
	return 0;
}