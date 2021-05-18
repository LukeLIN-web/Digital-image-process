#include<opencv2/opencv.hpp>
#include<iostream>
#include<vector>
#include<string>
#include<sstream>

using namespace cv;
using namespace std;

int number = 0;

void harris_corner(Mat im, int num, int ksize = 3) {

	stringstream ss;
	ss << num;
	string ind = ss.str(); // ���numת�ַ���

	cout << "Show the original image.." << endl;
	imshow("RESULT", im); // չʾԭʼͼ��
	imwrite(ind + "_original.jpg", im); // д���ļ�
	waitKey(0);

	Mat gray; // �Ҷ�ͼ
	cvtColor(im, gray, COLOR_BGR2GRAY);

	double k = 0.04; // kֵ
	double threshold = 0.01; // ��ֵ��������
	int width = gray.cols;
	int height = gray.rows;
	
	// ����Ix, Iy
	Mat grad_x, grad_y;
	Sobel(gray, grad_x, CV_16S, 1, 0, ksize);
	Sobel(gray, grad_y, CV_16S, 0, 1, ksize);
	// ����Ix^2, Iy^2, Ix*Iy
	Mat Ix2, Iy2, Ixy;
	Ix2 = grad_x.mul(grad_x);
	Iy2 = grad_y.mul(grad_y);
	Ixy = grad_x.mul(grad_y);
	// ��˹�˲�
	GaussianBlur(Ix2, Ix2, Size(ksize, ksize), 2);
	GaussianBlur(Iy2, Iy2, Size(ksize, ksize), 2);
	GaussianBlur(Ixy, Ixy, Size(ksize, ksize), 2);

	// ����lambda_max, lambda_min, R
	double lambda_max_max = 0, lambda_max_min = 0, lambda_min_max = 0, lambda_min_min = 20, R_max = 0, R_min = 0;
	Mat lambda_max(gray.size(), gray.type());
	Mat lambda_min(gray.size(), gray.type());
	Mat R_im(gray.size(), gray.type());
	vector<vector<double>> R(height, vector<double>(width, 0));
	vector<vector<double>> max_array(height, vector<double>(width, 0));
	vector<vector<double>> min_array(height, vector<double>(width, 0));

	cout << "Calculate the eigen values..." << endl;
	for (int i = 0; i < height; i++) {
		for (int j = 0; j < width; j++) {
			double Marray[2][2] = { Ix2.at<short>(i, j), Ixy.at<short>(i, j), Ixy.at<short>(i, j), Iy2.at<short>(i, j) };
			Mat M(2, 2, CV_64FC1, Marray);
			Mat lambda, temp;
			eigen(M, lambda, temp); // ��������ֵ��lambda����
			double max = lambda.at<double>(1, 0), min = lambda.at<double>(0, 0); //���max����ֵ��min����ֵ
			if (lambda.at<double>(0, 0) > lambda.at<double>(1, 0)) {
				max = lambda.at<double>(0, 0);
				min = lambda.at<double>(1, 0);
			}
			max_array[i][j] = max; // max����ֵ
			min_array[i][j] = min; // min����ֵ
			if (max > lambda_max_max) lambda_max_max = max;
			if (max < lambda_max_min) lambda_max_min = max;
			if (min > lambda_min_max) lambda_min_max = min;
			if (min < lambda_min_min) lambda_min_min = min;
			R[i][j] = (max * min - k * (max + min) * (max + min)); // ����Rֵ
			if (R[i][j] > R_max) R_max = R[i][j];
			if (R[i][j] < R_min) R_min = R[i][j];
		}
	}

	cout << "Write the eigen values and R values into image.." << endl;
	for (int i = 0; i < height; i++) {
		for (int j = 0; j < width; j++) {
			//lambda_max.at<uchar>(i, j) = (uchar)((max_array[i][j] - lambda_max_min) * 1.0 / (lambda_max_max - lambda_max_min) * 1.0 * 255);
			//lambda_min.at<uchar>(i, j) = (uchar)((min_array[i][j] - lambda_min_min) * 1.0 / (lambda_min_max - lambda_min_min) * 1.0 * 255);
			//R_im.at<uchar>(i, j) = (uchar)((R[i][j] - R_min) * 1.0 / (R_max-R_min) * 1.0 * 255);
			R_im.at<uchar>(i, j) = (uchar)(R[i][j] * 1.0 / R_max * 1.0 * 255);
			lambda_max.at<uchar>(i, j) = (uchar)(max_array[i][j] * 1.0/ lambda_max_max * 1.0 * 255);
			lambda_min.at<uchar>(i, j) = (uchar)(min_array[i][j] * 1.0/ lambda_min_max * 1.0 * 255);
			if (R[i][j] > R_max * threshold) { // corner
				im.at<Vec3b>(i, j)[0] = 0;
				im.at<Vec3b>(i, j)[1] = 0;
				im.at<Vec3b>(i, j)[2] = 255; // red
			}
		}
	}
	Mat colorR;
	applyColorMap(R_im, colorR, COLORMAP_JET);
	

	cout << "Now you can press space to show the next result .." << endl;
	cout << "lambda_max image: " << endl;
	imshow("RESULT", lambda_max);
	imwrite(ind + "_lambda_max.jpg", lambda_max);
	waitKey(0);
	cout << "lambda_min image: " << endl;
	imshow("RESULT", lambda_min);
	imwrite(ind + "_lambda_min.jpg", lambda_min);
	waitKey(0);
	cout << "R image: " << endl;
	imshow("RESULT", colorR);
	imwrite(ind + "_colorR.jpg", colorR);
	waitKey(0);
	cout << "Harris corner result: " << endl;
	imshow("RESULT", im);
	imwrite(ind + "_Corner.jpg", im);
}

int write(Mat im) { // ��ʾim��ʱ��Ϊ40�������⵽�ո�����ͣ��esc���˳�
	imshow("RESULT", im);
	int key = waitKey(40);
	if (key == 32) { // һ֡��ʾʱ��Ϊ40���ڼ������⵽�ո񣬽���harris����

		// ��ʼ����ͼƬ����ʾ
		harris_corner(im, ++number);

		int key1 = waitKey(0); // �ȴ���������
		while (key1 != 32) { // һֱ�ȴ��������룬ֱ���пո�
			if (key1 == 27) { // �����ͣ�ڼ䰴��esc��
				goto here1; // �˳�
			}
			key1 = waitKey(0);
		}
	}
	else if (key == 27) { // �������esc��
	here1:;
		destroyAllWindows();
		return 1;
	}
	return 0;
}

int main() {
	namedWindow("RESULT", 1); // ����RESULT����
	VideoCapture capture(0); // ����ͷ������
	Mat frame; // �����浽frame
	while (capture.read(frame)) { // ������ɹ�ʱ
 		int i = write(frame); // ����write��������
		if (i == 1) break; // ����û�ѡ���˳������˳�
	}
	destroyAllWindows(); // �ر����д���
	return 0;
}