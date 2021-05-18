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
	string ind = ss.str(); // 获得num转字符串

	cout << "Show the original image.." << endl;
	imshow("RESULT", im); // 展示原始图像
	imwrite(ind + "_original.jpg", im); // 写入文件
	waitKey(0);

	Mat gray; // 灰度图
	cvtColor(im, gray, COLOR_BGR2GRAY);

	double k = 0.04; // k值
	double threshold = 0.01; // 阈值参数设置
	int width = gray.cols;
	int height = gray.rows;
	
	// 计算Ix, Iy
	Mat grad_x, grad_y;
	Sobel(gray, grad_x, CV_16S, 1, 0, ksize);
	Sobel(gray, grad_y, CV_16S, 0, 1, ksize);
	// 计算Ix^2, Iy^2, Ix*Iy
	Mat Ix2, Iy2, Ixy;
	Ix2 = grad_x.mul(grad_x);
	Iy2 = grad_y.mul(grad_y);
	Ixy = grad_x.mul(grad_y);
	// 高斯滤波
	GaussianBlur(Ix2, Ix2, Size(ksize, ksize), 2);
	GaussianBlur(Iy2, Iy2, Size(ksize, ksize), 2);
	GaussianBlur(Ixy, Ixy, Size(ksize, ksize), 2);

	// 计算lambda_max, lambda_min, R
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
			eigen(M, lambda, temp); // 计算特征值到lambda矩阵
			double max = lambda.at<double>(1, 0), min = lambda.at<double>(0, 0); //获得max特征值和min特征值
			if (lambda.at<double>(0, 0) > lambda.at<double>(1, 0)) {
				max = lambda.at<double>(0, 0);
				min = lambda.at<double>(1, 0);
			}
			max_array[i][j] = max; // max特征值
			min_array[i][j] = min; // min特征值
			if (max > lambda_max_max) lambda_max_max = max;
			if (max < lambda_max_min) lambda_max_min = max;
			if (min > lambda_min_max) lambda_min_max = min;
			if (min < lambda_min_min) lambda_min_min = min;
			R[i][j] = (max * min - k * (max + min) * (max + min)); // 计算R值
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

int write(Mat im) { // 显示im，时长为40，如果检测到空格则暂停，esc则退出
	imshow("RESULT", im);
	int key = waitKey(40);
	if (key == 32) { // 一帧显示时间为40，期间如果检测到空格，进入harris过程

		// 开始处理图片并显示
		harris_corner(im, ++number);

		int key1 = waitKey(0); // 等待键盘输入
		while (key1 != 32) { // 一直等待键盘输入，直到有空格
			if (key1 == 27) { // 如果暂停期间按到esc键
				goto here1; // 退出
			}
			key1 = waitKey(0);
		}
	}
	else if (key == 27) { // 如果按下esc键
	here1:;
		destroyAllWindows();
		return 1;
	}
	return 0;
}

int main() {
	namedWindow("RESULT", 1); // 创建RESULT窗口
	VideoCapture capture(0); // 摄像头捕获画面
	Mat frame; // 捕获画面到frame
	while (capture.read(frame)) { // 捕获画面成功时
 		int i = write(frame); // 进入write交互函数
		if (i == 1) break; // 如果用户选择退出程序，退出
	}
	destroyAllWindows(); // 关闭所有窗口
	return 0;
}