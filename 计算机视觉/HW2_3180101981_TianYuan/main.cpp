#include<opencv2/opencv.hpp>
#include<iostream>
#include<vector>
#include<cmath>
#include <iterator>
#include <valarray>

#define PI 3.1415927

using namespace cv;
using namespace std;


int th1, th2;

vector<Vec2f> lines;
vector<Vec3f> circles;

void findLines(
	Mat image, 
	double pace1, double pace2,double range,
	double thre) {
	// pace1 距离步长
	// pace2 角度步长
	// range 容错范围
	// thre 累加器阈值
	int height = image.rows; int width = image.cols;
	int maxlength = sqrt(height * height + width * width);
	int countLength = (int)(maxlength / pace1 + 1);
	int countTheta = (int)(2 * PI / pace2 + 1);
	vector<vector <double> > counter(countLength, vector<double>(countTheta, 0));
	for (int i = 0; i < height; i ++) {
		for (int j = 0; j < width; j++) {
			if (image.at<uchar>(i, j) <= 0) continue;
			for (int rho = 0; rho * pace1 < maxlength; rho ++) {
				for (int theta = 0; theta * pace2 < 2 * PI; theta ++) {
					if (counter[rho][theta] == -1) continue;
					double temp = rho * pace1 - j * cos(theta * pace2) - i * sin(theta * pace2);
					if (abs(temp) <= range) {
						counter[rho][theta] ++;
						if (counter[rho][theta] >= thre) {
							lines.push_back(Vec2f(rho * pace1, theta * pace2));
							counter[rho][theta] = -1;
						}
					}
				}
			}
		}
	}
}

void findCircles(
	Mat image,  // image输入
	vector<Point> points, // 边缘点坐标
	double pace, // 角度的分割
	double mindist, // 同心圆的半径差
	double thre, // 计数器的阈值
	double minradius = 0, double maxradius = 0) {
	int height = image.rows; int width = image.cols;
	int sizer = (maxradius - minradius) / mindist + 1;
	int sizea = (2 * PI / pace) + 1;
	vector<vector <vector<int> > > counter(height, vector<vector<int>>(width, vector<int>(sizer, 0)));
	int max = 0;
	for (int i = 0; i < points.size(); i++) {
		int x = points[i].x; // rows
		int y = points[i].y; // cols
		for (int r = 0; r < sizer; r++) {
			for (int angle = 0; angle < sizea; angle++) {
				int a = x - (maxradius - r * mindist) * sin(angle * pace);
				int b = y - (maxradius - r * mindist) * cos(angle * pace);
				if (a >= 0 && a < height && b >= 0 && b < width && counter[a][b][r] >= 0) {
					counter[a][b][r] ++;
					if (counter[a][b][r] > max) 
						max = counter[a][b][r];
					if (counter[a][b][r] >= thre) {
						double radius = maxradius - r * mindist;
						circles.push_back(Vec3f(a, b, radius));
						for (int temp1 = 0; temp1 < 3; temp1++) {
							for (int temp2 = 0; temp2 < 3; temp2++) {
								for (int tempr = 0; tempr < 3; tempr++) {
									counter[a+temp1][b+temp2][r-tempr] = -1;
								}
							}
						}
					}
				}
			}
		}
	}
}

Mat myCanny(Mat picOriginal, int th1, int th2) { 
	Mat picEdge; 
	Canny(picOriginal, picEdge, th1, th2); 
	return picEdge; 
}

int main(int argc, char** argv) {

	vector<string> images(3);
	images[0]="hw-highway.jpg";
	images[1]="hw-coin.jpg";
	images[2] ="hw-seal.jpg";
	namedWindow("RESULT", 0);

	for (int imagen = 0; imagen < 3; imagen++) {

		Mat picOriginal;
		Mat gray, gray1;
		Mat binary;
		circles.clear();
		lines.clear();

		picOriginal = imread(images[imagen]);

		int kvalue = 15;
		int maxcolor = 255;

		resize(picOriginal, picOriginal, Size(picOriginal.cols * 0.4, picOriginal.rows * 0.4), 0.4, 0.4);
		cvtColor(picOriginal, gray, COLOR_BGR2GRAY);
		GaussianBlur(gray, gray, Size(3, 3), 3, 3);
		bilateralFilter(gray, gray1, kvalue, kvalue * 2, kvalue / 2);
		
		th1 = 200, th2 = 255;

		binary = myCanny(gray1, th1, th2);
		imshow("RESULT", binary);
		cv::waitKey(0);

		vector<Point> points; // 记录所有边缘点的坐标
		for (int i = 0; i < binary.rows; i++) {
			for (int j = 0; j < binary.cols; j++) {
				if (binary.at<uchar>(i, j) > 0)
					points.push_back(Point(i, j));
			}
		}

		int Min = min(binary.rows, binary.cols);
		int Max = max(binary.rows, binary.cols);

		if(imagen == 2) findCircles(binary, points, 0.1, 5, 25, Min / 7, Max / 2); // for seal
		else if (imagen == 1)findCircles(binary, points, 0.1, 5, 19, Min / 18, Max / 4); // for coin
		for (int i = 0; i < circles.size(); i++) {
			Point center(circles[i].val[1], circles[i].val[0]);
			circle(picOriginal, center, circles[i].val[2], Scalar(255, 0, 0));
		}

		findLines(binary, 2, PI / 32.0, 1, 100);
		for (size_t i = 0; i < lines.size(); i++) {
			Vec2f linex = lines[i];
			float rho = lines[i][0], theta = lines[i][1];
			Point pt1, pt2;
			double a = cos(theta), b = sin(theta);
			double x0 = a * rho, y0 = b * rho; // x = tho cos theta, y = tho sin theta
			pt1.x = cvRound(x0 + 1000 * (-b));
			pt1.y = cvRound(y0 + 1000 * a);
			pt2.x = cvRound(x0 - 1000 * (-b));
			pt2.y = cvRound(y0 - 1000 * (a));
			line(picOriginal, pt1, pt2, cv::Scalar(0, 255, 0), 1);
		}

		imshow("RESULT", picOriginal);
		cv::waitKey(0);
	}
	return 0;
}
