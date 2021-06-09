
#include <iostream>  
#include <opencv2/core/core.hpp>  
#include <opencv2/highgui/highgui.hpp>  
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>
#include <time.h>
using namespace cv;

void ColorSalt(Mat& image, int n);
void ColorPepper(Mat& image, int n);
void AverFiltering(const Mat &src, Mat &dst);
void MedianFlitering(const Mat &src, Mat &dst);
uchar Median(uchar n1, uchar n2, uchar n3, uchar n4, uchar n5,
	uchar n6, uchar n7, uchar n8, uchar n9);

int main(){
	Mat dest1, dest2;
	Mat srcImg = imread("Lina.jpg");//Opencv默认将读入图像强制转换为一幅三通道彩色图像，
	Mat scr = srcImg.clone();
	namedWindow("图片");// 创建一个名为 "图片"窗口 ,第一个参数表示新窗口的名称，显示在窗口的顶部，同时用作highgui中其他函数调用该窗口的句柄。
	imshow("图片", srcImg);// 在窗口中显示原来图片   
	ColorSalt(scr, 5000);  //加入白盐噪声  
	ColorPepper(scr, 1000); //加入黑椒噪声  
	imshow("带噪声的图像", scr);
	//namedWindow("降采样后1", CV_WINDOW_AUTOSIZE);
	Mat ave = imread("Lina.jpg", IMREAD_GRAYSCALE);
	AverFiltering(ave, dest1);
	//AverFiltering(scr, dest1);
	imshow("average filter", dest1);
	// MedianFlitering(scr, dest2);
	// imshow("MedianFlitering", dest2);
	waitKey(0); //不消失
	return 0;
}


void MedianFlitering(const Mat &src, Mat &dst) {
	if (!src.data)return;
	Mat _dst(src.size(), src.type());
	for (int i = 0; i < src.rows; ++i)
		for (int j = 0; j < src.cols; ++j) {
			if ((i - 1) > 0 && (i + 1) < src.rows && (j - 1) > 0 && (j + 1) < src.cols) {
				_dst.at<Vec3b>(i, j)[0] = Median(src.at<Vec3b>(i, j)[0], src.at<Vec3b>(i + 1, j + 1)[0],
					src.at<Vec3b>(i + 1, j)[0], src.at<Vec3b>(i, j + 1)[0], src.at<Vec3b>(i + 1, j - 1)[0],
					src.at<Vec3b>(i - 1, j + 1)[0], src.at<Vec3b>(i - 1, j)[0], src.at<Vec3b>(i, j - 1)[0],
					src.at<Vec3b>(i - 1, j - 1)[0]);
				_dst.at<Vec3b>(i, j)[1] = Median(src.at<Vec3b>(i, j)[1], src.at<Vec3b>(i + 1, j + 1)[1],
					src.at<Vec3b>(i + 1, j)[1], src.at<Vec3b>(i, j + 1)[1], src.at<Vec3b>(i + 1, j - 1)[1],
					src.at<Vec3b>(i - 1, j + 1)[1], src.at<Vec3b>(i - 1, j)[1], src.at<Vec3b>(i, j - 1)[1],
					src.at<Vec3b>(i - 1, j - 1)[1]);
				_dst.at<Vec3b>(i, j)[2] = Median(src.at<Vec3b>(i, j)[2], src.at<Vec3b>(i + 1, j + 1)[2],
					src.at<Vec3b>(i + 1, j)[2], src.at<Vec3b>(i, j + 1)[2], src.at<Vec3b>(i + 1, j - 1)[2],
					src.at<Vec3b>(i - 1, j + 1)[2], src.at<Vec3b>(i - 1, j)[2], src.at<Vec3b>(i, j - 1)[2],
					src.at<Vec3b>(i - 1, j - 1)[2]);
			}
			else
				_dst.at<Vec3b>(i, j) = src.at<Vec3b>(i, j);
		}
	_dst.copyTo(dst);//拷贝
}

uchar Median(uchar n1, uchar n2, uchar n3, uchar n4, uchar n5,
	uchar n6, uchar n7, uchar n8, uchar n9) {
	uchar arr[9];
	arr[0] = n1;
	arr[1] = n2;
	arr[2] = n3;
	arr[3] = n4;
	arr[4] = n5;
	arr[5] = n6;
	arr[6] = n7;
	arr[7] = n8;
	arr[8] = n9;
	for (int gap = 9 / 2; gap > 0; gap /= 2)//希尔排序
		for (int i = gap; i < 9; ++i)
			for (int j = i - gap; j >= 0 && arr[j] > arr[j + gap]; j -= gap)
				swap(arr[j], arr[j + gap]);
	return arr[4];//返回中值
}


void AverFiltering(const Mat &src, Mat &dst) {
	if (!src.data) return;
	//at访问像素点
	dst = Mat::zeros(src.size(), src.type());// 不初始化会报错
	for (int i = 1; i < src.cols; ++i)
		for (int j = 1; j < src.rows; ++j) {
			if ((i - 1 >= 0) && (j - 1) >= 0 && (i + 1) < src.rows && (j + 1) < src.cols) {//边缘不进行处理
				uchar aver;
				aver = (src.at<uchar>(i, j) + src.at<uchar>(i - 1, j - 1) + src.at<uchar>(i - 1, j) + src.at<uchar>(i, j - 1) +
					src.at<uchar>(i - 1, j + 1) + src.at<uchar>(i + 1, j - 1) + src.at<uchar>(i + 1, j + 1) + src.at<uchar>(i, j + 1) +
					src.at<uchar>(i + 1, j)) / 9;
				//aver[0] = (src.at<Vec3b>(i, j)[0] + src.at<Vec3b>(i - 1, j - 1)[0] + src.at<Vec3b>(i - 1, j)[0] + src.at<Vec3b>(i, j - 1)[0] +
				//	src.at<Vec3b>(i - 1, j + 1)[0] + src.at<Vec3b>(i + 1, j - 1)[0] + src.at<Vec3b>(i + 1, j + 1)[0] + src.at<Vec3b>(i, j + 1)[0] +
				//	src.at<Vec3b>(i + 1, j)[0]) / 9;
				//aver[1] = (src.at<Vec3b>(i, j)[1] + src.at<Vec3b>(i - 1, j - 1)[1] + src.at<Vec3b>(i - 1, j)[1] + src.at<Vec3b>(i, j - 1)[1] +
				//	src.at<Vec3b>(i - 1, j + 1)[1] + src.at<Vec3b>(i + 1, j - 1)[1] + src.at<Vec3b>(i + 1, j + 1)[1] + src.at<Vec3b>(i, j + 1)[1] +
				//	src.at<Vec3b>(i + 1, j)[1]) / 9;
				//aver[2] = (src.at<Vec3b>(i, j)[2] + src.at<Vec3b>(i - 1, j - 1)[2] + src.at<Vec3b>(i - 1, j)[2] + src.at<Vec3b>(i, j - 1)[2] +
				//	src.at<Vec3b>(i - 1, j + 1)[2] + src.at<Vec3b>(i + 1, j - 1)[2] + src.at<Vec3b>(i + 1, j + 1)[2] + src.at<Vec3b>(i, j + 1)[2] +
				//	src.at<Vec3b>(i + 1, j)[2]) / 9;
				dst.at<uchar>(i, j) = aver;
				//dst.at<Vec3b>(i, j)[0] = aver[0];// 这里会异常
				//dst.at<Vec3b>(i, j)[1] = aver[1];
				//dst.at<Vec3b>(i, j)[2] = aver[2];
			}
			else {//边缘赋值
				//dst.at<Vec3b>(i, j)[0] = src.at<Vec3b>(i, j)[0];
				//dst.at<Vec3b>(i, j)[1] = src.at<Vec3b>(i, j)[1];
				//dst.at<Vec3b>(i, j)[2] = src.at<Vec3b>(i, j)[2];
			}
		}
}

void ColorSalt(Mat& image, int n)//本函数加入彩色盐噪声  
{
	srand((unsigned)time(NULL));
	for (int k = 0; k < n; k++)//将图像中n个像素随机置零  
	{
		int i = rand() % image.cols;
		int j = rand() % image.rows; //RAND_MAX
			//将图像颜色随机改变  
		image.at<Vec3b>(j, i)[0] = 250;
		image.at<Vec3b>(j, i)[1] = 150;
		image.at<Vec3b>(j, i)[2] = 250;
	}
}

void ColorPepper(Mat& image, int n)//本函数加入彩色椒噪声  
{
	srand((unsigned)time(NULL));
	for (int k = 0; k < n; k++) {
		int i = rand() % image.cols;
		int j = rand() % image.rows;
		//将图像颜色随机改变  
		image.at<Vec3b>(j, i)[0] = 250;
		image.at<Vec3b>(j, i)[1] = 150;
		image.at<Vec3b>(j, i)[2] = 50;
	}
}
