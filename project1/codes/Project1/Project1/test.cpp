
#include <iostream>  
#include <opencv2/core/core.hpp>  
#include <opencv2/highgui/highgui.hpp>  
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>
using namespace cv;

float Bicubic(float x);
Mat ThreeInter(Mat &srcImage, double kx, double ky);
Mat LinerInter(Mat &srcImage, double kx, double ky);
Mat NearInter(Mat &srcImage, double kx, double ky)
{
	int rows = cvRound(srcImage.rows*kx);
	int cols = cvRound(srcImage.cols*ky);
	Mat resultImg(rows, cols, srcImage.type());//OpenCV2.0封装的一个C++类，用来表示一个图像，自动内存管理
	int i, j, x, y;
	for (i = 0; i < rows; i++){
		x = static_cast<int>((i + 1) / kx + 0.5) - 1;
		for (j = 0; j < cols; j++){
			y = static_cast<int>((j + 1) / ky + 0.5) - 1;
			resultImg.at<Vec3b>(i, j) = srcImage.at<Vec3b>(x, y);
		}
	}
	return resultImg;
}
int main()
{
	// 读入一张图片（poyanghu缩小图）    
	Mat dest1, dest2;
	Mat srcImg = imread("Lina.jpg");//Opencv默认将读入图像强制转换为一幅三通道彩色图像，
	// 创建一个名为 "图片"窗口    
	namedWindow("图片");//第一个参数表示新窗口的名称，显示在窗口的顶部，同时用作highgui中其他函数调用该窗口的句柄。
	// 在窗口中显示图片   
	imshow("图片", srcImg);
	// 等待6000 ms后窗口自动关闭    
	//namedWindow("降采样后1", CV_WINDOW_AUTOSIZE);
	//imshow("降采样后1", dest1);
	Mat nearImg = NearInter(srcImg, 0.6, 1.2);
	Mat doubleImg = LinerInter(srcImg, 0.6, 1.2);
	Mat ThreeInterImg = ThreeInter(srcImg, 0.6, 1.2);
	imshow("nearImg", nearImg);
	imshow("resultImg", ThreeInterImg);
	imshow("doubleImg", doubleImg);
	waitKey(0);
	return 0;
}




Mat LinerInter(Mat &srcImage, double kx, double ky)
{
	int rows = cvRound(srcImage.rows*kx);
	int cols = cvRound(srcImage.cols*ky);
	Mat resultImg(rows, cols, srcImage.type());
	int i, j;
	int xi;
	int yi;
	int x11;
	int y11;
	double xm;
	double ym;
	double dx;
	double dy;

	for (i = 0; i < rows; i++)
	{
		xm = i / kx;
		xi = (int)xm;
		x11 = xi + 1;
		dx = xm - xi;
		for (j = 0; j < cols; j++)
		{
			ym = j / ky;
			yi = (int)ym;
			y11 = yi + 1;
			dy = ym - yi;
			//判断边界
			if (x11 > (srcImage.rows - 1))
			{
				x11 = xi - 1;
			}
			if (y11 > (srcImage.cols - 1))
			{
				y11 = yi - 1;
			}
			//bgr
			resultImg.at<Vec3b>(i, j)[0] = (int)(srcImage.at<Vec3b>(xi, yi)[0] * (1 - dx)*(1 - dy)
				+ srcImage.at<Vec3b>(x11, yi)[0] * dx*(1 - dy)
				+ srcImage.at<Vec3b>(xi, y11)[0] * (1 - dx)*dy
				+ srcImage.at<Vec3b>(x11, y11)[0] * dx*dy);
			resultImg.at<Vec3b>(i, j)[1] = (int)(srcImage.at<Vec3b>(xi, yi)[1] * (1 - dx)*(1 - dy)
				+ srcImage.at<Vec3b>(x11, yi)[1] * dx*(1 - dy)
				+ srcImage.at<Vec3b>(xi, y11)[1] * (1 - dx)*dy
				+ srcImage.at<Vec3b>(x11, y11)[1] * dx*dy);
			resultImg.at<Vec3b>(i, j)[2] = (int)(srcImage.at<Vec3b>(xi, yi)[2] * (1 - dx)*(1 - dy)
				+ srcImage.at<Vec3b>(x11, yi)[2] * dx*(1 - dy)
				+ srcImage.at<Vec3b>(xi, y11)[2] * (1 - dx)*dy
				+ srcImage.at<Vec3b>(x11, y11)[2] * dx*dy);
		}

	}
	return resultImg;
}

float Bicubic(float y)
{
	float x = abs(y);
	float a = -0.5;
	if (x <= 1.0)
	{
		return (a + 2)*pow(x, 3) - (a + 3)*pow(x, 2) + 1;
	}
	else if (x < 2.0)
	{
		return a * pow(x, 3) + 5 * a*pow(x, 2) - 4 * a;
	}
	else
	{
		return 0.0;
	}
}

Mat ThreeInter(Mat &srcImage, double kx, double ky)
{
	int rows = cvRound(srcImage.rows * kx);
	int cols = cvRound(srcImage.cols * ky);
	Mat resultImg(rows, cols, srcImage.type());
	int i, j;
	int xm, ym;
	int x0, y0, xi, yi, x1, y1, x2, y2;
	float wx0, wy0, wxi, wyi, wx1, wy1, wx2, wy2;
	float w00, w01, w02, w0i, w10, w11, w12, w1i, w20, w21, w22, w2i, wi0, wi1, wi2, wii;
	for (i = 0; i < rows; i++)
	{
		xm = i / kx;
		xi = (int)xm;
		x0 = xi - 1;
		x1 = xi + 1;
		x2 = xi + 2;
		wx0 = Bicubic(x0 - xm);
		wxi = Bicubic(xi - xm);
		wx1 = Bicubic(x1 - xm);
		wx2 = Bicubic(x2 - xm);
		for (j = 0; j < cols; j++)
		{
			ym = j / ky;
			yi = (int)ym;
			y0 = yi - 1;
			y1 = yi + 1;
			y2 = yi + 2;
			wy0 = Bicubic(y0 - ym);
			wyi = Bicubic(yi - ym);
			wy1 = Bicubic(y1 - ym);
			wy2 = Bicubic(y2 - ym);
			w00 = wx0 * wy0;
			w01 = wx0 * wy1;
			w02 = wx0 * wy2;
			w0i = wx0 * wyi;
			w10 = wx1 * wy0;
			w11 = wx1 * wy1;
			w12 = wx1 * wy2;
			w1i = wx1 * wyi;
			w20 = wx2 * wy0;
			w21 = wx2 * wy1;
			w22 = wx2 * wy2;
			w2i = wx2 * wyi;
			wi0 = wxi * wy0;
			wi1 = wxi * wy1;
			wi2 = wxi * wy2;
			wii = wxi * wyi;
			if ((x0 >= 0) && (x2 < srcImage.rows) && (y0 >= 0) && (y2 < srcImage.cols))
			{
				resultImg.at<Vec3b>(i, j) = (srcImage.at<Vec3b>(x0, y0)*w00 + srcImage.at<Vec3b>(x0, y1)*w01 + srcImage.at<Vec3b>(x0, y2)*w02 + srcImage.at<Vec3b>(x0, yi)*w0i
					+ srcImage.at<Vec3b>(x1, y0)*w10 + srcImage.at<Vec3b>(x1, y1)*w11 + srcImage.at<Vec3b>(x1, y2)*w12 + srcImage.at<Vec3b>(x1, yi)*w1i
					+ srcImage.at<Vec3b>(x2, y0)*w20 + srcImage.at<Vec3b>(x2, y1)*w21 + srcImage.at<Vec3b>(x2, y2)*w22 + srcImage.at<Vec3b>(x2, yi)*w2i
					+ srcImage.at<Vec3b>(xi, y0)*wi0 + srcImage.at<Vec3b>(xi, y1)*wi1 + srcImage.at<Vec3b>(xi, y2)*wi2 + srcImage.at<Vec3b>(xi, yi)*wii);
			}
		}
	}
	return resultImg;
}