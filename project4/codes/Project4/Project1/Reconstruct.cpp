

#include <iostream>  
#include <opencv2/core/core.hpp>  
#include <opencv2/highgui/highgui.hpp>  
#include <opencv2/imgproc/imgproc.hpp>
#include "opencv2/imgproc.hpp"
#include <opencv2/imgcodecs.hpp>
#include<opencv2/opencv.hpp>
#include <time.h>
#include <string>
#include "opencv2/imgproc.hpp"

#define MAX_CHANNEL 3
#define MAX_CORE_X 11
#define MAX_CORE_Y 11
#define MAX_CORE_LENGTH MAX_CORE_X*MAX_CORE_Y
#define pi 3.1415926
#define e 2.7182818

using namespace cv;
using namespace std;

Mat addSaltNoise(const Mat srcImage, int n)
{
	Mat dstImage = srcImage.clone();
	for (int k = 0; k < n; k++)
	{
		//随机取值行列
		int i = rand() % dstImage.rows;
		int j = rand() % dstImage.cols;
		//图像通道判定
		if (dstImage.channels() == 1)
		{
			dstImage.at<uchar>(i, j) = 255;		//盐噪声
		}
		else
		{
			dstImage.at<Vec3b>(i, j)[0] = 255;
			dstImage.at<Vec3b>(i, j)[1] = 255;
			dstImage.at<Vec3b>(i, j)[2] = 255;
		}
	}
	for (int k = 0; k < n; k++)
	{
		//随机取值行列
		int i = rand() % dstImage.rows;
		int j = rand() % dstImage.cols;
		//图像通道判定
		if (dstImage.channels() == 1)
		{
			dstImage.at<uchar>(i, j) = 0;		//椒噪声
		}
		else
		{
			dstImage.at<Vec3b>(i, j)[0] = 0;
			dstImage.at<Vec3b>(i, j)[1] = 0;
			dstImage.at<Vec3b>(i, j)[2] = 0;
		}
	}
	return dstImage;
}
int motionBlurCore(float* core, int width, int height, int dir)
{
	if (height != 1 || width % 2 != 1)
		return 0;

	int count = width / 2 + 1;
	if (dir == 0)
	{
		for (int i = 0; i < count; i++)
			core[i] = 1.0 / count;
		for (int i = count; i < width; i++)
			core[i] = 0.0;
	}
	else {
		for (int i = 0; i < count - 1; i++)
			core[i] = 0.0;
		for (int i = count - 1; i < width; i++)
			core[i] = 1.0 / count;
	}
	return 1;
}

void filter(Mat src, Mat dst, const float core[], int cx, int cy)
{
	int width = src.cols;
	int height = src.rows;
	int channel = src.channels();

	uchar *pout;
	uchar *tmp;

	int line[MAX_CORE_LENGTH] = { 0 };
	int cx2 = cx / 2;
	int cy2 = cy / 2;
	for (int j = 0; j < height; j++)
	{
		pout = dst.ptr<uchar>(j);
		for (int i = 0; i < width; i++)
		{
			float sum[MAX_CHANNEL] = { 0 };

			for (int y = 0; y < cy; y++)
			{
				for (int x = 0; x < cx; x++)
				{
					int tx = i + x - cx2;
					int cp = y * cx + x;

					for (int c = 0; c < channel; c++)
					{
						if (j + y - cy2 < 0)
						{
							tmp = src.ptr<uchar>(0);
							if (tx < 0)
								sum[c] += tmp[c] * core[cp];
							else if (tx >= width)
								sum[c] += tmp[(width - 1)*channel + c] * core[cp];
							else
								sum[c] += tmp[tx*channel + c] * core[cp];
						}
						else if (j + y - cy2 >= height)
						{
							tmp = src.ptr<uchar>(height - 1);
							if (tx < 0)
								sum[c] += tmp[c] * core[cp];
							else if (tx >= width)
								sum[c] += tmp[(width - 1)*channel + c] * core[cp];
							else
								sum[c] += tmp[tx*channel + c] * core[cp];
						}
						else {
							tmp = src.ptr<uchar>(j);
							int ty = (y - cy2)*width*channel;
							if (tx < 0)
								sum[c] += tmp[ty + c] * core[cp];
							else if (tx >= width)
								sum[c] += tmp[ty + (width - 1)*channel + c] * core[cp];
							else
								sum[c] += tmp[ty + tx * channel + c] * core[cp];
						}
					}
				}
			}
			for (int r = 0; r < channel; r++)
			{
				pout[i*channel + r] = (int)sum[r];
			}
		}
	}
}

int main(){
	Mat srcImg = imread("blurred.jpg");

	if (!srcImg.data)
	{
		return -1;
	}
	imshow("source", srcImg);

	//Mat dstImg(srcImg.size(), CV_8UC3);

	//float core[9];
	//int x = 9, y = 1;
	//if (motionBlurCore(core, x, y, 0))
	//	filter(srcImg, dstImg, core, x, y); //恢复的时候要根据模糊图像来

	/*imshow("result", dstImg);
	imwrite("ShaonvBlur.jpg", dstImg);*/
	Mat noise = addSaltNoise(srcImg, 3000);
	imshow("addnoise", noise);
	imwrite("ShaonvBlurNoise.jpg", noise);
	//Mat noBlur = addSaltNoise(srcImg, 3000);
	//imshow("onlyNoise", noBlur);
	waitKey(0);
	return 0;
}