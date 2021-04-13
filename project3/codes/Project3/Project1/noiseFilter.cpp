
#include <iostream>  
#include <opencv2/core/core.hpp>  
#include <opencv2/highgui/highgui.hpp>  
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>
#include <time.h>
using namespace cv;
// 零填充
cv::Mat image_add_border(cv::Mat &src)
{
	int w = 2 * src.cols;
	int h = 2 * src.rows;
	std::cout << "src: " << src.cols << "*" << src.rows << std::endl;

	cv::Mat padded;
	copyMakeBorder(src, padded, 0, h - src.rows, 0, w - src.cols,
		cv::BORDER_CONSTANT, cv::Scalar::all(0));
	padded.convertTo(padded, CV_32FC1);
	std::cout << "opt: " << padded.cols << "*" << padded.rows << std::endl;
	return padded;
}

//transform to center 中心化
void center_transform(cv::Mat &src)
{
	for (int i = 0; i < src.rows; i++) {
		float *p = src.ptr<float>(i);
		for (int j = 0; j < src.cols; j++) {
			p[j] = p[j] * pow(-1, i + j);
		}
	}
}

//对角线交换内容
void zero_to_center(cv::Mat &freq_plane)
{
	//    freq_plane = freq_plane(Rect(0, 0, freq_plane.cols & -2, freq_plane.rows & -2));
		//这里为什么&上-2具体查看opencv文档
		//其实是为了把行和列变成偶数 -2的二进制是11111111.......10 最后一位是0
	int cx = freq_plane.cols / 2; int cy = freq_plane.rows / 2;//以下的操作是移动图像  (零频移到中心)
	cv::Mat part1_r(freq_plane, cv::Rect(0, 0, cx, cy)); 
	cv::Mat part2_r(freq_plane, cv::Rect(cx, 0, cx, cy));
	cv::Mat part3_r(freq_plane, cv::Rect(0, cy, cx, cy));
	cv::Mat part4_r(freq_plane, cv::Rect(cx, cy, cx, cy));

	cv::Mat tmp;
	part1_r.copyTo(tmp);
	part4_r.copyTo(part1_r);
	tmp.copyTo(part4_r);

	part2_r.copyTo(tmp);
	part3_r.copyTo(part2_r);
	tmp.copyTo(part3_r);
}


void show_spectrum(cv::Mat &complexI)
{
	cv::Mat temp[] = { cv::Mat::zeros(complexI.size(),CV_32FC1),
					  cv::Mat::zeros(complexI.size(),CV_32FC1) };
	//显示频谱图
	cv::split(complexI, temp);
	cv::Mat aa;
	cv::magnitude(temp[0], temp[1], aa);
	//    zero_to_center(aa);
	cv::divide(aa, aa.cols*aa.rows, aa);
	cv::imshow("src_img_spectrum", aa);// 怎么可以显示每个的频谱呢?
}

//频率域滤波
cv::Mat frequency_filter(cv::Mat &padded, cv::Mat &blur){
	cv::Mat plane[] = { padded, cv::Mat::zeros(padded.size(), CV_32FC1) };
	cv::Mat complexIm;

	cv::merge(plane, 2, complexIm);
	cv::dft(complexIm, complexIm);//fourior transform
	show_spectrum(complexIm);

	cv::multiply(complexIm, blur, complexIm);
	cv::idft(complexIm, complexIm, CV_DXT_INVERSE);
	cv::Mat dst_plane[2];
	cv::split(complexIm, dst_plane);
	center_transform(dst_plane[0]);
	//    center_transform(dst_plane[1]);

	cv::magnitude(dst_plane[0], dst_plane[1], dst_plane[0]);
//    center_transform(dst_plane[0]);        //center transform

	return dst_plane[0];
}

//没有提升的核
cv::Mat withoutEnhance_high_kernel(cv::Mat &scr, float D0) {
	cv::Mat gaussian_high_pass(scr.size(), CV_32FC2);
	int row_num = scr.rows;
	int col_num = scr.cols;
	float d0 = 2 * D0 * D0; // 2 *d0 ^2
	for (int i = 0; i < row_num; i++) {
		float *p = gaussian_high_pass.ptr<float>(i);
		for (int j = 0; j < col_num; j++) {
			float d = pow((i - row_num / 2), 2) + pow((j - col_num / 2), 2);//移动到中间
			p[2 * j] = 1 - expf(-d / d0);// 没有提升
			p[2 * j + 1] = 1 - expf(-d / d0); //没有用原函数混合补充
		}
	}

	//cv::Mat temp[] = { cv::Mat::zeros(scr.size(), CV_32FC1),
	//				   cv::Mat::zeros(scr.size(), CV_32FC1) };
	//cv::split(gaussian_high_pass, temp);
	//std::string name = "高斯高通滤波器d0=" + std::to_string(D0);
	//cv::Mat show;
	//cv::normalize(temp[0], show, 1, 0, CV_MINMAX);
	//cv::imshow(name, show);
	return gaussian_high_pass;
}
//没有提升的滤波器
cv::Mat withoutEnhance_highpass_filter(cv::Mat &src, float D0){
	cv::Mat padded = image_add_border(src);
	center_transform(padded);// 零填充, 然后放在中间
	cv::Mat gaussian_kernel = withoutEnhance_high_kernel(padded, D0);
	cv::Mat result = frequency_filter(padded, gaussian_kernel);
	return result;
}


//高斯高通提升滤波器
cv::Mat gaussian_high_kernel(cv::Mat &scr, float D0){
	cv::Mat gaussian_high_pass(scr.size(), CV_32FC2);
	int row_num = scr.rows;
	int col_num = scr.cols;
	float d0 = 2 * D0 * D0; // 2 *d0 ^2
	for (int i = 0; i < row_num; i++){
		float *p = gaussian_high_pass.ptr<float>(i);
		for (int j = 0; j < col_num; j++) {
			float d = pow((i - row_num / 2), 2) + pow((j - col_num / 2), 2);//移动到中间
			p[2 * j] = 0.5 + 0.75*(1 - expf(-d / d0));// a=0.5，b=0.75
			p[2 * j + 1] = 0.5 + 0.75*(1 - expf(-d / d0)); //a = 0.5，b = 0.75
		}
	}

	//cv::Mat temp[] = { cv::Mat::zeros(scr.size(), CV_32FC1),
	//				   cv::Mat::zeros(scr.size(), CV_32FC1) };
	//cv::split(gaussian_high_pass, temp);
	//std::string name = "高斯高通提升滤波器d0=" + std::to_string(D0);
	//cv::Mat show;
	//cv::normalize(temp[0], show, 1, 0, CV_MINMAX);
	//cv::imshow(name, show);
	return gaussian_high_pass;
}

//高斯高通滤波器
cv::Mat gaussian_highpass_filter(cv::Mat &src, float D0){
	cv::Mat padded = image_add_border(src);
	center_transform(padded);// 零填充, 然后放在中间
	cv::Mat gaussian_kernel = gaussian_high_kernel(padded, D0);
	cv::Mat result = frequency_filter(padded, gaussian_kernel);
	return result;
}

int main()
{

	Mat image = imread("Lina.jpg",IMREAD_GRAYSCALE);
	if (image.empty())
		return -1;
	float radius = 2;

	cv::imshow("src", image);
	cv::Mat src_eq;
	cv::equalizeHist(image, src_eq);// 直方图均衡
	cv::imshow("直方图均衡后", src_eq);

	Mat withoutHance_result = withoutEnhance_highpass_filter(image, radius);
	withoutHance_result = withoutHance_result(Rect(0, 0, image.cols, image.rows));
	cv::normalize(withoutHance_result, withoutHance_result, 255, 0, CV_MINMAX);
	withoutHance_result.convertTo(withoutHance_result, CV_8U);
	
	imshow("没有提升的滤波", withoutHance_result);
	//equalizeHist(withoutHance_result, withoutHance_result);

	cv::Mat gaussian_result = gaussian_highpass_filter(image, radius);//高斯高通滤波.
	gaussian_result = gaussian_result(cv::Rect(0, 0, image.cols, image.rows));//变成一个矩形.
	//归一化
	cv::normalize(gaussian_result, gaussian_result, 255, 0, CV_MINMAX);
	gaussian_result.convertTo(gaussian_result, CV_8U);
	
	cv::imshow("高频提升滤波", gaussian_result);
	Mat equal;
	cv::equalizeHist(gaussian_result,equal);// 直方图均衡
	cv::imshow("高频提升滤波再均衡化", gaussian_result);
	cv::waitKey(0);

	return 1;
}

