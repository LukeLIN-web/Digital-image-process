
#include <iostream>  
#include <opencv2/core/core.hpp>  
#include <opencv2/highgui/highgui.hpp>  
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>
using namespace cv;

Mat LinerInter(Mat &srcImage, double kx, double ky);
Mat Downsampling(Mat &srcImage, double kx, double ky);
Mat ThreeInter(Mat &srcImage);
float a = -0.5;//BiCubic������
void getW_x(float w_x[4], float x);
void getW_y(float w_y[4], float y);
Mat NearInter(Mat &srcImage, double kx, double ky){
	int rows = cvRound(srcImage.rows*kx);// 0.6��, �߱�Ϊ0.6
	int cols = cvRound(srcImage.cols*ky);// ���Ϊ1.2
	Mat resultImg(rows, cols, srcImage.type());//OpenCV2.0��װ��һ��C++�࣬������ʾһ��ͼ���Զ��ڴ����
	int i, j, x, y;
	for (i = 0; i < rows; i++){
		x = static_cast<int>((i + 1) / kx + 0.5) - 1;// static_cast<int>(i / kx + 0.5)  �������Ҳ��֪��Ϊɶ
		for (j = 0; j < cols; j++){
			y = static_cast<int>((j + 1) / ky + 0.5) - 1;//
			resultImg.at<Vec3b>(i, j) = srcImage.at<Vec3b>(x, y);//Ŀ������ص�ĻҶ�ֵ����Դͼ�����������ڽ����صĻҶ�ֵ��
		}
	}
	return resultImg;
}
int main(){
	Mat dest1, dest2;
	Mat srcImg = imread("Lina.jpg");//OpencvĬ�Ͻ�����ͼ��ǿ��ת��Ϊһ����ͨ����ɫͼ��
	namedWindow("ͼƬ");// ����һ����Ϊ "ͼƬ"���� ,��һ��������ʾ�´��ڵ����ƣ���ʾ�ڴ��ڵĶ�����ͬʱ����highgui�������������øô��ڵľ����
	imshow("ͼƬ", srcImg);// �ڴ�������ʾͼƬ   
	dest1 = Downsampling(srcImg, 0.5, 0.5);
	namedWindow("��������1", CV_WINDOW_AUTOSIZE);
	imshow("��������1", dest1);
	Mat nearImg = NearInter(dest1, 2, 2);
	Mat doubleImg = LinerInter(dest1, 2, 2);
	Mat ThreeInterImg = ThreeInter(dest1);
	imshow("nearImg", nearImg);
	imshow("doubleImg", doubleImg);
	imshow("ThreeInterImg", ThreeInterImg);
	waitKey(0); //����ʧ
	return 0;
}

Mat ThreeInter(Mat &image) {
	float Row_B = image.rows * 2;// �߷Ŵ�����
	float Col_B = image.cols * 2;

	Mat resImage(Row_B, Col_B, CV_8UC3);

	for (int i = 2; i < Row_B - 4; i++) {
		for (int j = 2; j < Col_B - 4; j++) {
			float x = i * (image.rows / Row_B);//�Ŵ���ͼ�������λ�������Դͼ���λ��
			float y = j * (image.cols / Col_B);//�õ�B��X,Y����ͼ��A�ж�Ӧ��λ�ã�x,y��=(X*(m/M),Y*(N/n))

			float w_x[4], w_y[4];//���з���ļ�Ȩϵ��
			getW_x(w_x, x);//������ѡ��Ļ������������Ӧ��ÿ�����ص�Ȩֵ
			getW_y(w_y, y);

			Vec3f temp = { 0, 0, 0 };
			for (int s = 0; s <= 3; s++) {
				for (int t = 0; t <= 3; t++) {
					temp = temp + (Vec3f)(image.at<Vec3b>(int(x) + s - 1, int(y) + t - 1))*w_x[s] * w_y[t];
				}
			}
			resImage.at<Vec3b>(i, j) = (Vec3b)temp;
		}
	}
	return resImage;
}

Mat Downsampling(Mat &srcImage,double kx,double ky ) {
	int rows = cvRound(srcImage.rows*kx);
	int cols = cvRound(srcImage.cols*ky);
	Mat resultImg(rows, cols, srcImage.type());//OpenCV2.0��װ��һ��C++�࣬������ʾһ��ͼ���Զ��ڴ����
	int i, j, x, y;
	for (i = 0; i < rows; i++) {
		x = static_cast<int>((i + 1) / kx + 0.5) - 1;// x = 
		for (j = 0; j < cols; j++) {
			y = static_cast<int>((j + 1) / ky + 0.5) - 1;//
			resultImg.at<Vec3b>(i, j) = srcImage.at<Vec3b>(x, y);//
		}
	}
	return resultImg;
}

Mat LinerInter(Mat &srcImage, double kx, double ky){
	int rows = cvRound(srcImage.rows*kx);
	int cols = cvRound(srcImage.cols*ky);
	Mat resultImg(rows, cols, srcImage.type());//Ҳ������Mat resImage(Row_B, Col_B, CV_8UC3);RGB3ͨ������CV_8UC3
	int i, j;
	double dx, dy;

	for (i = 0; i < rows; i++){
		// �õ���ԭͼ�еĵ�
		int xi = (int)(i / kx);//���
		int x11 = xi + 1; //�ұ�
		dx = i / kx - xi;// dx=  int(i / kx) -xi
		for (j = 0; j < cols; j++){
			int yi = (int)(j / ky);
			int y11 = yi + 1;
			dy = j / ky - yi;
			//�жϱ߽�
			if (x11 > (srcImage.rows - 1))
				x11 = xi - 1;
			if (y11 > (srcImage.cols - 1))
				y11 = yi - 1;
			for (int k = 0; k < 3; k++) {
				resultImg.at<Vec3b>(i, j)[k] = (int)(srcImage.at<Vec3b>(xi, yi)[k] * (1 - dx)*(1 - dy)
					+ srcImage.at<Vec3b>(x11, yi)[k] * dx*(1 - dy)
					+ srcImage.at<Vec3b>(xi, y11)[k] * (1 - dx)*dy
					+ srcImage.at<Vec3b>(x11, y11)[k] * dx*dy );
			}//f(x,y)=f(0,0)(1-x)(1-y)+f(1,0)x(1-y)+f(0,1)(1-x)y+f(1,1)xy
		}
	}
	return resultImg;
}


/*����ϵ��*/
void getW_x(float w_x[4], float x) {
	int X = (int)x;//ȡ��������, �������ҵ�һ��������
	float stemp_x[4];
	stemp_x[0] = 1 + (x - X);
	stemp_x[1] = x - X;
	stemp_x[2] = 1 - (x - X);
	stemp_x[3] = 2 - (x - X);
	// ���ݹ�ʽ���� ϵ��
	w_x[0] = a * abs(stemp_x[0] * stemp_x[0] * stemp_x[0]) - 5 * a*stemp_x[0] * stemp_x[0] + 8 * a*abs(stemp_x[0]) - 4 * a;// x<2
	w_x[1] = (a + 2)*abs(stemp_x[1] * stemp_x[1] * stemp_x[1]) - (a + 3)*stemp_x[1] * stemp_x[1] + 1;// x< 1
	w_x[2] = (a + 2)*abs(stemp_x[2] * stemp_x[2] * stemp_x[2]) - (a + 3)*stemp_x[2] * stemp_x[2] + 1;// x< 1
	w_x[3] = a * abs(stemp_x[3] * stemp_x[3] * stemp_x[3]) - 5 * a*stemp_x[3] * stemp_x[3] + 8 * a*abs(stemp_x[3]) - 4 * a;// x<2
}
void getW_y(float w_y[4], float y) {
	int Y = (int)y;
	float stemp_y[4];
	stemp_y[0] = 1.0 + (y - Y);
	stemp_y[1] = y - Y;
	stemp_y[2] = 1.0 - (y - Y);
	stemp_y[3] = 2.0 - (y - Y);
	// ���ݹ�ʽ���� ϵ��
	w_y[0] = a * abs(stemp_y[0] * stemp_y[0] * stemp_y[0]) - 5 * a*stemp_y[0] * stemp_y[0] + 8 * a*abs(stemp_y[0]) - 4 * a;// x<2
	w_y[1] = (a + 2)*abs(stemp_y[1] * stemp_y[1] * stemp_y[1]) - (a + 3)*stemp_y[1] * stemp_y[1] + 1;// x  <1
	w_y[2] = (a + 2)*abs(stemp_y[2] * stemp_y[2] * stemp_y[2]) - (a + 3)*stemp_y[2] * stemp_y[2] + 1;// x< 1
	w_y[3] = a * abs(stemp_y[3] * stemp_y[3] * stemp_y[3]) - 5 * a*stemp_y[3] * stemp_y[3] + 8 * a*abs(stemp_y[3]) - 4 * a; // x<2
}