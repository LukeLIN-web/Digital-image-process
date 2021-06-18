%Harris角点提取算法并精确至亚像素级                             
clear;clc;
tic; %计时开始
Image_a = imread('withnoise.png'); % 读取图像
Image_a = rgb2gray(Image_a);% 转化为灰度图像 
Image=Image_a ;
%%%%%%%%%%%%%%%%%%%%%%%%%%%角点从第5行第5列开始找%%%%%%%%%%%%%
Image(1:4,:)=[];
Image(:,end-3:end)=[];
Image(end-3:end,:)=[];
Image(:,1:4)=[];
HdImage=Image;                  % 转化后的灰度图像
fx = [-1 0 1;-1 0 1;-1 0 1]; % x方向的Prewitt算子，用于对x方向滤波 
Ix = filter2(fx,HdImage);     % 对x方向滤波 
fy = [-1 -1 -1;0 0 0;1 1 1]; % y方向的Prewitt算子，用于对y方向滤波 
Iy = filter2(fy,HdImage);% 对y方向滤波 
Ix2 = Ix.^2; % .^2用来求数组的平方
Iy2 = Iy.^2; % .^2用来求数组的平方
Ixy = Ix.*Iy;% 数组相乘
h= fspecial('gaussian',[60 60],2); % 产生9*9的高斯窗口，sigma=2，产生的窗口越大，得到的角点越少（7-11）
A = filter2(h,Ix2);% 用产生的高斯窗口处理Ix2得到A 
B = filter2(h,Iy2);% 用产生的高斯窗口处理Iy2得到B 
C = filter2(h,Ixy); % 用产生的高斯窗口处理Ixy得到C
height = size(HdImage,1); % 计算图像的第一维的元素数，即行数
width = size(HdImage,2);  % 计算图像的第二维的元素数，即列数
CRF = zeros(height,width);  % 生成一个和图像大小一致的全0的double型数组，用来保存角点响应函数值
CRFmax = 0; % 保存图像中最大的角点响应函数值 
%M = [A(i,j) C(i,j);C(i,j) B(i,j)];             
%CRF(i,j) = det(M)-0.05*(trace(M))^2; % 计算角点响应函数值，k的取值一般在0.04--0.06   
CRF=(A.*B - C.^2) - 0.05*(A + B).^2; % 代码的优化把for循环改为向量循环，k=0.05 
CRFmax=max(max(CRF)); % ME01=max(E02)%对一个矩阵的每一列求最大值；找到最大的角点响应函数（用来设置阈值时用）
l=ordfilt2(CRF,10^2,ones(10));% 生成在7*7的窗口进行非最大值抑制（排序滤波器）
k=(l==CRF)&(CRF>0.001*CRFmax);% 设定阈值为0.01*CRFmax,只有是局部最大值并且角点响应函数值大于阈值才是角点
[v1,u1] = find(k); % 找到角点的位置，并保存,m*n图像中，u是n量，纵坐标，v是m量，横坐标。
v=v1+4;
u=u1+4;
count = size(v,1);  % 用来记录角点的个数
disp('检测到的角点个数为：')  
disp(count) % 输出角点个数 
figure,imshow(HdImage);  % 显示灰度图像
hold on; 
plot(u1,v1,'g.');% 在灰度图像上用绿色‘.‘标出角点的位置