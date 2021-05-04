% 输入灰度级        输出彩色
% 0～63  1/4         蓝色
% 64～127 2/4        紫色
% 128～191 3/4       黄色
% 192～255  4/4      红色
clc;
clear;
close all;
%grayImage=rgb2gray(imread('cluster.png'));
grayImage=rgb2gray(imread('page33.png'));
%grayImage=rgb2gray(imread('Lina.jpg'));
%grayImage=rgb2gray(imread('explode.png'));
figure,imshow(grayImage);
[row,col]=size(grayImage);
range=255;%每个通道的最大灰度值
%x= 0:0.01:3*pi;
%y1=abs(sin(x));
%figure,subplot(3,1,1);
%plot(x,y1);
%y2=abs(sin(x-pi/4));
%subplot(3,1,2);
%plot(x,y2);
%y3=abs(sin(x-pi/2));
%subplot(3,1,3);
%plot(x,y3);
R = zeros(row,col);
G = zeros(row,col);
B = zeros(row,col);
for i=1:row
     for j=1:col
          %R(i,j) = y1(grayImage(i,j)*pi/100);% 100, 我要映射到[0,2.5pi] pi 
          x1 = im2double(grayImage(i,j))*2.5*pi;
          x2 = im2double(grayImage(i,j))*2.5*pi -pi/4;
          x3 = im2double(grayImage(i,j))*2.5*pi -pi/2;
          R(i,j) = abs(sin(x1));
          G(i,j) = abs(sin(x2));
          B(i,j) = abs(sin(x3));
%           G(i,j) = y2(grayImage(i,j)*pi/100);
%           B(i,j) = y3(grayImage(i,j)*pi/100);
    end
end
 
out = zeros(row,col);
for i=1:row
    for j=1:col
              out(i,j,1)=R(i,j);
              out(i,j,2)=G(i,j);
              out(i,j,3)=B(i,j);
    end
end

%out=out/256;
%out = out*255; %*255会太白了
figure,imshow(out);
%% 分段函数
% for i=1:row
%      for j=1:col
%         if grayImage(i,j)<=range/4% [0,64]偏蓝
%             R(i,j)=0;
%             G(i,j)=4*grayImage(i,j);
%             B(i,j)=range;
%             else if grayImage(i,j)<=range/2% (64,128] 偏紫
%                  R(i,j)=0;
%                  G(i,j)=range;
%                  B(i,j)=-4*grayImage(i,j)+2*range;
%                 else if grayImage(i,j)<=3*range/4% (128, 192]
%                         R(i,j)=4*grayImage(i,j)-2*range;
%                         G(i,j)=range;
%                         B(i,j)=0;
%                      else
%                         R(i,j)=range;
%                         G(i,j)=4*(range-grayImage(i,j));
%                         B(i,j)=0;
%                     end
%                 end
%         end
%     end
% end