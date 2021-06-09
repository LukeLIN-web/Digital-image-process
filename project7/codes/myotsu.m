clear all;
close all;
clc;
%img=imread('cluster.png');
% img=imread('explode.png');
% img=imread('page33.png');
% img=imread('otsudemo.png');
%img=imread('withnoise.png');
img=imread('smallfront.png');
im=rgb2gray(img);
figure,imshow(im);
[row,rol] = size(im);
h = imhist(im);% 求出直方图
% 概率
p = h / (row*rol);
% 平均灰度
m = zeros(256,1);
% 累加
for i = 2:256
    m(i) = m(i-1)+p(i)*(i-1);
    p(i) = p(i-1)+p(i);
    % 防止除以0 无穷大
    if(p(i-1) == 0)
        p(i-1) = 0.0001;
    end
end
% 全局均值
mg = sum(sum(im))/row/rol;
p1 = ones(256,1);

% 类间方差
a = ((mg*p-m).^2)./(p.*(p1-p));
[max_a,yu] = max(a);
yu-1
T = graythresh(im)
I3= im2bw(im,T);
figure,imshow(I3);