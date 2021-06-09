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
h = imhist(im);% ���ֱ��ͼ
% ����
p = h / (row*rol);
% ƽ���Ҷ�
m = zeros(256,1);
% �ۼ�
for i = 2:256
    m(i) = m(i-1)+p(i)*(i-1);
    p(i) = p(i-1)+p(i);
    % ��ֹ����0 �����
    if(p(i-1) == 0)
        p(i-1) = 0.0001;
    end
end
% ȫ�־�ֵ
mg = sum(sum(im))/row/rol;
p1 = ones(256,1);

% ��䷽��
a = ((mg*p-m).^2)./(p.*(p1-p));
[max_a,yu] = max(a);
yu-1
T = graythresh(im)
I3= im2bw(im,T);
figure,imshow(I3);