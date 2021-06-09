clear all;
close all;
clc;
 
%img=imread('page33.png');
I=imread('Lina.jpg');
%img=imread('explode.png');
%I=imread('cluster.png');
%img = rgb2gray(img);

figure,imshow(I);
I2=rgb2gray(I);
figure,imshow(I2);
figure,imhist(I2);
[T,SM]=graythresh(I2);
I3=im2bw(I2,T);
figure,imshow(I3);
fprintf('%f %f\n',T*255,SM);