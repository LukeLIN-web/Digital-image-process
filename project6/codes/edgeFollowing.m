clear all;
close all;
clc;
 
%img=imread('page33.png');
%img=imread('Lina.jpg');
%img=imread('explode.png');
img=imread('cluster.png');
img = rgb2gray(img);
I=img>128;
imshow(img)
figure();
imshow(I);

%获得图像大小
[M,N]=size(I);
%存放膨胀后的图像
J=zeros(M,N);
%===============================边界提取===================================
%结构元素
n=3;
B=ones(n,n);
n_B=length(find(B==1));
%这里需要B对其原点进行翻转，因为B是对称的，所以翻转后的结果与其本身相同
l_m=floor(n/2);
l_n=floor(n/2);
%腐蚀操作
I_pad=padarray(I,[l_m,l_n],'symmetric');
for x=1:M
    for y=1:N
        %从扩展图像中取出子图像
         Block=I_pad(x:x+2*l_m,y:y+2*l_n);
         %将结构元素与子图像点乘,即逻辑“与”操作
         c=B.*Block;
         %比较结构元素与c中的1的数量，如果一样多，则该点的值为1
         ind=find(c==1);
         if length(ind)==n_B
             J(x,y)=1;
         end
    end
end
Beta=I-J;
figure()
imshow(Beta,[]);