close all;
clear all;
clc;
% Display the original image.
I = im2double(imread('shaonv.jfif'));
[hei,wid,~] = size(I);
figure();
imshow(I);
title('Original Image');

% Simulate a motion blur.
LEN = 21;
THETA = 11;
PSF = fspecial('motion', LEN, THETA);
blurred = imfilter(I, PSF, 'conv', 'circular');

figure();
imshow(blurred); title('Blurred Image');
%imwrite(blurred,'blurred.jpg')

% Inverse filter
If = fft2(blurred);
Pf = fft2(PSF,hei,wid);
deblurred = ifft2(If./Pf);
figure();
imshow(deblurred); 
title('Restore Image') % 没有噪声的时候, 维纳滤波相当于逆滤波，恢复运动模糊效果是极好的

% Simulate additive noise.
% noise_mean = 0;
% noise_var = 0.0001;
% blurred_noisy = imnoise(blurred, 'gaussian', ...
%                         noise_mean, noise_var);
% %subplot(2,3,4), 
% figure();
% imshow(blurred_noisy)
% title('Simulate Blur and Noise')


%%
blurred_noisy = im2double(imread('ShaonvBlurNoise.jpg'));
imshow(blurred_noisy)
title('Simulate Blur and Noise')
%上面是matlab模糊后, 手动cpp加噪声然后复制过来的图像

% Try restoration assuming no noise.
If = fft2(blurred_noisy);
deblurred2 = ifft2(If./Pf);
%subplot(2,3,5), 
figure();
imshow(deblurred2)
title('Restoration of Blurred Assuming No Noise')%看出噪信比估计的准确性对图像影响比较大的

% Try restoration with noise is known.
noisy = blurred_noisy - blurred;
Nf = fft2(noisy);
deblurred2 = ifft2(If./Pf - Nf./Pf);
%subplot(2,3,6), 
figure();
imshow(deblurred2)
title('Restoration of Blurred with Noise Is Known')