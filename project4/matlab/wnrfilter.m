close all;
clear all;
clc;
%Read image
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

%Restore the blurred image
wnr1 = deconvwnr(blurred, PSF, 0);
figure();
imshow(wnr1);
title('Restored Image');

%Simulate blur and noise
noise_mean = 0;
noise_var = 0.0001;
blurred_noisy = imnoise(blurred, 'gaussian', ...
                        noise_mean, noise_var);
figure();         
imshow(blurred_noisy)
title('Simulate Blur and Noise')


%%
% blurred_noisy = im2double(imread('ShaonvBlurNoise.jpg'));
% figure();      
% imshow(blurred_noisy)
% title('Simulate Blur and Noise')
% Ip=imsubtract(blurred_noisy,blurred);
% noise_var = var(Ip(:));
%上面是matlab模糊后, 手动cpp加噪声然后复制过来的图像

%Restore the blurred and noisy image:First attempt
wnr2 = deconvwnr(blurred_noisy, PSF, 0);
figure();        
imshow(wnr2);title('Restoration of Blurred, Noisy Image Using NSR = 0')

%Restore the Blurred and Noisy Image: Second Attempt

signal_var = var(I(:));
wnr3 = deconvwnr(blurred_noisy, PSF, noise_var / signal_var);
figure();      
imshow(wnr3)
title('Restoration of Blurred, Noisy Image Using Estimated NSR');