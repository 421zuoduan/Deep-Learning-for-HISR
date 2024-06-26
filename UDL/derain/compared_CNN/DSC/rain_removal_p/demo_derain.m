clear all 
disp('====================================================================');
disp('This demo shows how to get a de-rained image from an input rainy image ')
disp('====================================================================');
%% 
addpath('./ksvdbox13/');
addpath('./ompbox10/');

imname = 'rain-077.png';
%% input image
I_rainy = imread(['./Rainy_image/',imname]);
I_rainy = im2double(I_rainy);

%% parameter setting for dictionary initialization
param_dic.p_size = 16;   % the patch size
param_dic.K2 = 128;      % the number of rain atoms 
param_dic.K1 = 512;      % the number of the non-rain atoms.
param_dic.step_size = 3; % the step size for generating the training samples.
param_dic.p_num = 10000; % the number of the training samples.
param_dic.iter_num = 5;  % the iteration number for training the initial dictionary

%% parameter setting for the algorithm
param_alg.step_s = 2;   % the step size for generating rainy patches
param_alg.T1 = 5;       % the sparsity degree of each column of image patches
param_alg.T2 = 8;       % the sparsity degree of each column of rain patches
param_alg.iter_num = 5; % the iteration number of the main loops

%% de-raining
im_derain = DSC_derain(I_rainy,param_dic,param_alg);
figure;imshow(I_rainy);title('Rainy image');
figure;imshow(im_derain);title('De-rained image');
imwrite(im_derain, imname);
