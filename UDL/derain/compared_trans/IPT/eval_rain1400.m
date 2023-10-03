clear all;
close all;

gt_path = strcat('../derain/dataset/Rain1400/ground_truth', '/');

% PReNet = '../results/Rain1400/PReNet/';
% PReNet_r = '../results/Rain1400/PReNet_r/';
% PRN = '../results/Rain1400/PRN6/';
% PRN_r = '../results/Rain1400/PRN_r/';
file_path = strcat('../DFTL/my_model_results/DDN/', '/');

nimgs=100;nrain=14;
nmodel = length({1});


for nnn = 1:nmodel
    
    tp=0;ts=0;te=0;
    nstart = 900;
    count = 1;
    for iii=nstart+1:nstart+nimgs
        for jjj=1:nrain
            %         fprintf('img=%d,kernel=%d\n',iii,jjj);
            x_true=im2double(imread(fullfile(gt_path,sprintf('%d.jpg',iii))));%x_true
            x_true = rgb2ycbcr(x_true);x_true=x_true(:,:,1);
            

            %%
            x = (im2double(imread(fullfile(file_path,sprintf('%d_%d.png',iii,jjj)))));
            x = rgb2ycbcr(x);x = x(:,:,1);
            tp = mean(psnr(x,x_true));
            ts = ssim(x*255,x_true*255);
            
            psnrs(count,nnn)=tp;
            ssims(count,nnn)=ts;
            
            count = count + 1;
            
            %
        end
    end
    
    fprintf('psnr=%6.4f, ssim=%6.4f\n',mean(psnrs(:,nnn)),mean(ssims(:,nnn)));
    
end
