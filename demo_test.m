clc;
clear;
close all;
run matconvnet/matlab/vl_setupnn ;
% addpath('reconstruction');
%% read ground truth image
tic
imagename = '5.png';
I = im2double(imread(imagename));
%I = I(10:end-9,10:end-9,:);
% if size(I,3)<3
%     I = repmat(I,[1,1,3]);
% end
% I_5 = convert_image2coarse(I,'/home/sensetime/Sensetime/MY_CODD_v3_from_computer_dehazing/coarseModel/net-epoch-84.mat');

%% set parameters
% up_scale = 3;
%
model = 'models/non-noise-dehazing.mat';
load(model)
%
pad_outputsize = 0;
for lay = 1: length(netA.layers)
    if strcmp(netA.layers{lay}.type, 'conv') ;
        pad_outputsize = pad_outputsize + (size(netA.layers{lay}.weights{1},1)-1)/2;
    end
    if strcmp(netA.layers{lay}.type, 'pool') 
        pad_outputsize = pad_outputsize + (netA.layers{lay}.pool(1)-1)/2;
    end
end

tmp = zeros(size(I,1), size(I,2),size(I,3)+1, 'single');
tmp  = tmp(pad_outputsize + 1: end-pad_outputsize,pad_outputsize + 1: end-pad_outputsize,:,:);
netA.layers{end}.class = tmp;
netT.layers{end}.class = tmp;

%
I_tmp = I(pad_outputsize + 1: end-pad_outputsize,pad_outputsize + 1: end-pad_outputsize,:,:);
J = [];
tmin = 1e-5;
%

%%%%%%% without gamma transform %%%%%%%%%%%%%
[resA, resT] = vl_simplenn(netA, netT, single(I), [], [], [], pad_outputsize,...
    'accumulate', false, ...
    'disableDropout', false, ...
    'conserveMemory', false, ...
    'backPropDepth', +inf, ...
    'sync', false, ...
    'cudnn', true) ;

%
A = resA(end-1).x;
t = resT(end-1).x;
t = max(t,0.1);
% imwrite(t, [imagename(1:end-4) '_t.png']);

% r = 5;
% eps = 10^-10;
% filtered_t = guidedfilter(rgb2gray(I_tmp), t, r, eps);
% filtered_t = repmat(filtered_t, [1,1, size(I,3)]);
filtered_t = imguidedfilter(t,I_tmp,'NeighborhoodSize',50,'DegreeOfSmoothing',0.01);
filtered_t = repmat(filtered_t, [1,1, size(I,3)]);
% imwrite(filtered_t, [imagename(1:end-4) '_t_filtered.png']);

filtered_A = imguidedfilter(A,I_tmp,'NeighborhoodSize',7,'DegreeOfSmoothing',0.0001);
%filtered_A = repmat(filtered_A, [1,1, size(I,3)]);

t = repmat(t, [1,1, size(I,3)]);
% A = repmat(A, [1,1, size(I,3)]);

% t_coarse = repmat(t_coarse, [1,1, size(im_original_hazy,3)]);
% A_coarse = repmat(A_coarse, [1,1, size(im_original_hazy,3)]);

J = (I_tmp - A)./t + A;
J_guided_T = (I_tmp - A)./filtered_t + A;
J_guided_T_A = (I_tmp - filtered_A)./filtered_t + filtered_A;
%J_gamma = J_guided_T.^(1/1.3);


toc
% figure; imshow([t_guided,t],[]);
% figure; title('guided_J'), imshow([I_tmp,J_guided_T],[]);
% figure; title('J'), imshow([I_tmp,J],[]);

%figure; imshow([I_tmp,A,t, J],[]);
%imagename_wri2 = [imagename(1:end-4) '_AN.png'];
%imwrite(A, imagename_wri2);
%imagename_wri2 = [imagename(1:end-4) '_t_Test.png'];
%imwrite(filtered_t, imagename_wri2);
%imagename_wri2 = [imagename(1:end-4) '_Test.png'];
%imwrite(J_guided_T, imagename_wri2);

figure; imshow([I_tmp,A,filtered_t, J_guided_T],[]);
figure; imshow([I_tmp,filtered_A,filtered_t, J_guided_T_A],[]);
% imagename_wri2 = [imagename(1:end-4) '_v1.png'];
% imwrite(J_guided_T, imagename_wri2);
% 
%figure; imshow([I_tmp, J_guided_T, J_gamma],[]);
%imwrite(J_gamma,[imagename(1:end-4) '_our_gamma.png']);
%imwrite(filtered_t,[imagename(1:end-4) '_t_our.png']);
% imagename_wri2 = [imagename(1:end-4) '_v1_gamma.png'];
% imwrite(J_gamma, imagename_wri2);
%%
% final_re = [];
% for ii = 1:size(J_guided_T,3);
%     final_re(:,:,ii) = imadjust(J_guided_T(:,:,ii));
% end
% figure; imshow(final_re,[]);
% J_gamma = imadjust(J,[0 1.0],[0 1.0], 0.6);
% figure; imshow([J,J_gamma],[]);

% figure; imshow([I_tmp,A,t, J],[]);
% imagename_wri = [imagename(1:end-4) '_out_guidedFilter_T.png'];
% imwrite(t_guided, imagename_wri);
% imagename_wri = [imagename(1:end-4) '_out_T.png'];
% imwrite(t, imagename_wri);

% imagename_wri2 = [imagename(1:end-4) '_9911_xx1_filtered_adjusted.png'];
% imwrite(final_re, imagename_wri2);


% imagename_wri_A = [imagename(1:end-4) '_A.png'];
% imwrite(A/max(A(:)), imagename_wri_A);
% imagename_wri_T = [imagename(1:end-4) '_T.png'];
% imwrite(t/max(t(:)), imagename_wri_T);



%%%%%%%%%%%% with gammma transform %%%%%%%%%%%%%
% I_gamma = I.^(2.2);
% [resA, resT] = vl_simplenn(netA, netT, single(I_gamma), [], [], [], pad_outputsize,...
%     'accumulate', false, ...
%     'disableDropout', false, ...
%     'conserveMemory', false, ...
%     'backPropDepth', +inf, ...
%     'sync', false, ...
%     'cudnn', true) ;
% 
% %
% A = resA(end-1).x;
% t = resT(end-1).x;
% 
% I_tmp_gamma = I_tmp.^(2.2);
% r = 40;
% eps = 10^-6;
% filtered_t = guidedfilter(rgb2gray(I_tmp_gamma), t, r, eps);
% filtered_t = repmat(filtered_t, [1,1, size(I,3)]);
% A = repmat(A,[1,1,3]);
% 
% % t_guided = imguidedfilter(t,I_tmp,'NeighborhoodSize',10,'DegreeOfSmoothing',0.0001);
% 
% J_gamma_2 = (I_tmp_gamma - A)./filtered_t + A;
% J_gamma_2 = J_gamma_2.^(1/2.2);
% 
% toc;






% figure; imshow([I_tmp,A, filtered_t, J_gamma_2],[]);
% imagename_wri2 = [imagename(1:end-4) '_9911_xx1_filtered_gamma_2.png'];
% imwrite(J_gamma_2, imagename_wri2);
% 
% figure;imshow([I_tmp,J,J_guided_T,J_gamma,J_gamma_2],[]);
% 
% imagename_wri2 = [imagename(1:end-4) '_9911_xx1_filtered_gamma_T.png'];
% imwrite(filtered_t, imagename_wri2);


%% set parameters
% up_scale = 3;
% %
% model = '/home/sensetime/Sensetime/MY_CODD_v3_from_computer_dehazing/data_LAST_x1_19_relu/exp/net-epoch-51.mat';
% load(model)
% %
% pad_outputsize = 0;
% for lay = 1: length(netA.layers)
%     if strcmp(netA.layers{lay}.type, 'conv') ;
%         pad_outputsize = pad_outputsize + (size(netA.layers{lay}.weights{1},1)-1)/2;
% %         netA.layers{lay}.weights{1}= gather(netA.layers{lay}.weights{1});
% %         netA.layers{lay}.weights{2}= gather(netA.layers{lay}.weights{2});
% %         netT.layers{lay}.weights{1}= gather(netT.layers{lay}.weights{1});
% %         netT.layers{lay}.weights{2}= gather(netT.layers{lay}.weights{2});
%     end
%     if strcmp(netA.layers{lay}.type, 'pool') 
%         pad_outputsize = pad_outputsize + (netA.layers{lay}.pool(1)-1)/2;
%     end
% end
% 
%  
% tmp = zeros(size(I,1), size(I,2),size(I,3)+1, 'single');
% tmp  = tmp(pad_outputsize + 1: end-pad_outputsize,pad_outputsize + 1: end-pad_outputsize,:,:);
% netA.layers{end}.class = tmp;
% netT.layers{end}.class = tmp;
% 
% %
% I_tmp_old = I(pad_outputsize + 1: end-pad_outputsize,pad_outputsize + 1: end-pad_outputsize,:,:);
% J_old = [];
% tmin = 1e-5;
% %
% tic
% 
% [resA, resT] = vl_simplenn(netA, netT, single(I), [], [], [], pad_outputsize,...
%     'accumulate', false, ...
%     'disableDropout', false, ...
%     'conserveMemory', false, ...
%     'backPropDepth', +inf, ...
%     'sync', false, ...
%     'cudnn', true) ;
% 
% %
% A = resA(end-1).x;
% t = resT(end-1).x;
% t = repmat(t, [1,1, size(I,3)]);
% A = repmat(A, [1,1, size(I,3)]);
% 
% J_old = (I_tmp_old - A)./t + A;
% t_guided = imguidedfilter(t,I_tmp_old,'NeighborhoodSize',15,'DegreeOfSmoothing',0.00001);
% J = (I_tmp_old - A)./t + A;
% J_guided = (I_tmp_old - A)./t_guided + A;
% toc
% % figure; title('J_2'), imshow([I_tmp_old,J_guided],[]);
% % figure; title('J_2'), imshow([I_tmp_old,J_old],[]);
% 
% % 
% % figure; title('J_2_____'), imshow([I_tmp_old,A,t, J_old],[]);
% % imagename_wri = [imagename(1:end-4) '_relu.png'];
% % imwrite(J_old, imagename_wri);
% 
% 
% 
% %%
% final_re_old = [];
% for ii = 1:size(J_old,3);
%     final_re_old(:,:,ii) = imadjust(J_old(:,:,ii));
% end
% for ii = 1:size(J_old,3);
%     final_re_old_(:,:,ii) = J_old(:,:,ii)*1.1;
% end
% % figure; imshow([final_re_old,final_re_old_],[]);
% % figure;
% % subplot(1,2,1); imshow(J);
% % subplot(1,2,2); imshow(J_old);
% % J_gamma = imadjust(J_old,[0 1.0],[0 1.0], 0.6);
% % figure; imshow([J_old,J_gamma],[]);
% 
% 
% 
% % 
% % I_tmp = im2double(imread('/home/sensetime/Sensetime/MY_CODD_v3_from_computer_dehazing/hazy_image_data/NYU2/hazy/14_hazy.png'));
% % tt = repmat(tt,[1,1,3]);
% % t_guided = imguidedfilter(tt,I_tmp,'NeighborhoodSize',7,'DegreeOfSmoothing',0.00001);
% % imshow([tt,t_guided],[]);
