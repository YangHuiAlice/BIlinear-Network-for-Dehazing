%%%%%%%%%%%%%%%%%%%%% for 
clc;
clear;
close all;
load('hazy_image_data/nyu_depth_v2_labeled.mat')
for ii = 1: size(images,4)
    de = depths(:,:,ii);
    de = de./max(de(:));
    I = im2double(images(:,:,:,ii));
    beta = 0.5 + 1 *rand(1);   % beta (0.5, 1.5)
    tt = exp(-beta * de);
    tt = imguidedfilter(tt, I, 'NeighborhoodSize',30,'DegreeOfSmoothing',0.0001);    
    A = 0.7 + 0.3 * rand(1);       % A (0.7 , 1.0)
    hazy = I.*repmat(tt,[1 1 size(I,3)])+ A .* (1-repmat(tt,[1 1 size(I,3)]));
    %
    clearname = sprintf('hazy_image_data/NYU_filtered_A70_beta55/clear/%d.png', ii);
    hazyname = sprintf('hazy_image_data/NYU_filtered_A70_beta55/hazy/%d_hazy.png', ii);
    depthname = sprintf('hazy_image_data/NYU_filtered_A70_beta55/depths/%d_depth.mat', ii);
    imwrite(I, clearname);
    imwrite(hazy, hazyname);
    save(depthname,'tt');
end



%%%%%%%%%%%%%%% for frida %%%%%%%%%%%%%%%%%%%%
% clc;
% clear;
% close all;
% for i = 1: 18
%     % original scene without fog
% 	withoutfogfn=sprintf('LIma-%.6d.png',i);
% 	withoutfog=im2double(imread(withoutfogfn));	% now between O and 1
% 	% depthmap as a float point array
% 	depthmapfn=sprintf('Dmap-%.6d.fdd',i);
% 	depthmap=double(load(depthmapfn))/1000.0; 	% now in meters	
% 	d=1.0-depthmap./(100+depthmap);
%     tt = exp(-d);
% 	% with uniform fog 
% 	u080fn=sprintf('U080-%.6d.png',i);
% 	u080=im2double(imread(u080fn)); 		% now between O and 1
% 	% with heterogeneous fog 
% 	k080fn=sprintf('K080-%.6d.png',i);
% 	k080=im2double(imread(k080fn)); 		% now between O and 1
% 	% with cloudy fog  
% 	l080fn=sprintf('L080-%.6d.png',i);
% 	l080=im2double(imread(l080fn)); 		% now between O and 1
% 	% with cloudy heterogeneous fog  
% 	m080fn=sprintf('M080-%.6d.png',i);
% 	m080=im2double(imread(m080fn)); 		% now between O and 1 
%    
%     %%
%     clearname = sprintf('clear/%d.png', i);
%     imwrite(withoutfog, clearname);
%     
%     hazyname = sprintf('hazy/%d_hazy_1.png', i);
%     imwrite(u080, hazyname);
%     
%     hazyname = sprintf('hazy/%d_hazy_2.png', i);
%     imwrite(k080, hazyname);
%     
%     hazyname = sprintf('hazy/%d_hazy_3.png', i);
%     imwrite(l080, hazyname);
%     
%     hazyname = sprintf('hazy/%d_hazy_4.png', i);
%     imwrite(m080, hazyname);
%     
%     depthname = sprintf('depths/%d_depth.mat', i);
%     save(depthname,'tt');
% end



%%%%%%%%%%%%%%%%% For synthetic pathes %%%%%%%%%%%%%%%%%%
% dataPath = '/home/sensetime/Sensetime/MY_CODD_v3_from_computer_dehazing/hazy_image_data/outdoor_134/';
% listing  = dir(dataPath);
% for i=3:70
%     disp(['Processing image ' listing(i).name '...']);
%     im = im2double(imread([dataPath listing(i).name]));
%     patch_size = 5;
%     depth = zeros(size(im,1),size(im,2),size(im,3));
%     meters = 2000000;    
%     for r = 1 : patch_size : size(im,1)
%         for c = 1 : patch_size : size(im,2)
%             if ~(r+patch_size-1 > size(im,1) || c+patch_size-1 > size(im,2))
%                 depth(r:r+patch_size-1, c:c+patch_size-1, :) = meters.* ones(patch_size, patch_size,size(im,3));
%                 meters = meters - rand * 40;
%             end
%         end
%     end
%     depth = depth./max(depth(:));
%     tt = exp(-depth);
% %     figure(),imshow([im,tt],[]);
%     hazy = im.*tt+ 1*(1-tt);
% %     figure(),imshow([im,hazy],[]);
% 
%     clearname = sprintf('hazy_image_data/outdoor_134/clear/%d.png', i+64);
%     hazyname = sprintf('hazy_image_data/outdoor_134/hazy/%d_hazy.png', i+64);
%     depthname = sprintf('hazy_image_data/outdoor_134/depths/%d_depth.mat', i+64);
%     imwrite(im, clearname);
%     imwrite(hazy, hazyname);
%     save(depthname,'tt');
% end

%%%%%%%%%%%%%%%%% for usc synthetic images %%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% clc;
% clear;
% close all;
% dataPath = '/home/sensetime/Sensetime/MY_CODD_v3_from_computer_dehazing/hazy_image_data/Frida/';
% listing = dir([dataPath 'clear_data/']);
% for i =3:length(listing)
%     im = im2double(imread([dataPath listing(i).name]));
%     im_name = listing(3).name;
%     depth_name = [im_name(1:end-4) '_depth.mat'];
%     load([dataPath 'depths_data/' depth_name]);
%     hazy = im.*repmat(tt,[1 1 size(im,3)])+ 1*(1-repmat(tt,[1 1 size(im,3)]));
%     
%     clearname = sprintf('clear/%d.png', i-2);
%     hazyname = sprintf('hazy/%d_hazy.png', i-2);
%     
%     imwrite(I, clearname);
%     imwrite(hazy, hazyname);
% end


