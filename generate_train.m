clear;close all;
%% settings
folder = 'NYU_filtered_A70_beta55/clear/';
folderhazy = 'NYU_filtered_A70_beta55/dehaze/';
folderdepth = 'NYU_filtered_A70_beta55/dehaze_t/';
size_input = 32;
size_label = 32;
stride = 28;


% model = 'net-epoch-31.mat';
% load(model)
% pad_outputsize = 0;
% for lay = 1: length(netA.layers)
%     if strcmp(netA.layers{lay}.type, 'conv') ;
%         pad_outputsize = pad_outputsize + (size(netA.layers{lay}.weights{1},1)-1)/2;
%     end
%     if strcmp(netA.layers{lay}.type, 'pool') 
%         pad_outputsize = pad_outputsize + (netA.layers{lay}.pool(1)-1)/2;
%     end
% end

%% initialization
data = zeros(size_input, size_input, 3, 1);
label = zeros(size_label, size_label, 3, 1);
depth = zeros(size_input, size_input, 1, 1);
padding = abs(size_input - size_label)/2;
count = 0;

boader = 9;

%% generate data
filepaths = dir(fullfile(folder,'*.png'));

aa = randperm(length(filepaths));
 
for i =1:length(filepaths)
    if (mod(i,10) == 0)
            fprintf('Extracting image: %d / %d\n', i,length(filepaths));
    end
    image = imread(fullfile([folder num2str(i) '.png']));
    image = im2double(image);
%     image = image(boader + 1: end-boader,boader + 1: end-boader,:);
%     image = image(pad_outputsize + 1: end-pad_outputsize,pad_outputsize + 1: end-pad_outputsize,:,:);
    %%
    hazyname = [num2str(i) '.png'];
    labels_image = fullfile(folderhazy,hazyname);
    im_label = imread(labels_image);
    im_label = im2double(im_label);
%     im_label = im_label(boader + 1: end-boader,boader + 1: end-boader,:);
    %%
    depthname = [num2str(i) '.png'];
    depth_image = fullfile(folderdepth,depthname);
    tt = imread(depth_image);
    tt = im2double(tt);
    tt = tt(:,:,1);
%     tt = tt(boader + 1: end-boader,boader + 1: end-boader,:);
    %%
    [hei,wid, chs] = size(im_label);

    for x = 1 : stride : hei-size_input+1
        for y = 1 :stride : wid-size_input+1
            
            subim_input = im_label(x : x+size_input-1, y : y+size_input-1, :);
            subim_label = image(x : x+size_input-1, y : y+size_input-1, :);
            subim_depth = tt(x : x+size_input-1, y : y+size_input-1, :);
            
            %%
            [g1x, g1y] = gradient(subim_input);
            [g2x, g2y] = gradient(subim_label);
            g1norm = abs(g1x).^2 + abs(g1y).^2;
            g2norm = abs(g2x).^2 + abs(g2y).^2;
            if sum(g1norm(:))<1e-1&&sum(g2norm(:))<1e-1
                continue;
            end
            %%
            count=count+1;
            data(:, :, :, count) = subim_input;
            label(:, :, :, count) = subim_label;
            depth(:, :, :, count) = subim_depth;
        end
    end
end

order = randperm(count);
data = data(:, :, :, order);
label = label(:, :, :, order); 
depth = depth(:, :, :, order); 

train_num = round(count*0.9) + 1;

samples = data(:,:,:,1:train_num);
labels = label(:, :, :, 1:train_num);
depths = depth(:, :, :, 1:train_num);

%%
test_samples = single(data(:,:,:,train_num+1:end));
test_labels = single(label(:,:,:,train_num+1:end));
test_depths = single(depth(:,:,:,train_num+1:end));

save('mat/train/patches_1','-v7.3','samples','labels','depths');
save('mat/val/val_1','-v7.3','test_samples','test_labels','test_depths');
%%


