
clc;
clear;
close all;
run matconvnet/matlab/vl_setupnn ;
%
opts.batchSize = 64;
opts.numEpochs = 500;
opts.gpus = [1];
opts.continue = true ;

netA = initialize_dehazing_CNN_A( ) ;
netT = initialize_dehazing_CNN_T( ) ;

training_path = 'mat/train/';
validation_path = 'mat/val/';
modelName = 'net-epoch-437.mat';
pad_outputsize = 0;
for i = 1: length(netA.layers)
    if strcmp(netA.layers{i}.type, 'conv')
        pad_outputsize = pad_outputsize + (size(netA.layers{i}.weights{1},1)-1)/2;
    end
    if strcmp(netA.layers{i}.type, 'pool') 
        pad_outputsize = pad_outputsize + (netA.layers{i}.pool(1)-1)/2;
    end
end
opts.pad_outputsize = pad_outputsize;
% [netA, netT, info] = cnn_train_dehazing_fine(netA, netT, @getBatch_fine, training_path, validation_path, modelName, opts) ;
[netA, netT, info] = cnn_train_dehazing(netA, netT, @getBatch, training_path, validation_path, opts) ;