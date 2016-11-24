function [im, out] = getBatch(samples, labels, depths, batch, pad_outputsize)
%%
% if strcmp(mode, 'training') % training
%randchannel = unidrnd(size(samples,3));
im = single(samples(:,:,:,batch)) ;
%%
%out = zeros(size(labels, 1), size(labels, 2), size(labels, 3)+ size(depths,3), size(im,4), 'single');
out(:,:,1:size(labels, 3), :) = single(labels(pad_outputsize + 1: end-pad_outputsize,pad_outputsize + 1: end-pad_outputsize,:,batch)) ;
out(:,:, size(labels, 3)+1:size(labels, 3)+ size(depths,3), :) = single(depths(pad_outputsize + 1: end-pad_outputsize,pad_outputsize + 1: end-pad_outputsize,:,batch)) ;
