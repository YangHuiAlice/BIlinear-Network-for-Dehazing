function [resA, resT] = vl_simplenn(netA, netT, x, dzdy, resA, resT, pad_outputsize, varargin)
% VL_SIMPLENN  Evaluates a simple CNN
%   RES = VL_SIMPLENN(NET, X) evaluates the convnet NET on data X.
%   RES = VL_SIMPLENN(NET, X, DZDY) evaluates the convnent NET and its
%   derivative on data X and output derivative DZDY.
%
%   The network has a simple (linear) topology, i.e. the computational
%   blocks are arranged in a sequence of layers. Please note that
%   there is no need to use this wrapper, which is provided for
%   convenience. Instead, the individual CNN computational blocks can
%   be evaluated directly, making it possible to create significantly
%   more complex topologies, and in general allowing greater
%   flexibility.
%
%   The NET structure contains two fields:
%
%   - net.layers: the CNN layers.
%   - net.normalization: information on how to normalize input data.
%
%   The network expects the data X to be already normalized. This
%   usually involves rescaling the input image(s) and subtracting a
%   mean.
%
%   RES is a structure array with one element per network layer plus
%   one representing the input. So RES(1) refers to the zeroth-layer
%   (input), RES(2) refers to the first layer, etc. Each entry has
%   fields:
%
%   - res(i+1).x: the output of layer i. Hence res(1).x is the network
%     input.
%
%   - res(i+1).aux: auxiliary output data of layer i. For example,
%     dropout uses this field to store the dropout mask.
%
%   - res(i+1).dzdx: the derivative of the network output relative to
%     variable res(i+1).x, i.e. the output of layer i. In particular
%     res(1).dzdx is the derivative of the network output with respect
%     to the network input.
%
%   - res(i+1).dzdw: the derivative of the network output relative to
%     the parameters of layer i. It can be a cell array for multiple
%     parameters.
%
%   net.layers is a cell array of network layers. The following
%   layers, encapsulating corresponding functions in the toolbox, are
%   supported:
%
%   Convolutional layer::
%     The convolutional layer wraps VL_NNCONV(). It has fields:
%
%     - layer.type = 'conv'
%     - layer.weights = {filters, biases}
%     - layer.stride: the sampling stride (usually 1).
%     - layer.pad: the padding (usually 0).
%
%   Convolution transpose layer::
%     The convolution transpose layer wraps VL_NNCONVT(). It has fields:
%
%     - layer.type = 'convt'
%     - layer.weights = {filters, biases}
%     - layer.upsample: the upsampling factor.
%     - layer.crop: the amount of output cropping.
%
%   Max pooling layer::
%     The max pooling layer wraps VL_NNPOOL(). It has fields:
%
%     - layer.type = 'pool'
%     - layer.method: pooling method ('max' or 'avg').
%     - layer.pool: the pooling size.
%     - layer.stride: the sampling stride (usually 1).
%     - layer.pad: the padding (usually 0).
%
%   Normalization layer::
%     The normalization layer wraps VL_NNNORMALIZE(). It has fields
%
%     - layer.type = 'normalize'
%     - layer.param: the normalization parameters.
%
%   Spatial normalization layer:
%     This is similar to the layer above, but wraps VL_NNSPNORM():
%
%     - layer.type = 'spnorm'
%     - layer.param: the normalization parameters.
%
%   Batch normalization layer:
%     This layer wraps VL_NNBNORM(). It has fields:
%
%     - layer.type = 'bnorm'
%     - layer.weights = {multipliers, biases}.
%
%   ReLU and Sigmoid layers::
%     The ReLU layer wraps VL_NNRELU(). It has fields:
%
%     - layer.type = 'relu'
%
%     The sigmoid layer is the same, but for the sigmoid function, with
%     `relu` replaced by `sigmoid`.
%
%   Dropout layer::
%     The dropout layer wraps VL_NNDROPOUT(). It has fields:
%
%     - layer.type = 'dropout'
%     - layer.rate: the dropout rate.
%
%   Softmax layer::
%     The softmax layer wraps VL_NNSOFTMAX(). It has fields
%
%     - layer.type = 'softmax'
%
%   Log-loss layer::
%     The log-loss layer wraps VL_NNLOSS(). It has fields:
%
%     - layer.type = 'loss'
%     - layer.class: the ground-truth class.
%
%   Softmax-log-loss layer::
%     The softmax-log-loss layer wraps VL_NNSOFTMAXLOSS(). It has
%     fields:
%
%     - layer.type = 'softmaxloss'
%     - layer.class: the ground-truth class.
%
%   P-dist layer:
%     The pdist layer wraps VL_NNPDIST(). It has fields:
%
%     - layer.type = 'pdist'
%     - layer.p = P parameter of the P-distance
%     - layer.noRoot = whether to raise the distance to the P-th power
%     - layer.epsilon = regularization parameter for the derivatives
%
%   Custom layer::
%     This can be used to specify custom layers.
%
%     - layer.type = 'custom'
%     - layer.forward: a function handle computing the block.
%     - layer.backward: a function handle computing the block derivative.
%
%     The first function is called as res(i+1) = forward(layer, res(i), res(i+1))
%     where res() is the struct array specified before. The second function is
%     called as res(i) = backward(layer, res(i), res(i+1)). Note that the
%     `layer` structure can contain additional fields if needed.

% Copyright (C) 2014 Andrea Vedaldi.
% All rights reserved.
%
% This file is part of the VLFeat library and is made available under
% the terms of the BSD license (see the COPYING file).

%%
original_input = x(pad_outputsize + 1: end-pad_outputsize,pad_outputsize + 1: end-pad_outputsize,:,:);
%%
opts.res = [] ;
opts.conserveMemory = false ;
opts.sync = false ;
opts.disableDropout = false ;
opts.freezeDropout = false ;
opts.accumulate = false ;
opts.cudnn = true ;
opts.backPropDepth = +inf ;

opts = vl_argparse(opts, varargin);

n = numel(netA.layers) ; % netA and netT have the same layers

if (nargin <= 2) || isempty(dzdy)
  doder = false ;
else
  doder = true ;
end

if opts.cudnn
  cudnn = {'CuDNN'} ;
else
  cudnn = {'NoCuDNN'} ;
end

gpuMode = isa(x, 'gpuArray') ;

if nargin <= 3 || isempty(resA)
  resA = struct(...
    'x', cell(1,n+1), ...
    'dzdx', cell(1,n+1), ...
    'dzdw', cell(1,n+1), ...
    'aux', cell(1,n+1), ...
    'time', num2cell(zeros(1,n+1)), ...
    'backwardTime', num2cell(zeros(1,n+1))) ;
%%
  resT = struct(...
    'x', cell(1,n+1), ...
    'dzdx', cell(1,n+1), ...
    'dzdw', cell(1,n+1), ...
    'aux', cell(1,n+1), ...
    'time', num2cell(zeros(1,n+1)), ...
    'backwardTime', num2cell(zeros(1,n+1))) ;
end
resA(1).x = x;
resT(1).x = x;

for i=1:n
  lA = netA.layers{i} ;
  lT = netT.layers{i} ;
  resA(i).time = tic ;
  switch lA.type
    case 'conv'
      if isfield(lA, 'weights')
        resA(i+1).x = vl_nnconv(resA(i).x, lA.weights{1}, lA.weights{2}, ...
                               'pad', lA.pad, 'stride', lA.stride, ...
                               cudnn{:}) ;  
        %%
        resT(i+1).x = vl_nnconv(resT(i).x, lT.weights{1}, lT.weights{2}, ...
                               'pad', lT.pad, 'stride', lT.stride, ...
                               cudnn{:}) ; 
      else
        resA(i+1).x = vl_nnconv(resA(i).x, lA.filters, lA.biases, ...
                               'pad', lA.pad, 'stride', lA.stride, ...
                               cudnn{:}) ;
        %%
        resT(i+1).x = vl_nnconv(resT(i).x, lT.filters, lT.biases, ...
                               'pad', lT.pad, 'stride', lT.stride, ...
                               cudnn{:}) ;
      end
    case 'convt'
      if isfield(lA, 'weights')
        resA(i+1).x = vl_nnconvt(resA(i).x, lA.weights{1}, lA.weights{2}, ...
                               'crop', lA.crop, 'upsample', lA.upsample, ...
                               cudnn{:}) ;
        %%
        resT(i+1).x = vl_nnconvt(resT(i).x, lT.weights{1}, lT.weights{2}, ...
                               'crop', lT.crop, 'upsample', lT.upsample, ...
                               cudnn{:}) ;
      else
        resA(i+1).x = vl_nnconv(resA(i).x, lA.filters, lA.biases, ...
                               'crop', lA.pad, 'upsample', lA.upsample, ...
                               cudnn{:}) ;
        %%
        resT(i+1).x = vl_nnconv(resT(i).x, lT.filters, lT.biases, ...
                               'crop', lT.pad, 'upsample', lT.upsample, ...
                               cudnn{:}) ;
      end
    case 'pool'
      resA(i+1).x = vl_nnpool(resA(i).x, lA.pool, ...
                             'pad', lA.pad, 'stride', lA.stride, ...
                             'method', lA.method, ...
                             cudnn{:}) ;
      %%
      resT(i+1).x = vl_nnpool(resT(i).x, lT.pool, ...
                             'pad', lT.pad, 'stride', lT.stride, ...
                             'method', lT.method, ...
                             cudnn{:}) ;
    case 'normalize'
      resA(i+1).x = vl_nnnormalize(resA(i).x, lA.param) ;
      %%
      resT(i+1).x = vl_nnnormalize(resT(i).x, lT.param) ;
      %%
    case 'hazesquareloss' %% combine layer
      [~,~,resA(i+1).x] = vl_nnhazesquareloss(resA(i).x, resT(i).x, lA.class, original_input) ;
      resT(i+1).x = resA(i+1).x;
      %%
    case 'hazesquareRobustloss' %% combine layer
      [~,~,resA(i+1).x] = vl_nnhazerobustloss(resA(i).x, resT(i).x, lA.class, original_input) ;
      resT(i+1).x = resA(i+1).x;
      %%
    case 'hazesquareGradientloss' %% combine layer
      [~,~,resA(i+1).x] = vl_nnhazesquareloss_gradient_v2(resA(i).x, resT(i).x, lA.class, original_input) ;
      resT(i+1).x = resA(i+1).x;
      %%
    case 'relu'
      resA(i+1).x = vl_nnrelu(resA(i).x) ;
      %%
      resT(i+1).x = vl_nnrelu(resT(i).x) ;
    case 'sigmoid'
      resA(i+1).x = vl_nnsigmoid(resA(i).x) ;
      %%
      resT(i+1).x = vl_nnsigmoid(resT(i).x) ;
    otherwise
      error('Unknown layer type %s', lA.type) ;
  end
  % optionally forget intermediate results
  forget = opts.conserveMemory ;
  forget = forget & (~doder || strcmp(lA.type, 'relu')) ;
  forget = forget & ~(strcmp(lA.type, 'loss') || strcmp(lA.type, 'softmaxloss')) ;
  forget = forget & (~isfield(lA, 'rememberOutput') || ~lA.rememberOutput) ;
  if forget
    resA(i).x = [] ;
  end
  if gpuMode & opts.sync
    % This should make things slower, but on MATLAB 2014a it is necessary
    % for any decent performance.
    wait(gpuDevice) ;
  end
  resA(i).time = toc(resA(i).time) ;
  resT(i).time = resA(i).time;
end



%%
if doder
  resA(n+1).dzdx = dzdy ;
  resT(n+1).dzdx = dzdy ;
  for i=n:-1:max(1, n-opts.backPropDepth+1)
    lA = netA.layers{i} ;
    lT = netT.layers{i} ;
    resA(i).backwardTime = tic ;
    switch lA.type
      case 'conv'
        if ~opts.accumulate
          if isfield(lA, 'weights')
            [resA(i).dzdx, resA(i).dzdw{1}, resA(i).dzdw{2}] = ...
                vl_nnconv(resA(i).x, lA.weights{1}, lA.weights{2}, ...
                          resA(i+1).dzdx, ...
                          'pad', lA.pad, 'stride', lA.stride, ...
                          cudnn{:}) ;
             %%
            [resT(i).dzdx, resT(i).dzdw{1}, resT(i).dzdw{2}] = ...
                vl_nnconv(resT(i).x, lT.weights{1}, lT.weights{2}, ...
                          resT(i+1).dzdx, ...
                          'pad', lT.pad, 'stride', lT.stride, ...
                          cudnn{:}) ;

          else
            % Legacy code: will go
            [resA(i).dzdx, resA(i).dzdw{1}, resA(i).dzdw{2}] = ...
                vl_nnconv(resA(i).x, lA.filters, lA.biases, ...
                          resA(i+1).dzdx, ...
                          'pad', lA.pad, 'stride', lA.stride, ...
                          cudnn{:}) ;
             %%
             [resT(i).dzdx, resT(i).dzdw{1}, resT(i).dzdw{2}] = ...
                vl_nnconv(resT(i).x, lT.filters, lT.biases, ...
                          resT(i+1).dzdx, ...
                          'pad', lT.pad, 'stride', lT.stride, ...
                          cudnn{:}) ;
          end
        else
          dzdwA = cell(1,2) ;
          dzdwT = dzdwA ;
          if isfield(lA, 'weights')
            [resA(i).dzdx, dzdwA{1}, dzdwA{2}] = ...
                vl_nnconv(resA(i).x, lA.weights{1}, lA.weights{2}, ...
                          resA(i+1).dzdx, ...
                          'pad', lA.pad, 'stride', lA.stride, ...
                          cudnn{:}) ;
            %%
            [resT(i).dzdx, dzdwT{1}, dzdwT{2}] = ...
                vl_nnconv(resT(i).x, lT.weights{1}, lT.weights{2}, ...
                          resT(i+1).dzdx, ...
                          'pad', lT.pad, 'stride', lT.stride, ...
                          cudnn{:}) ;
          else
            % Legacy code: will go
            [resA(i).dzdx, dzdwA{1}, dzdwA{2}] = ...
                vl_nnconv(resA(i).x, lA.filters, lA.biases, ...
                          resA(i+1).dzdx, ...
                          'pad', lA.pad, 'stride', lA.stride, ...
                          cudnn{:}) ;
            %%
            [resT(i).dzdx, dzdwT{1}, dzdwT{2}] = ...
                vl_nnconv(resT(i).x, lT.filters, lT.biases, ...
                          resT(i+1).dzdx, ...
                          'pad', lT.pad, 'stride', lT.stride, ...
                          cudnn{:}) ;
          end
          for j=1:2
            resA(i).dzdw{j} = resA(i).dzdw{j} + dzdwA{j} ;
            %%
            resT(i).dzdw{j} = resT(i).dzdw{j} + dzdwT{j} ;
          end
          clear dzdwA  dzdwT ;
        end

      case 'convt'
        if ~opts.accumulate
          if isfield(lA, 'weights')
            [resA(i).dzdx, resA(i).dzdw{1}, resA(i).dzdw{2}] = ...
                vl_nnconvt(resA(i).x, lA.weights{1}, lA.weights{2}, ...
                          resA(i+1).dzdx, ...
                          'crop', lA.crop, 'upsample', lA.upsample, ...
                          cudnn{:}) ;
            %%
            [resT(i).dzdx, resT(i).dzdw{1}, resT(i).dzdw{2}] = ...
                vl_nnconvt(resT(i).x, lT.weights{1}, lT.weights{2}, ...
                          resT(i+1).dzdx, ...
                          'crop', lT.crop, 'upsample', lT.upsample, ...
                          cudnn{:}) ;
          else
            % Legacy code: will go
            [resA(i).dzdx, resA(i).dzdw{1}, resA(i).dzdw{2}] = ...
                vl_nnconvt(resA(i).x, lA.filters, lA.biases, ...
                         resA(i+1).dzdx, ...
                          'crop', lA.crop, 'upsample', lA.upsample, ...
                          cudnn{:}) ;
            %%
            [resT(i).dzdx, resT(i).dzdw{1}, resT(i).dzdw{2}] = ...
                vl_nnconvt(resT(i).x, lT.filters, lT.biases, ...
                         resT(i+1).dzdx, ...
                          'crop', lT.crop, 'upsample', lT.upsample, ...
                          cudnn{:}) ;
          end
        else
          dzdwA = cell(1,2) ;
          dzdwT = dzdwA ;
          if isfield(lA, 'weights')
            [resA(i).dzdx, dzdwA{1}, dzdwA{2}] = ...
                vl_nnconvt(resA(i).x, lA.weights{1}, lA.weights{2}, ...
                          resA(i+1).dzdx, ...
                          'crop', lA.crop, 'upsample', lA.upsample, ...
                          cudnn{:}) ;
             %%
             [resT(i).dzdx, dzdwT{1}, dzdwT{2}] = ...
                vl_nnconvt(resT(i).x, lT.weights{1}, lT.weights{2}, ...
                          resT(i+1).dzdx, ...
                          'crop', lT.crop, 'upsample', lT.upsample, ...
                          cudnn{:}) ;
          else
            % Legacy code: will go
            [resA(i).dzdx, dzdwA{1}, dzdwA{2}] = ...
                vl_nnconvt(resA(i).x, lA.filters, lA.biases, ...
                          resA(i+1).dzdx, ...
                          'crop', lA.crop, 'upsample', lA.upsample, ...
                          cudnn{:}) ;
            %%
            [resT(i).dzdx, dzdwT{1}, dzdwT{2}] = ...
                vl_nnconvt(resT(i).x, lT.filters, lT.biases, ...
                          resT(i+1).dzdx, ...
                          'crop', lT.crop, 'upsample', lT.upsample, ...
                          cudnn{:}) ;
          end
          for j=1:2
            resA(i).dzdw{j} = resA(i).dzdw{j} + dzdwA{j} ;
            %%
            resT(i).dzdw{j} = resT(i).dzdw{j} + dzdwT{j} ;
          end
          clear dzdwA dzdwT ;
        end
       
      case 'pool'
        resA(i).dzdx = vl_nnpool(resA(i).x, lA.pool, resA(i+1).dzdx, ...
                                'pad', lA.pad, 'stride', lA.stride, ...
                                'method', lA.method, ...
                                cudnn{:}) ;
        %%
        resT(i).dzdx = vl_nnpool(resT(i).x, lT.pool, resT(i+1).dzdx, ...
                                'pad', lT.pad, 'stride', lT.stride, ...
                                'method', lT.method, ...
                                cudnn{:}) ;
      case 'normalize'
        resA(i).dzdx = vl_nnnormalize(resA(i).x, lA.param, resA(i+1).dzdx) ;
        %%
        resT(i).dzdx = vl_nnnormalize(resT(i).x, lT.param, resT(i+1).dzdx) ;
      case 'hazesquareloss'
        [resA(i).dzdx, resT(i).dzdx, ~] = vl_nnhazesquareloss(resA(i).x, resT(i).x, lA.class, original_input, resA(i+1).dzdx) ;
        %%
      case 'hazesquareRobustloss'
        [resA(i).dzdx, resT(i).dzdx, ~] = vl_nnhazerobustloss(resA(i).x, resT(i).x, lA.class, original_input, resA(i+1).dzdx) ;
        %%  
      case 'hazesquareGradientloss'
        [resA(i).dzdx, resT(i).dzdx, ~] = vl_nnhazesquareloss_gradient_v2(resA(i).x, resT(i).x, lA.class, original_input, resA(i+1).dzdx) ;
        %%  
      case 'relu'
        if ~isempty(resA(i).x)
          resA(i).dzdx = vl_nnrelu(resA(i).x, resA(i+1).dzdx) ;
          %%
          resT(i).dzdx = vl_nnrelu(resT(i).x, resT(i+1).dzdx) ;
        else
          % if res(i).x is empty, it has been optimized away, so we use this
          % hack (which works only for ReLU):
          resA(i).dzdx = vl_nnrelu(resA(i+1).x, resA(i+1).dzdx) ;
          %%
          resT(i).dzdx = vl_nnrelu(resT(i+1).x, resT(i+1).dzdx) ;
        end
      case 'sigmoid'
        resA(i).dzdx = vl_nnsigmoid(resA(i).x, resA(i+1).dzdx) ;
        resT(i).dzdx = vl_nnsigmoid(resT(i).x, resT(i+1).dzdx) ;
    end
    if opts.conserveMemory
      resA(i+1).dzdx = [] ;
      resT(i+1).dzdx = [] ;
    end
    if gpuMode & opts.sync
      wait(gpuDevice) ;
    end
    resA(i).backwardTime = toc(resA(i).backwardTime) ;
    resT(i).backwardTime = resA(i).backwardTime;
  end
end
