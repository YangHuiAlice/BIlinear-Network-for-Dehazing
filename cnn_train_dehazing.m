function [netA, netT, info] = cnn_train_dehazing(netA, netT, getBatch, training_path, validation_path, varargin)
% CNN_TRAIN   Demonstrates training a CNN
%    CNN_TRAIN() is an example learner implementing stochastic
%    gradient descent with momentum to train a CNN. It can be used
%    with different datasets and tasks by providing a suitable
%    getBatch function.
%
%    The function automatically restarts after each training epoch by
%    checkpointing.
%
%    The function supports training on CPU or on one or more GPUs
%    (specify the list of GPU IDs in the `gpus` option). Multi-GPU
%    support is relatively primitive but sufficient to obtain a
%    noticable speedup.

% Copyright (C) 2014-15 Andrea Vedaldi.
% All rights reserved.
%
% This file is part of the VLFeat library and is made available under
% the terms of the BSD license (see the COPYING file).

opts.batchSize = 112 ;
opts.numSubBatches = 1 ;
opts.train = [] ;
opts.val = [] ;
opts.numEpochs = 500 ;
opts.gpus = [] ; % which GPU devices to use (none, one, or more)
opts.learningRate = 0.01 ; % 0.001
opts.continue = false ;
opts.expDir = fullfile('data_Exp4','exp') ;
opts.conserveMemory = false ;
opts.backPropDepth = +inf ;
opts.sync = false ;
opts.prefetch = false ;
opts.cudnn = true ;
opts.weightDecay = 0.05 ; % 0.0005
opts.momentum = 0.9 ; % 0.9
opts.errorFunction = 'hazesquareloss' ;
opts.errorLabels = {} ;
opts.plotDiagnostics = false ;
opts.memoryMapFile = fullfile(tempdir, 'matconvnet.bin') ;
%
opts.pad_outputsize = 0;
opts = vl_argparse(opts, varargin) ;

if ~exist(opts.expDir, 'dir'), mkdir(opts.expDir) ; end
% if isempty(opts.train), opts.train = find(imdb.images.set==1) ; end
% if isempty(opts.val), opts.val = find(imdb.images.set==2) ; end
if isnan(opts.train), opts.train = [] ; end

%% Check the number patches for training and validation
train_mat = strcat(training_path, 'patches_', num2str(1), '.mat');
load(train_mat);
opts.train = 1:size(samples,4);
val_mat = strcat(validation_path, 'val_', num2str(1), '.mat');
load(val_mat);
opts.val = 1:size(test_samples,4);
% -------------------------------------------------------------------------
%                                                    Network initialization
% -------------------------------------------------------------------------

evaluateMode = isempty(opts.train) ;
if ~evaluateMode
    %% ==================for net A==================== %%
    for i=1:numel(netA.layers)
        if isfield(netA.layers{i}, 'weights')
            J = numel(netA.layers{i}.weights) ;
            for j=1:J
                netA.layers{i}.momentum{j} = zeros(size(netA.layers{i}.weights{j}), 'single') ;
            end
            if ~isfield(netA.layers{i}, 'learningRate')
                netA.layers{i}.learningRate = ones(1, J, 'single') ;
            end
            if ~isfield(netA.layers{i}, 'weightDecay')
                netA.layers{i}.weightDecay = ones(1, J, 'single') ;
            end
        end
        % Legacy code: will be removed
        if isfield(netA.layers{i}, 'filters')
            netA.layers{i}.momentum{1} = zeros(size(netA.layers{i}.filters), 'single') ;
            netA.layers{i}.momentum{2} = zeros(size(netA.layers{i}.biases), 'single') ;
            if ~isfield(netA.layers{i}, 'learningRate')
                netA.layers{i}.learningRate = ones(1, 2, 'single') ;
            end
            if ~isfield(netA.layers{i}, 'weightDecay')
                netA.layers{i}.weightDecay = single([1 0]) ;
            end
        end
    end
    %% ==================End for net A==================== %%
    
    %%
    %% ==================for net T==================== %%
    for i=1:numel(netT.layers)
        if isfield(netT.layers{i}, 'weights')
            J = numel(netT.layers{i}.weights) ;
            for j=1:J
                netT.layers{i}.momentum{j} = zeros(size(netT.layers{i}.weights{j}), 'single') ;
            end
            if ~isfield(netT.layers{i}, 'learningRate')
                netT.layers{i}.learningRate = ones(1, J, 'single') ;
            end
            if ~isfield(netT.layers{i}, 'weightDecay')
                netT.layers{i}.weightDecay = ones(1, J, 'single') ;
            end
        end
        % Legacy code: will be removed
        if isfield(netT.layers{i}, 'filters')
            netT.layers{i}.momentum{1} = zeros(size(netT.layers{i}.filters), 'single') ;
            netT.layers{i}.momentum{2} = zeros(size(netT.layers{i}.biases), 'single') ;
            if ~isfield(netT.layers{i}, 'learningRate')
                netT.layers{i}.learningRate = ones(1, 2, 'single') ;
            end
            if ~isfield(netT.layers{i}, 'weightDecay')
                netT.layers{i}.weightDecay = single([1 0]) ;
            end
        end
    end
    %% ==================End for net T==================== %%
end

% setup GPUs
numGpus = numel(opts.gpus) ;
if numGpus > 1
  if isempty(gcp('nocreate')),
    parpool('local',numGpus) ;
    spmd, gpuDevice(opts.gpus(labindex)), end
  end
elseif numGpus == 1
  gpuDevice(opts.gpus)
end
if exist(opts.memoryMapFile), delete(opts.memoryMapFile) ; end

% setup error calculation function
if isstr(opts.errorFunction)
  switch opts.errorFunction
    case 'none'
      opts.errorFunction = @error_none ;
    case 'multiclass'
      opts.errorFunction = @error_multiclass ;
      if isempty(opts.errorLabels), opts.errorLabels = {'top1e', 'top5e'} ; end
    case 'binary'
      opts.errorFunction = @error_binary ;
      if isempty(opts.errorLabels), opts.errorLabels = {'bine'} ; end
    case 'hazesquareloss'
      opts.errorFunction = @error_squareloss ;
      if isempty(opts.errorLabels), opts.errorLabels = {'hazesquareloss'} ; end
    case 'hazesquareRobustloss'
      opts.errorFunction = @error_squareloss ;
      if isempty(opts.errorLabels), opts.errorLabels = {'hazesquareRobustloss'} ; end
    case 'hazesquareGradientloss'
      opts.errorFunction = @error_squareloss ;
      if isempty(opts.errorLabels), opts.errorLabels = {'hazesquareGradientloss'} ; end    
    otherwise
      error('Uknown error function ''%s''', opts.errorFunction) ;
  end
end

% -------------------------------------------------------------------------
%                                                        Train and validate
% -------------------------------------------------------------------------

for epoch=1:opts.numEpochs
  learningRate = opts.learningRate(min(epoch, numel(opts.learningRate))) ;
  learningRate = learningRate/ (1 + epoch * 1e-5 );

  % fast-forward to last checkpoint
  modelPath = @(ep) fullfile(opts.expDir, sprintf('net-epoch-%d.mat', ep));
  modelFigPath = fullfile(opts.expDir, 'net-train.pdf') ;
  if opts.continue
    if exist(modelPath(epoch),'file')
      if epoch == opts.numEpochs
        load(modelPath(epoch), 'netA', 'netT', 'info') ;
      end
      continue ;
    end
    if epoch > 1
      fprintf('resuming by loading epoch %d\n', epoch-1) ;
      load(modelPath(epoch-1), 'netA', 'netT', 'info') ;
    end
  end

  % train one epoch and validate
  train = opts.train(randperm(numel(opts.train))) ; % shuffle
  val = opts.val ;
  if numGpus <= 1
    [netA, netT, stats.train] = process_epoch(opts, getBatch, epoch, train, learningRate, training_path, netA, netT) ;
    [~,~,stats.val] = process_epoch(opts, getBatch, epoch, val, 0, validation_path, netA, netT) ;
  else
    spmd(numGpus)
      [netA_, netT_, stats_train_] = process_epoch(opts, getBatch, epoch, train, learningRate, training_path, netA, netT) ;
      [~, ~, stats_val_] = process_epoch(opts, getBatch, epoch, val, 0, validation_path, netA, netT) ;
    end
    netA = netA_{1} ;
    netT = netT_{1} ;
    stats.train = sum([stats_train_{:}],2) ;
    stats.val = sum([stats_val_{:}],2) ;
  end

  % save
  if evaluateMode, sets = {'val'} ; else sets = {'train', 'val'} ; end
  for f = sets
    f = char(f) ;
    n = numel(eval(f)) ;
    info.(f).speed(epoch) = n / stats.(f)(1) * max(1, numGpus) ;
    info.(f).objective(epoch) = stats.(f)(2) / n * opts.batchSize;
    info.(f).error(:,epoch) = stats.(f)(3:end) / n * opts.batchSize;
  end
  if ~evaluateMode, save(modelPath(epoch), 'netA', 'netT', 'info') ; end

  figure(1) ; clf ;
  hasError = isa(opts.errorFunction, 'function_handle') ;
  subplot(1,1+hasError,1) ;
  if ~evaluateMode
    semilogy(1:epoch, info.train.objective, '.-', 'linewidth', 2) ;
    hold on ;
  end
  semilogy(1:epoch, info.val.objective, '.--') ;
  xlabel('training epoch') ; ylabel('energy') ;
  grid on ;
  h=legend(sets) ;
  set(h,'color','none');
  title('objective') ;
  if hasError
    subplot(1,2,2) ; leg = {} ;
    if ~evaluateMode
      %plot(1:epoch, info.train.error', '.-', 'linewidth', 2) ;
      plot(1:epoch, info.train.objective, '.-', 'linewidth', 2) ;
      hold on ;
      leg = horzcat(leg, strcat('train ', opts.errorLabels)) ;
    end
    %plot(1:epoch, info.val.error', '.--') ;
    plot(1:epoch, info.val.objective, '.--') ;
    leg = horzcat(leg, strcat('val ', opts.errorLabels)) ;
    set(legend(leg{:}),'color','none') ;
    grid on ;
    xlabel('training epoch') ; ylabel('error') ;
    title('error') ;
  end
  drawnow ;
  print(1, modelFigPath, '-dpdf') ;
end

% -------------------------------------------------------------------------
function err = error_multiclass(opts, labels, res)
% -------------------------------------------------------------------------
predictions = gather(res(end-1).x) ;
[~,predictions] = sort(predictions, 3, 'descend') ;

% be resilient to badly formatted labels
if numel(labels) == size(predictions, 4)
  labels = reshape(labels,1,1,1,[]) ;
end

% skip null labels
mass = single(labels(:,:,1,:) > 0) ;
if size(labels,3) == 2
  % if there is a second channel in labels, used it as weights
  mass = mass .* labels(:,:,2,:) ;
  labels(:,:,2,:) = [] ;
end

error = ~bsxfun(@eq, predictions, labels) ;
err(1,1) = sum(sum(sum(mass .* error(:,:,1,:)))) ;
err(2,1) = sum(sum(sum(mass .* min(error(:,:,1:5,:),[],3)))) ;

% -------------------------------------------------------------------------
function err = error_binary(opts, labels, res)
% -------------------------------------------------------------------------
predictions = gather(res(end-1).x) ;
error = bsxfun(@times, predictions, labels) < 0 ;
err = sum(error(:)) ;

function err = error_squareloss(opts, labels, res)
% -------------------------------------------------------------------------
predictions = gather(res(end-1).x) ;
a=(predictions-labels).^2;
err=0.5*sum(a(:))/ size(labels,4);

% -------------------------------------------------------------------------
function err = error_none(opts, labels, res)
% -------------------------------------------------------------------------
err = zeros(0,1) ;

% -------------------------------------------------------------------------
function  [netA_cpu, netT_cpu, stats, prof] = process_epoch(opts, getBatch, epoch, subset, learningRate, data_path, netA_cpu, netT_cpu)
% -------------------------------------------------------------------------

% move CNN to GPU as needed
numGpus = numel(opts.gpus) ;
if numGpus >= 1
  netA = vl_simplenn_move(netA_cpu, 'gpu') ;
  netT = vl_simplenn_move(netT_cpu, 'gpu') ;
else
  netA = netA_cpu ;
  netA_cpu = [] ;
  %%
  netT = netT_cpu ;
  netT_cpu = [] ;
end

% validation mode if learning rate is zero
training = learningRate > 0 ;
if training, mode = 'training' ; else, mode = 'validation' ; end
if nargout > 3, mpiprofile on ; end

numGpus = numel(opts.gpus) ;
if numGpus >= 1
  one = gpuArray(single(1)) ;
else
  one = single(1) ;
end
resA = [] ; %% for network A
%%
resT = [] ; %% for network T
%%
mmap = [] ;
stats = [] ;

%----------------------------------------%
%----------------------------------------------------
points_seen = 0;
display_points = 5000;
save_points = 50000;
net_work_number = 0;
modelPath_sub = @(ep, ep2) fullfile(opts.expDir, sprintf('net-epoch-%d-%d.mat', ep, ep2));
%----------------------------------------------------
%%
listing = dir(strcat(data_path,'/*.mat')); %% loop all the training data
for pp  = 1:3              % for 600 pics
    if training
        dataname_mat = strcat(data_path, 'patches_', num2str(pp), '.mat');
        load(dataname_mat); 
        patch = 1:size(samples,4);
        subset = patch(randperm(numel(patch)));
    else %loop all the validation data
        dataname_mat = strcat(data_path, 'val_', num2str(pp), '.mat');
        load(dataname_mat); 
        subset = 1:size(test_samples,4);
    end   
       
    for t=1:opts.batchSize:numel(subset)
        %%
        points_seen = points_seen + opts.batchSize;
        
        fprintf('%s: epoch %02d: batch %3d/%3d: ', mode, epoch, ...
            fix(t/opts.batchSize)+1, ceil(numel(subset)/opts.batchSize)) ;
        batchSize = min(opts.batchSize, numel(subset) - t + 1) ;
        batchTime = tic ;
        numDone = 0 ;
        error = [] ;
        for s=1:opts.numSubBatches
            % get this image batch and prefetch the next
            batchStart = t + (labindex-1) + (s-1) * numlabs ;
            batchEnd = min(t+opts.batchSize-1, numel(subset)) ;
            batch = subset(batchStart : opts.numSubBatches * numlabs : batchEnd) ;
            
            %[im, labels] = getBatch(imdb, batch) ;
            if training
                [in, out] = getBatch(samples, labels, depths, batch, opts.pad_outputsize) ;
            else
                [in, out] = getBatch(test_samples, test_labels, test_depths, batch, opts.pad_outputsize) ;
            end
            
            if opts.prefetch
                if s==opts.numSubBatches
                    batchStart = t + (labindex-1) + opts.batchSize ;
                    batchEnd = min(t+2*opts.batchSize-1, numel(subset)) ;
                else
                    batchStart = batchStart + numlabs ;
                end
                nextBatch = subset(batchStart : opts.numSubBatches * numlabs : batchEnd) ;
                %getBatch(imdb, nextBatch) ;
                if training
                    getBatch(samples, labels, depths, nextBatch) ;
                else
                    getBatch(test_samples, test_labels, test_depths, nextBatch) ;
                end
            end
            
            if numGpus >= 1
                in = gpuArray(in) ;
                out = gpuArray(out) ;
            end
            
            % evaluate CNN
            netA.layers{end}.class = out ;
            netT.layers{end}.class = out ;
            if training, dzdy = one; else, dzdy = [] ; end
            [resA, resT] = vl_simplenn_grad(netA, netT, in, dzdy, resA, resT, opts.pad_outputsize, ...
                'accumulate', s ~= 1, ...
                'disableDropout', ~training, ...
                'conserveMemory', opts.conserveMemory, ...
                'backPropDepth', opts.backPropDepth, ...
                'sync', opts.sync, ...
                'cudnn', opts.cudnn) ;
            
            
            % accumulate training errors
%             error = sum([error, [...
%                 sum(double(gather(resA(end).x))) ;
%                 reshape(opts.errorFunction(opts, out, resA),[],1) ; ]],2) ;
            %%
            error = sum([error, sum(double(gather(resA(end).x)))],2);
            numDone = numDone + numel(batch) ;
        end
        
        % gather and accumulate gradients across labs
        if training
            if numGpus <= 1
                %[net,res] = accumulate_gradients(opts, learningRate, batchSize, net, res) ;
                [netA, netT, resA, resT] = accumulate_gradients(opts, learningRate, batchSize, netA, netT, resA, resT) ;
            else %not use
                if isempty(mmap)
                    %mmap = map_gradients(opts.memoryMapFile, net, res, numGpus) ;
                    mmap = map_gradients(opts.memoryMapFile, netA, resA, numGpus) ;
                end
                write_gradients(mmap, netA, resA) ;
                labBarrier() ;
                %[net,res] = accumulate_gradients(opts, learningRate, batchSize, net, res, mmap) ;
                [netA, netT, resA, resT] = accumulate_gradients(opts, learningRate, batchSize, netA, netT, resA, resT) ;
            end
        end
        
        % print learning statistics
        batchTime = toc(batchTime) ;
        stats = sum([stats,[batchTime ; error]],2); % works even when stats=[]
        speed = batchSize/batchTime ;
        
        fprintf(' %.5f s (%.1f data/s)', batchTime, speed) ;
        %n = (t + batchSize - 1) / max(1,numlabs) / batchSize;
        %fprintf(' obj:%.3g', stats(2)/n) ;
        fprintf(' obj:%.3g', double(gather(resA(end).x))) ;
        for i=1:numel(opts.errorLabels)
            %fprintf(' %s:%.3g', opts.errorLabels{i}, stats(i+2)/n) ;
            fprintf(' %s:%.3g', opts.errorLabels{i}, double(gather(resA(end).x))) ;
        end
        fprintf(' [%d/%d]', numDone, batchSize);
        fprintf('\n') ;
        
        % debug info
        if opts.plotDiagnostics && numGpus <= 1
            figure(2) ; vl_simplenn_diagnose(netA,resA) ; drawnow ;
        end
        %%-------------------------------------------------%%
        %% save point
        if(mod(points_seen, save_points) == 0)
            %fprintf('\n%s\n', datestr(now, 'dd-mm-yyyy HH:MM:SS FFF'));
            net_work_number = net_work_number + 1;
            save(modelPath_sub(epoch, net_work_number), 'netA', 'netT') ;
        end
    end
end
%----------------end training data path------------------%

if nargout > 3
  prof = mpiprofile('info');
  mpiprofile off ;
end

if numGpus >= 1
  netA_cpu = vl_simplenn_move(netA, 'cpu') ;
  netT_cpu = vl_simplenn_move(netT, 'cpu') ;
else
  netA_cpu = netA ;
  netT_cpu = netT ;
end

% -------------------------------------------------------------------------
function [netA, netT, resA, resT] = accumulate_gradients(opts, lr, batchSize, netA, netT, resA, resT, mmap)
% -------------------------------------------------------------------------
for l=numel(netA.layers):-1:1
  for j=1:numel(resA(l).dzdw)
    thisDecay = opts.weightDecay * netA.layers{l}.weightDecay(j) ;
    thisLR = lr * netA.layers{l}.learningRate(j) ;

    % accumualte from multiple labs (GPUs) if needed
    if nargin >= 8
      tag = sprintf('l%d_%d',l,j) ;
      tmp = zeros(size(mmap.Data(labindex).(tag)), 'single') ;
      for g = setdiff(1:numel(mmap.Data), labindex)
        tmp = tmp + mmap.Data(g).(tag) ;
      end
      resA(l).dzdw{j} = resA(l).dzdw{j} + tmp ;
    end

    if isfield(netA.layers{l}, 'weights')
      netA.layers{l}.momentum{j} = ...
        opts.momentum * netA.layers{l}.momentum{j} ...
        - thisDecay * netA.layers{l}.weights{j} ...
        - (1 / batchSize) * resA(l).dzdw{j} ;
      netA.layers{l}.weights{j} = netA.layers{l}.weights{j} + thisLR * netA.layers{l}.momentum{j} ;
    else
      % Legacy code: to be removed
      if j == 1
        netA.layers{l}.momentum{j} = ...
          opts.momentum * netA.layers{l}.momentum{j} ...
          - thisDecay * netA.layers{l}.filters ...
          - (1 / batchSize) * resA(l).dzdw{j} ;
        netA.layers{l}.filters = netA.layers{l}.filters + thisLR * netA.layers{l}.momentum{j} ;
      else
        netA.layers{l}.momentum{j} = ...
          opts.momentum * netA.layers{l}.momentum{j} ...
          - thisDecay * netA.layers{l}.biases ...
          - (1 / batchSize) * resA(l).dzdw{j} ;
        netA.layers{l}.biases = netA.layers{l}.biases + thisLR * netA.layers{l}.momentum{j} ;
      end
    end
  end
end

%%==========================for network T===============================%%
for l=numel(netT.layers):-1:1
  for j=1:numel(resT(l).dzdw)
    thisDecay = opts.weightDecay * netT.layers{l}.weightDecay(j) ;
    thisLR = lr * netT.layers{l}.learningRate(j) ;

    % accumualte from multiple labs (GPUs) if needed
    if nargin >= 8
      tag = sprintf('l%d_%d',l,j) ;
      tmp = zeros(size(mmap.Data(labindex).(tag)), 'single') ;
      for g = setdiff(1:numel(mmap.Data), labindex)
        tmp = tmp + mmap.Data(g).(tag) ;
      end
      resT(l).dzdw{j} = resT(l).dzdw{j} + tmp ;
    end

    if isfield(netT.layers{l}, 'weights')
      netT.layers{l}.momentum{j} = ...
        opts.momentum * netT.layers{l}.momentum{j} ...
        - thisDecay * netT.layers{l}.weights{j} ...
        - (1 / batchSize) * resT(l).dzdw{j} ;
      netT.layers{l}.weights{j} = netT.layers{l}.weights{j} + thisLR * netT.layers{l}.momentum{j} ;
    else
      % Legacy code: to be removed
      if j == 1
        netT.layers{l}.momentum{j} = ...
          opts.momentum * netT.layers{l}.momentum{j} ...
          - thisDecay * netT.layers{l}.filters ...
          - (1 / batchSize) * resT(l).dzdw{j} ;
        netT.layers{l}.filters = netT.layers{l}.filters + thisLR * netT.layers{l}.momentum{j} ;
      else
        netT.layers{l}.momentum{j} = ...
          opts.momentum * netT.layers{l}.momentum{j} ...
          - thisDecay * netT.layers{l}.biases ...
          - (1 / batchSize) * resT(l).dzdw{j} ;
        netT.layers{l}.biases = netT.layers{l}.biases + thisLR * netT.layers{l}.momentum{j} ;
      end
    end
  end
end
%%========================================================================%%
% -------------------------------------------------------------------------
function mmap = map_gradients(fname, net, res, numGpus)
% -------------------------------------------------------------------------
format = {} ;
for i=1:numel(net.layers)
  for j=1:numel(res(i).dzdw)
    format(end+1,1:3) = {'single', size(res(i).dzdw{j}), sprintf('l%d_%d',i,j)} ;
  end
end
format(end+1,1:3) = {'double', [3 1], 'errors'} ;
if ~exist(fname) && (labindex == 1)
  f = fopen(fname,'wb') ;
  for g=1:numGpus
    for i=1:size(format,1)
      fwrite(f,zeros(format{i,2},format{i,1}),format{i,1}) ;
    end
  end
  fclose(f) ;
end
labBarrier() ;
mmap = memmapfile(fname, 'Format', format, 'Repeat', numGpus, 'Writable', true) ;

% -------------------------------------------------------------------------
function write_gradients(mmap, net, res)
% -------------------------------------------------------------------------
for i=1:numel(net.layers)
  for j=1:numel(res(i).dzdw)
    mmap.Data(labindex).(sprintf('l%d_%d',i,j)) = gather(res(i).dzdw{j}) ;
  end
end
