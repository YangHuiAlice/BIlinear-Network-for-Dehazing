function [delta, loss] = vl_nnsparsegradA(featuremap)
%
weight_grad = 0.01;
%
[dx,dy] = cal_diff(featuremap);
%
[dloss,delta] = GradSparse(dx,dy);
loss = sum(dloss(:))/size(featuremap,4);
%delta = weight_grad*single(delta) + (featuremap - 1);
%%
ddx = zeros(size(featuremap));
%%%%%%%%%%%%%%%%%%%%%%%%%%%
ddx=gpuArray(ddx);
%%%%%%%%%%%%%%%%%%%%%%%%%%%
ddy = ddx;
for chs = 1:size(dx,3)
    for batch_num = 1:size(dx,4)
        dx_tmp = dx(:,:,chs,batch_num);
        ddx(:,:,chs,batch_num) = [dx_tmp(:,end,:) - dx_tmp(:, 1,:), -diff(dx_tmp,1,2)]; % transpose
        dy_tmp = dy(:,:,chs,batch_num); 
        ddy(:,:,chs,batch_num) = [dy_tmp(end,:,:) - dy_tmp(1, :,:); -diff(dy_tmp,1,1)];  % transpose
    end
end
%%
delta = single(ddx + ddy);
end


function [loss,delta] = GradSparse(dx, dy)
%
thr_e=0.01; 
%
delta = zeros(size(dx));
loss = 0;
%%%%%%%%%%%%%%%%%%%%
delta=gpuArray(delta);
loss=gpuArray(loss);
%%%%%%%%%%%%%%%%%%%%
for chs = 1:size(dx,3)
    for batch_num = 1:size(dx,4)
        
        dx_tmp = dx(:,:,chs,batch_num);
        ddx = [dx_tmp(:,end,:) - dx_tmp(:, 1,:), -diff(dx_tmp,1,2)]; % transpose 
        %
        dy_tmp = dy(:,:,chs,batch_num); 
        ddy = [dy_tmp(end,:,:) - dy_tmp(1, :,:); -diff(dy_tmp,1,1)];
        %
        denorminx = sqrt(abs(dx_tmp).^2 + 1e-3); 
        denorminy = sqrt(abs(dy_tmp).^2 + 1e-3); 
        %weightx = exp_a*max(abs(dx),thr_e).^(exp_a-1);
        %weighty = exp_a*max(abs(dy),thr_e).^(exp_a-1);
        %delta(:,:,chs,batch_num) = weightx.*ddx + weighty.*ddy;
        delta(:,:,chs,batch_num) = ddx./denorminx + ddy./denorminy;
        loss = loss + sum(denorminx(:))+sum(denorminy(:));
    end
end
loss = loss./size(dx,4);
end

function [dx,dy] = cal_diff(input)
dy = zeros(size(input));
dx = zeros(size(input));
for i = 1:size(input, 4)
    tmp = input(:,:,:,i);
    %%%%%%%%%%%
    tmp=gather(tmp);
    %%%%%%%%%%%
    dx(:,:,:,i) = [diff(tmp,1,2), tmp(:,1,:) - tmp(:,end,:)];
    dy(:,:,:,i) = [diff(tmp,1,1); tmp(1,:,:) - tmp(end,:,:)];
end
%%%%%%%%%%%%%%%%%%%
dx=gpuArray(single(dx));
dy=gpuArray(single(dy));
%%%%%%%%%%%%%%%%%%%
end
