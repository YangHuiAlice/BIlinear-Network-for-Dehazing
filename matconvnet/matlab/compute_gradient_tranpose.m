function [grad_x, grad_y] = compute_gradient_tranpose(T)
grad_x = zeros(size(T));
grad_x = gpuArray(grad_x);
grad_y = grad_x;
for chs = 1:size(T,3)
    for batch_num = 1:size(T,4)
        T_tmp = T(:,:,chs,batch_num);
        grad_x(:,:,chs,batch_num) = [T_tmp(:,end,:) - T_tmp(:, 1,:), -diff(T_tmp,1,2)]; % transpose
        grad_y(:,:,chs,batch_num) = [T_tmp(end,:,:) - T_tmp(1, :,:); -diff(T_tmp,1,1)];  % transpose
    end
end