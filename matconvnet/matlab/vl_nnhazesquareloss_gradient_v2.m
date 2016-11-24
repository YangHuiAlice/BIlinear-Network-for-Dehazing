function [delta_a, delta_t, energy_val] = vl_nnhazesquareloss_gradient_v2(A, T, labels, in, dzdy)
%
%
weight_grad = 0.001;
batch_num = size(in,4);
%
J = labels(:,:,1:3,:);
gt_T = labels(:,:,4:end,:);
%
A_ini = A;
T_ini = T;
%
T = repmat(T, [1,1, size(J,3), 1]);
A = repmat(A, [1,1, size(J,3), 1]);
hazyimg = T.*J + (1-T).*A;
resdual_error=(hazyimg-in);
%
resdual_error2=(T_ini-gt_T);

energy_val = 0.1 * sum(resdual_error(:).^2)./ batch_num + 0.9 * sum(resdual_error2(:).^2)./ batch_num;

%%
% fx = [1, -1];
% fy = [1; -1];
[gradAx, gradAy] =  compute_gradient(A);
[gradTx, gradTy] =  compute_gradient(T);
[gradJx, gradJy] =  compute_gradient(J);
[gradIx, gradIy] =  compute_gradient(in);
%grad
hazy_gradx_x = gradJx.*T + J.*gradTx + gradAx.*(1 - T) - A.*gradTx;
hazy_gradx_y = gradJy.*T + J.*gradTy + gradAy.*(1 - T) - A.*gradTy;
%
resdual_error_grad_x = hazy_gradx_x - gradIx;
resdual_error_grad_y = hazy_gradx_y - gradIy;
%
energy_val = energy_val + weight_grad.*(0.5*sum(resdual_error_grad_x(:).^2)./ batch_num + 0.5*sum(resdual_error_grad_y(:).^2))./ batch_num;

%%
if nargin <= 4 % return energy value
    delta_a = 0;
    delta_t = 0;
else
%     delta_a = (resdual_error .* (1-T))*dzdy./batch_num;
%     delta_t = (resdual_error .* (J-A))*dzdy./batch_num;
    delta_a = (resdual_error .* (1-T))*dzdy;
    delta_t = (resdual_error .* (J-A))*dzdy;
    %% vector:
    delta_a = 0.2 * sum(delta_a, 3);
    delta_t = 0.2 * sum(delta_t, 3) + 1.8 * resdual_error2;
    
    %% Gradident
%     sz = [size(in, 1)*size(in, 2), size(in, 1)*size(in, 2)];
%     Mx = make_imfilter_mat(fx, sz, 'replicate', 'same');
%     My = make_imfilter_mat(fy, sz, 'replicate', 'same');
%     tmp_ttx = Mx'*ones(sz(1));
%     tmp_tty = My'*ones(sz(1));
%     tmp_ttx = reshape(tmp_ttx, size(J,1), size(J,2));
%     tmp_ttx = repmat(tmp_ttx, [1,1, size(J,3)]);
%     tmp_tty = reshape(tmp_tty, size(J,1), size(J,2));
%     tmp_tty = repmat(tmp_tty, [1,1, size(J,3)]);
    [tmp_ttx, tmp_tty] = compute_gradient_tranpose(T);
    [tmp_ttx_a, tmp_tty_a] = compute_gradient_tranpose(A);
    delta_t_grad = resdual_error_grad_x.*(gradJx - gradAx + (J - A).*tmp_ttx)...
        + resdual_error_grad_y.*(gradJy - gradAy + (J - A).*tmp_tty);
    delta_a_grad = resdual_error_grad_x.*((1 - T).*tmp_ttx_a - gradTx)...
        + resdual_error_grad_y.*((1 - T).*tmp_tty_a - gradTy);
    delta_t_grad = sum(delta_t_grad,3);
    delta_a_grad = sum(delta_a_grad,3);
    delta_a = delta_a + weight_grad*delta_a_grad;
    delta_t = delta_t + weight_grad*delta_t_grad;
end

end
