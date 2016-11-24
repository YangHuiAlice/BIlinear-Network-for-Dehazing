function [delta_a, delta_t, energy_val] = vl_nnhazesquareloss(A, T, labels, in, dzdy)
%%%%%%%%%%%%%%%%% Error for hazy image %%%%%%%%%%%%%%%%%%%
batch_num = size(in,4);
%
J = labels(:,:,1:3,:);
gt_T = labels(:,:,4:end,:);
%
A_ini = A;
T_ini = T;
%
T = repmat(T, [1,1, size(J,3), 1]);

hazyimg = T.* J + (1-T).*A;
resdual_error1=(hazyimg-in(:,:,1:3,:));   % E1---composition error
resdual_error2=(T_ini-gt_T);              % E2---T error

energy_val = 0.5 * sum(resdual_error1(:).^2)./ batch_num +0.5 * sum(resdual_error2(:).^2)./ batch_num ;

%%
if nargin <= 4 % return energy value
    delta_a = 0;
    delta_t = 0;
else
    delta_a = (resdual_error1 .* (1-T))*dzdy;
    delta_t = (resdual_error1 .* (J-A))*dzdy;
    %% vector:
    delta_t = sum(delta_t, 3) + resdual_error2 ;
   
end



%%%%%%%%%%% Error for clear image %%%%%%%%%%%%
%
% tmin = 0.01;
% batch_num = size(in,4);
% %
% J = labels(:,:,1:3,:);
% gt_T = labels(:,:,4:end,:);
% %
% T_denom = max(tmin,T);
% %
% T = repmat(T_denom, [1,1, size(J,3), 1]);
% A = repmat(A, [1,1, size(J,3), 1]);
% J_syn = A + (in(:,:,1:3,:)-A)./T;
% resdual_error= J_syn-J;
% %
% resdual_error2=(T_denom-gt_T);
% 
% energy_val = 0.5 * sum(resdual_error(:).^2)./ batch_num + 0.5 * sum(resdual_error2(:).^2)./ batch_num;
% 
% %%
% if nargin <= 4 % return energy value
%     delta_a = 0;
%     delta_t = 0;
% else
%     delta_a = (resdual_error .* (1-1./T))*dzdy;
%     delta_t = (resdual_error .* ((A-in(:,:,1:3,:))./(T.^2)))*dzdy;
%     %% vector:
%     delta_a = sum(delta_a, 3);
%     delta_t = sum(delta_t, 3) + resdual_error2;
%    
% end




%%
%
% tmin = 0.1;
% batch_num = size(in,4);
% %
% T = repmat(T, [1,1, size(J,3), 1]);
% A = repmat(A, [1,1, size(J,3), 1]);
% %
% denom_T = (max(T, tmin));
% %hazyimg = T.*J + (1-T).*A;
% J_est = (in - A)./denom_T + A;
% %J_est = (in - (1-T).*A)./T;
% resdual_error=(J_est-J);
% %%
% %resdual_error2 = (T.*J + (1-T).*A - in);
% 
% energy_val = 0.5*sum(resdual_error(:).^2)./ batch_num;
% %%
% if nargin <= 4 % return energy value
%     delta_a = 0;
%     delta_t = 0;
% else
% %     delta_a = (resdual_error .* (1-T))*dzdy./batch_num;
% %     delta_t = (resdual_error .* (J-A))*dzdy./batch_num;
%     delta_a = (resdual_error .* (1-1./denom_T))*dzdy;
%     delta_t = (resdual_error .* ((A-in)./denom_T.^2))*dzdy;
%     %% vector:
%     delta_a = sum(delta_a, 3);
%     delta_t = sum(delta_t, 3);
% end