function [delta_a, delta_t, energy_val] = vl_nnhazesquareloss(A, T, labels, in, dzdy)
%%%%%%%%%%%%%%%%% Error for hazy image %%%%%%%%%%%%%%%%%%%
batch_num = size(in,4);
%
J = labels(:,:,1:3,:);
gt_T = labels(:,:,4:end,:);

A_ini = A;
T_ini = T;

%%% The whole loss fcuntion %%%%%%%%%
% E1 = J * T' + (1-T')*A' – hazy;
% E2 = T' – T_GT;
% delta_t = G( E1 *(J-A')) + G(E2);
% delta_A = (1-T')*E1;

% calculate E1 and E2 
T = repmat(T, [1,1, size(J,3), 1]);
hazyimg = T.* J + (1-T).*A;               % The reconstructed hazy image from learned A and T
resdual_error1=(hazyimg-in(:,:,1:3,:));   % E1---composition error
resdual_error2=(T_ini-gt_T);              % E2---T error
    
% The overall loss function
energy_val = 0.5 * sum(resdual_error1(:).^2)./ batch_num +0.5 * sum(resdual_error2(:).^2)./ batch_num ;

%% Do this if back propagation is not needed
if nargin <= 4 % return energy value
    delta_a = 0;
    delta_t = 0;
    
% Do back propagation
else  
    delta_a = resdual_error1.*(1-T);
    
    delta_t1 = resdual_error1.*(J-A);
    delta_t1 = gather(delta_t1);
    
    delta_t2 = resdual_error2;
    delta_t2 = gather(delta_t2);
    
    in = gather(in);
    delta_t_tmp_E1 = zeros(size(J,1),size(J,2),3,batch_num);
    delta_t_tmp_E2 = zeros(size(J,1),size(J,2),1,batch_num);
    % Calculate derivative for image guided filter
    for i=1:batch_num
        delta_t_tmp_E1(:,:,:,i) = imguidedfilter(delta_t1(:,:,:,i),in(:,:,:,i),'NeighborhoodSize',3,'DegreeOfSmoothing',0.01);
        delta_t_tmp_E2(:,:,:,i) = imguidedfilter(delta_t2(:,:,:,i),in(:,:,:,i),'NeighborhoodSize',3,'DegreeOfSmoothing',0.01);
    end
    delta_t = single(sum(delta_t_tmp_E1,3)./3 + delta_t_tmp_E2);
    delta_t = gpuArray(delta_t); 
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