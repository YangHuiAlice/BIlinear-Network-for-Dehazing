function Y = vl_nnsquareloss(X, label, dzdy)
%
%
resdual_error=(X-label);
if nargin <= 2 % compute energy value
    Y=0.5*sum(resdual_error(:).^2)./ size(X,4);
else % compute delta for gradient
    Y = (resdual_error)*dzdy./size(X,4);
end