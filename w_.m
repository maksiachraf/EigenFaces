function [W] = w_(Xim,l_min,U,x)
%W_ Summary of this function goes here
%   Detailed explanation goes here
W=zeros(l_min,1);
for k=1:l_min
W(k)=transpose(U(:,k))*(Xim-x);
end

end

