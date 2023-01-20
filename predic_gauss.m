function class = predic_gauss(w_test,Means,Cov,cls_trn)

[~,Nc]=size(Means);

Predict=zeros(1,Nc);

for i=1:Nc
Predict(i)=norm(Cov^(1/2)*(w_test-Means(:,i)))^2;
end

[~,class]=min(Predict);

class=cls_trn(class);
end

