function [class] = predic_kNN(k,W,w_test,cls_trn)

[~,N]=size(W);

W_dist=zeros(1,N);

for i=1:N
W_dist(i)=norm(w_test-W(:,i));
end

[~,I]=mink(W_dist,k);

[~,class]=max([length(intersect(I,[1:10])) ,length(intersect(I,[11:20])) ,length(intersect(I,[21:30])), length(intersect(I,[31:40])) ,length(intersect(I,[41:50])) ,length(intersect(I,[51:60])) ]);

class=cls_trn(class);
end

