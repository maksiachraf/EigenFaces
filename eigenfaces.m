% P. Vallet (Bordeaux INP), 2019

clc;
clear all;
close all;

%% Data extraction
% Training set
adr = './database/training1/';
fld = dir(adr);
nb_elt = length(fld);
% Data matrix containing the training images in its columns 
data_trn = []; 
% Vector containing the class of each training image
lb_trn = []; 
for i=1:nb_elt
    if fld(i).isdir == false
        lb_trn = [lb_trn ; str2num(fld(i).name(6:7))];
        img = double(imread([adr fld(i).name]));
        data_trn = [data_trn img(:)];
    end
end

% Testing set
adr = './database/test1/';
fld = dir(adr);
nb_elt = length(fld);
% Data matrix containing the testing images in its columns 
data_test = []; 
% Vector containing the class of each testing images
lb_test = []; 
for i=1:nb_elt
    if fld(i).isdir == false
        lb_test = [lb_test ; str2num(fld(i).name(6:7))];
        img = double(imread([adr fld(i).name]));
        data_test = [data_test img(:)];
    end
end
[~,Nb_test]=size(data_test);

% Size of the training set
[P,N] = size(data_trn);
% Classes contained in the training set
[~,I]=sort(lb_trn);
data_trn = data_trn(:,I);
[cls_trn,bd,~] = unique(lb_trn);
Nc = length(cls_trn); 
% Number of training images in each class
size_cls_trn = [bd(2:Nc)-bd(1:Nc-1);N-bd(Nc)+1]; 

% Display the database
% F = zeros(192*Nc,168*max(size_cls_trn));
% for i=1:Nc
%     for j=1:size_cls_trn(i)
%           pos = sum(size_cls_trn(1:i-1))+j;
%           F(192*(i-1)+1:192*i,168*(j-1)+1:168*j) = reshape(data_trn(:,pos),[192,168]);
%     end
% end
% figure;
% imagesc(F);
% colormap(gray);
% axis off;

%% ACL
X=zeros(P,N); 

x=transpose(mean(transpose(data_trn)));
for i=1:N
    X(:,i)= data_trn(:,i)-x;
end

X=X/sqrt(N);

Gram =transpose(X)*X;  %Matrice de Gramm

[V,D]=eig(Gram);

U=X*V*(transpose(V)*transpose(X)*X*V)^(-1/2);

figure;
for i=1:N
subplot(6,10,i);
imagesc(reshape(U(:,i),[192,168]));
colormap("gray");
end

l=[5 20 40 60];

Xim = data_trn(:,1);

Proj=zeros(P,4);
for j=1:4
for i=1:l(j)
Proj(:,j) = Proj(:,j) + (transpose(U(:,i))*(Xim-x))*U(:,i);
end

L=1:1:60;

% Affichage des Eigenfaces
figure;

for i=1:4
subplot(1,4,i)
imagesc(reshape(Proj(:,i)+x,[192,168]));
colormap("gray")
end

end

%% Determination de la taille l optimale du sous-espace

eig_val=sort(diag(D),'descend');

K_l=zeros(1,N);

for l=1:N
   K_l(l)=sum(eig_val(1:l))/ sum(eig_val(1:N));
end

[~,G]=find(K_l > 0.9);
l_min=G(1);

%% Classifieur K-NN
k=5;

W=zeros(l_min,N);

for i=1:N
   W(:,i) = w_(data_trn(:,i),l_min,U,x);
end  

% Evaluation des performances du classifieur KNN
Estim_KNN=[];

for t=1:Nb_test
    
w_test = w_(data_test(:,t),l_min,U,x);

Estim_KNN = [ Estim_KNN predic_kNN(k,W,w_test,cls_trn)]; 

end

[C,err_rate] = confmat(lb_test,Estim_KNN')

%% Classifieur Gaussien

Means=zeros(l_min,Nc);
Cov=zeros(l_min,l_min);

for i = 1:Nc
    for j=bd(i):bd(i)+9
        Means(:,i)=Means(:,i)+W(:,j);
    end
    Means(:,i)=Means(:,i)/10;
end

for i = 1:Nc
    for j=bd(i):bd(i)+9
       Cov=Cov+(W(:,j)-Means(:,i))*transpose(W(:,j)-Means(:,i));
    end
end

Cov=Cov/N;

% Evaluation des performances du classifieur Gaussien
Estim_gauss=[];

for t=1:Nb_test

w_test = w_(data_test(:,t),l_min,U,x);

Estim_gauss = [Estim_gauss predic_gauss(w_test,Means,Cov,cls_trn)]; 

end

[C_gauss, err_rate_gauss] = confmat(lb_test, Estim_gauss')

