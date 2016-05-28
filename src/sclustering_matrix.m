clc, clear all, close all;

addpath('/home/sid/Dev/SVM-Classification/resources/svm')
addpath('/home/sid/Dev/SVM-Classification/resources/SVMCourse/LSSVMlab')
addpath('export_fig')
load '../datasets/two3drings'

[N,d]=size(X);

perm=randperm(N);   % shuffle the data
X=X(perm,:);
sig2=0.05;              % set the kernel parameters
K=kernel_matrix(X,'RBF_kernel',sig2);   %compute the RBF kernel (affinity) matrix
D=diag(sum(K));         % compute the degree matrix (sum of the columns of K)
[U,lambda]=eigs(inv(D)*K,3);  % Compute the 3 largest eigenvalues/vectors using Lanczos
                              % The largest eigenvector does not contain
                              % clustering information. For binary clustering,
                              % the solution is the second largest eigenvector.
 
clust=sign(U(:,2)); % Threshold the eigenvector solution to obtain binary cluster indicators
[y,order]=sort(clust,'descend');    % Sort the data using the cluster information
Xsorted=X(order,:);
Ksorted=kernel_matrix(Xsorted,'RBF_kernel',sig2);   % Compute the kernel matrix of the
                                                    % sorted data.
proj=K*U(:,2:3);    % Compute the projections onto the subspace spanned by the second, and third largest eigenvectors.

% Matrix
figure('Color',[1 1 1]);
subplot(2,2,1);
imshow(K);
title('Kernel matrix of the original data (\sigma2=0.005)', 'FontSize', 18, 'FontWeight', 'normal');

subplot(2,2,3);
imshow(Ksorted);
title('Kernel matrix after sorting the data using the cluster information', 'FontSize', 18, 'FontWeight', 'normal');

% second
sig2=0.005;              % set the kernel parameters
K=kernel_matrix(X,'RBF_kernel',sig2);   %compute the RBF kernel (affinity) matrix
D=diag(sum(K));         % compute the degree matrix (sum of the columns of K)
[U,lambda]=eigs(inv(D)*K,3);  % Compute the 3 largest eigenvalues/vectors using Lanczos
                              % The largest eigenvector does not contain
                              % clustering information. For binary clustering,
                              % the solution is the second largest eigenvector.
 
clust=sign(U(:,2)); % Threshold the eigenvector solution to obtain binary cluster indicators
[y,order]=sort(clust,'descend');    % Sort the data using the cluster information
Xsorted=X(order,:);
Ksorted=kernel_matrix(Xsorted,'RBF_kernel',sig2);   % Compute the kernel matrix of the
                                                    % sorted data.
proj=K*U(:,2:3);    % Compute the projections onto the subspace spanned by the second, and third largest eigenvectors.

% Matrix
subplot(2,2,2);
imshow(K);
title('Kernel matrix of the original data (\sigma2=0.05)', 'FontSize', 18, 'FontWeight', 'normal');

subplot(2,2,4);
imshow(Ksorted);
title('Kernel matrix after sorting the data using the cluster information', 'FontSize', 18, 'FontWeight', 'normal');