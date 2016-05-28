clc, clear all, close all;

addpath('/home/sid/Dev/SVM-Classification/resources/svm')
addpath('/home/sid/Dev/SVM-Classification/resources/SVMCourse/LSSVMlab')
addpath('export_fig')
load '../datasets/two3drings'

[N,d]=size(X);

perm=randperm(N);   % shuffle the data
X=X(perm,:);
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
proj=K*U(:,2:3);    % Compute the projections onto the subspace spanned by the second,
                    % and third largest eigenvectors.

%%%% PLOTTING SECTION %%%%       
figure('Color', [1 1 1])
subplot(1,2,1)
scatter3(X(:,1),X(:,2),X(:,3),15);
title('Two interlaced rings in a 3D space', 'FontSize', 18, 'FontWeight', 'normal');

subplot(1,2,2);
scatter3(X(:,1),X(:,2),X(:,3),30,clust);
title('Clustering results', 'FontSize', 18, 'FontWeight', 'normal');

% export_fig('scluster_data.pdf');

figure('Color',[1 1 1]);
subplot(1,2,1);
imshow(K);
title('Kernel matrix of the original data', 'FontSize', 18, 'FontWeight', 'normal');

subplot(1,2,2);
imshow(Ksorted);
title('Kernel matrix after sorting the data using the cluster information', 'FontSize', 18, 'FontWeight', 'normal');

export_fig('scluster_matrix.pdf');

figure('Color',[1 1 1]);
scatter(proj(:,1),proj(:,2),15,clust);
title('Projections onto subspace spanned by the 2nd and 3rd largest eigenvectors', 'FontSize', 18, 'FontWeight', 'normal');

figure('Color', [1 1 1]);
figure('Color', [1 1 1]);

count=1;
for sig2=[0.001 0.005 0.01 0.05 0.1 0.5]
    K=kernel_matrix(X,'RBF_kernel',sig2);
    D=diag(sum(K));
    [U,lambda]=eigs(inv(D)*K,3);                              
    clust=sign(U(:,2)); 

    figure(1)
    subplot(2, 3, count);
    scatter3(X(:,1),X(:,2),X(:,3),30,clust);
    title(['\sigma^2: ' num2str(sig2)], 'FontSize', 18, 'FontWeight', 'normal');
    
    figure(2)
    subplot(2, 3, count);
    proj=K*U(:,2:3);
    scatter(proj(:,1),proj(:,2),15,clust);    
    title(['\sigma^2: ' num2str(sig2)], 'FontSize', 18, 'FontWeight', 'normal');
    count = count + 1;
end

% export_fig('sclustering_origcl.pdf')