clc, clear all, close all;

addpath('/home/sid/Dev/SVM-Classification/resources/svm')
addpath('/home/sid/Dev/SVM-Classification/resources/SVMCourse/LSSVMlab')
addpath('export_fig')

nb = 400; sig = 0.3; nb=nb/2;

% construct dataset
leng = 1;
for t=1:nb, 
  yin(t,:) = [2.*sin(t/nb*pi*leng) 2.*cos(.61*t/nb*pi*leng) (t/nb*sig)]; 
  yang(t,:) = [-2.*sin(t/nb*pi*leng) .45-2.*cos(.61*t/nb*pi*leng) (t/nb*sig)]; 
  samplesyin(t,:)  = [yin(t,1)+yin(t,3).*randn   yin(t,2)+yin(t,3).*randn];
  samplesyang(t,:) = [yang(t,1)+yang(t,3).*randn   yang(t,2)+yang(t,3).*randn];
end

% plot dataset
h=figure('Color', [1 1 1]); hold on
plot(samplesyin(:,1),samplesyin(:,2),'o');
plot(samplesyang(:,1),samplesyang(:,2),'o');
xlabel('X_1');
ylabel('X_2');
title('Structured dataset');
export_fig('kpca_dataset.pdf');

% get user-defined parameters
nc = input('\n Number of components to be extracted? [6] '); if isempty(nc) nc=6; end;
sig2 = input('\n RBF kernel parameter sig2? [0.4] '); if isempty(sig2) sig2=0.4; end;
approx = input('\n Approximation technique ? 1 for ''Lanczos'', 2 for ''Nystrom'' [1] '); if isempty(approx) approx=1; end;

if approx ==1
    approx='eigs';
else
    approx='eign';
end

% varying number of components
figure('Color', [1 1 1]);
count = 1;
for num_comp=[1 2 4 8 16 32]
    [lam,U] = kpca([samplesyin;samplesyang],'RBF_kernel',sig2,[],approx,num_comp);
    xd = denoise_kpca([samplesyin;samplesyang],'RBF_kernel',sig2,[],approx,num_comp);
    subplot(2,3,count)
    hold on
    plot(samplesyin(:,1),samplesyin(:,2),'.');
    plot(samplesyang(:,1),samplesyang(:,2),'.');
    plot(xd(:,1),xd(:,2),'r+');
    hold off
    title(['number of components: ' num2str(num_comp)], 'FontSize', 18, 'FontWeight', 'normal');
    count = count + 1;
end
export_fig('kpca_ncomp.pdf');


figure('Color', [1 1 1]);
count = 1;
nc = 6;
for sig2=[0.1 0.5 1 5 10 50]
    
    nc;
    [lam,U] = kpca([samplesyin;samplesyang],'RBF_kernel',sig2,[],approx,nc);

    xd = denoise_kpca([samplesyin;samplesyang],'RBF_kernel',sig2,[],approx,nc);
    subplot(2,3,count)
    hold on
    plot(samplesyin(:,1),samplesyin(:,2),'.','MarkerSize',5);
    plot(samplesyang(:,1),samplesyang(:,2),'.', 'MarkerSize',5);
    plot(xd(:,1),xd(:,2),'r+');
    hold off
    title(['RBF sigma^2: ' num2str(sig2)], 'FontSize', 18, 'FontWeight', 'normal');
    count = count + 1
end
export_fig('kpca_sigma.pdf');

disp('Press any key to continue');
pause;


% get user-defined parameters
nc = input('\n Number of components to be extracted? [6] '); if isempty(nc) nc=6; end;
sig2 = input('\n RBF kernel parameter sig2? [0.4] '); if isempty(sig2) sig2=0.4; end;
approx = input('\n Approximation technique ? 1 for ''Lanczos'', 2 for ''Nystrom'' [1] '); if isempty(approx) approx=1; end;

if approx ==1
    approx='eigs';
else
    approx='eign';
end


% calculate the eigenvectors in the feature space (principal components)
[lam,U] = kpca([samplesyin;samplesyang],'RBF_kernel',sig2,[],approx,nc);

% Denoise the data by minimizing the reconstruction error
xd = denoise_kpca([samplesyin;samplesyang],'RBF_kernel',sig2,[],approx,nc);
%h2=figure;
plot(samplesyin(:,1),samplesyin(:,2),'o');
plot(samplesyang(:,1),samplesyang(:,2),'o');
plot(xd(:,1),xd(:,2),'r+');
title('Kernel PCA - Denoised datapoints in red', 'FontSize', 18, 'FontWeight', 'normal');

disp('Press any key to continue');
pause;

% Projections on the first component using linear PCA
dat=[samplesyin;samplesyang];
dat(:,1)=dat(:,1)-mean(dat(:,1));
dat(:,2)=dat(:,2)-mean(dat(:,2));
[lam_lin,U_lin] = pca(dat);

%proj_lin=grid*U_lin;
figure;
plot(samplesyin(:,1),samplesyin(:,2),'o');hold on;
plot(samplesyang(:,1),samplesyang(:,2),'o');
%contour(Xax,Yax,reshape(proj_lin(:,1),length(Yax),length(Xax)));
xdl=dat*U_lin(:,1)*U_lin(:,1)';
plot(xdl(:,1),xdl(:,2),'r+');
title('Linear PCA - Denoised data points using the first principal component', 'FontSize', 18, 'FontWeight', 'normal');
export_fig('kpca_linear.pdf');
