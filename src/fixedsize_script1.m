clc, clear all, close all;

addpath('/home/sid/Dev/SVM-Classification/resources/svm')
addpath('/home/sid/Dev/SVM-Classification/resources/SVMCourse')
addpath('export_fig')

% dataset
X = 3.*randn(100,2);
ssize = 10;
subset = zeros(ssize,2);

figure('Color', [1 1 1]);
plot(X(:,1),X(:,2),'b*');

figure('Color', [1 1 1]);
i = 1;
for sig2 = [0.01 0.05 0.1 0.5 1 5] 
    for t = 1:100,
        % new candidate subset
        r = ceil(rand*ssize);
        candidate = [subset([1:r-1 r+1:end],:); X(t,:)];
        
        % is this candidate better than the previous?
        if kentropy(candidate, 'RBF_kernel',sig2)>...
                kentropy(subset, 'RBF_kernel',sig2),
            subset = candidate;
        end
  
        % make a figure
        subplot(2,3,i);
        plot(X(:,1),X(:,2),'b*'); hold on;
        plot(subset(:,1),subset(:,2),'ro','linewidth',6); hold off;
        title(['\sigma2: ' num2str(sig2)],'FontSize',18,'FontWeight', 'normal');
        %pause(1)
    end
    i = i + 1;
end

