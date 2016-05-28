clc, clear all, close all;

addpath('/home/sid/Dev/SVM-Classification/resources/svm')
addpath('/home/sid/Dev/SVM-Classification/resources/SVMCourse')
addpath('export_fig')

bc = load('../datasets/shuttle.dat', '-ascii');

[trainInd,valInd,testInd] = dividerand(58000,0.8,0.0,0.2);

X = bc(trainInd,1:end-1);
Y = bc(trainInd,end);

testX = bc(testInd,1:end-1);
testY = bc(testInd,end);

%Parameter for input space selection
%Please type >> help fsoperations; to get more information  

k = 6;
function_type = 'c'; %'c' - classification, 'f' - regression  
kernel_type = 'RBF_kernel'; % or 'lin_kernel', 'poly_kernel'
global_opt = 'csa'; % 'csa' or 'ds'

%Process to be performed
user_process={'FS-LSSVM', 'SV_L0_norm'};
window = [15,20,25];

[e,s,t] = fslssvm(X,Y,k,function_type,kernel_type,global_opt,user_process,window,testX,testY);

% saved results (executed on supercomputer, takes a very long time)
load 'fslssvm_shuttle.mat';

process_matrix_err =  e; process_matrix_sv = s; process_matrix_time = t;
figure('Color', [1 1 1]);
subplot(1,3,1);
boxplot(process_matrix_err, 'Label',user_process);
ylabel('Error estimate');
title('Error Comparison','FontSize',18,'FontWeight', 'normal');
subplot(1,3,2);
boxplot(process_matrix_sv,'Label', user_process);
ylabel('SV estimate');
title('SV comparison','FontSize',18,'FontWeight', 'normal');
subplot(1,3,3);
boxplot(process_matrix_time,'Label',user_process);
ylabel('Time estimate');
title('Time comparison','FontSize',18,'FontWeight', 'normal');