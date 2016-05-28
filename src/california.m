clc, clear all, close all;

addpath('/home/sid/Dev/SVM-Classification/resources/svm')
addpath('/home/sid/Dev/SVM-Classification/resources/SVMCourse')
addpath('export_fig')

bc = load('../datasets/california.dat', '-ascii');

[trainInd,valInd,testInd] = dividerand(20640,0.7,0.0,0.3);

X = bc(trainInd,1:end-1);
Y = bc(trainInd,end);

testX = bc(testInd,1:end-1);
testY = bc(testInd,end);

%Parameter for input space selection
%Please type >> help fsoperations; to get more information  

k = 6;
function_type = 'f'; %'c' - classification, 'f' - regression  
kernel_type = 'RBF_kernel'; % or 'lin_kernel', 'poly_kernel'
global_opt = 'csa'; % 'csa' or 'ds'

user_process={'FS-LSSVM', 'SV_L0_norm'};
window = [15,20,25];

[e,s,t] = fslssvm(X,Y,k,function_type,kernel_type,global_opt,user_process,window,testX,testY);