clear;clc;
close all;


%% load the .txt train data 

filename = 'mushrooms.txt';
%pathTr =['..\dataset\',filename,'.tr'];
%filename='rcv1_train.binary';
pathTr =['..\dataset\',filename];
[label, instance]=libsvmread(pathTr);
%disp(length(label));
%pause();
train_label = label(1:6499);
train_instance = instance(1:6499,:);

%[train_label, train_instance]=libsvmread(pathTr);
%disp(size(train_label))
%disp(size(train_instance))
% Convert the label to correct {-1, 1} class if it is required
ConvertLable=1;
if(ConvertLable==1)
    for i = 1:length(train_label),
        if(train_label(i,1)~=1)
            train_label(i,1) = -1;
        end
    end
end
%X=instance;
%y=label; 
% A:d*n  b:n*1
A = train_instance';
b = train_label;

[n,d]=size(train_instance);


%% =============================== Train of ADMM solver=================================


lambda =0.02; % fixed
theta=3.7;
regtype=3;


rho1 = 1.6;
rho2 = 1.6;


 %|------------------ using the selected parameters to train the data -----------------------------|
 verbose=0;
fprintf('\nTraining the Data ...\n');
tic;
[x, bias, history, iter] = nonconvexSolver(A, b, rho1, rho2, lambda,  theta, regtype,verbose);
toc;
%disp(history.time(iter+1));
%disp(bias);

NZF = length(nonzeros(x))/length(x)*100;
fprintf('The selected numberis %d totoal number is %d and NZF of ADMM solver is:%.2f%%\n',length(nonzeros(x)),length(x),NZF);

% ================================= plot the result ====================================
PLOT = 0;
if (PLOT == 1)
    figure;
    subplot(2,1,1);
    plot(1:iter+1, history.objval(1:iter+1), 'r-', 'MarkerSize', 10, 'LineWidth', 2)
    %plot(1:K, history.objval, 'k', 'MarkerSize', 10, 'LineWidth', 2);
    ylabel('obj'); xlabel('iter (k)');
    
    subplot(2,1,2);
    plot(history.time(1:iter+1), history.objval(1:iter+1), 'r-', 'MarkerSize', 10, 'LineWidth', 2);
    ylabel('Objective function value (logscaled)'); xlabel('CPU time (seconds)');
end

clear train_label;
clear train_instance;


%% ================================load the test data=========================================
% load the test data 
% rcv1_test.binary
%pathT =['..\dataset\',filename,'.t'];
%filename='rcv1_test.binary';
%pathT =['..\dataset\',filename];
%[test_label, test_instance]=libsvmread(pathT);
clear train_label;clear train_instance;
test_label = label(6500:8124);
test_instance = instance(6500:8124,:);
convertLabel=1;

%==================================== predict for scadsvm =======================================================
%temp_accuracy0 = test_accuracy(test_instance, test_label, w, b,convertLabel);

%fprintf('The test accuracy rate of ADMM solver is:%.2f%%\n\n',temp_accuracy0);

%==================================== predict for admm =======================================================
temp_accuracy = test_accuracy(test_instance, test_label, x, bias,convertLabel);

fprintf('The test accuracy rate of ADMM solver is:%.2f%%\n\n',temp_accuracy);



clear test_label;
clear test_instance;

