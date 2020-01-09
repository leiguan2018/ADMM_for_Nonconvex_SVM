%------------------------------------------------------------
% admm solver for non-convex SVM
% Comparasion:
% 1: liblinear
% 2: scad-svm
% 3: GIST
% 2018/3/27
% run the program and save the execution results
%-----------------------------------------------------------
clear;
clc;
close all;
  
%file={'heart_scale','mushrooms.txt','real-sim','news20','rcv1.binary'};
%file={'rcv1.binary1'};
file={'heart_scale'};
for file_index=1:length(file)
    filename =file(file_index);
% load the .txt train data 
%filename = 'mushrooms.txt';
%filename = 'heart_scale';
%filename = 'news20.binary';
%filename = 'real-sim';
%filename = 'rcv1_train.binary';
pathTr =['..\dataset\',char(filename),'\',char(filename),'.tr'];
[train_label, train_instance]=libsvmread(pathTr);

% Convert the label to correct {-1, 1} class if it is required
fprintf('------------------------------------------------------\n\n');
fprintf('          Running of %s                    \n\n',char(filename));
fprintf('------------------------------------------------------\n\n');
 ConvertLable=1;
if file_index == 2
    ConvertLable=1;
end
if(ConvertLable==1)
    for i = 1:length(train_label),
        if(train_label(i,1)~=1)
            train_label(i,1) = -1;
        end
    end
end


% A:d*n  b:n*1
%A = train_instance';
%b = train_label;
run_times = 1;                          % run 10 times and do averaging

[n,d]=size(train_instance);

% ================================ parameters setting ============================
lambda =2^(-6); % fixed
theta=3.7;
regtype=3;

% =============================== Train of ADMM solver ==========================
rho1 = 1.5;
rho2 = 10;

fprintf('\n---Training the Data using admm \n');
verbose = 0;
time_list0 = zeros(run_times,1);
for i = 1:run_times
tic;
[x, bias, history, iter] = nonconvexSolver(train_instance, train_label, rho1, rho2, lambda,  theta, regtype, verbose);
 time_list0(i) = toc; 
end
disp(time_list0');
avertime = mean(time_list0);
fprintf('The average runtime of admm solver is:');
disp(avertime);
fprintf('The iteration number of admm solver is:');
disp(iter);
NZF = length(nonzeros(x))/length(x)*100;
fprintf('The selected numberis %d totoal number is %d and NZF of ADMM solver is:%.2f%%\n\n',length(nonzeros(x)),length(x),NZF);


%lambda =0.04; % fixed
%theta=3.7; %fixed
% ================================ scad svm train =================================
fprintf('\n---Training the Data using scad-svm\n');
lambda2 =2^(-5);
%lambda2=0.00195;
w2 = zeros(d,1);
bias2 =0;
time_list2 = zeros(run_times,1);
for i = 1:run_times
try
tic;
[temp,bias2,index, iter2] = scadsvc(train_instance,train_label,theta,lambda2);
time_list2(i) = toc; 
       
catch
 disp('********************************');
 disp(' SQA can not process');
 disp('********************************');
end

end

%disp(time_list2');
try
avertime = mean(time_list2);
fprintf('The average runtime of scadsvm solver is:');
disp(avertime);
fprintf('The iteration number of scadsvm solver is:');
disp(iter2);
if (length(index)>1)
      w2(index') = temp;
end
NZF2 = length(temp)/length(w2)*100;
fprintf('The selected numberis %d totoal number is %d and NZF of scadsvm solver is:%.2f%%\n\n',length(temp),length(w2),NZF2);
catch
 disp('********************************');
 disp(' SQA can not process too');
 disp('********************************');
end
% ================================ GIST train =================================
fprintf('\n---Training the Data using GIST\n');
% input parameters
lambda1 = 2^(-10);
%lambda1 =0.000977;
% theta = 1e-2*lambda*abs(randn);
theta = 3.7;

% optional parameter settings

regtype = 3; % nonconvex regularization type (default: 1 [capped L1]) 

%randn('seed',100);
%w0 = randn(d,1); % starting point (default: zero vector)
w0 = zeros(d,1);

stopcriterion = 0; % stopping criterion (default: 1)

maxiter = 1000; % number of maximum iteration (default: 1000)

tol = 1e-5; % stopping tolerance (default: 1e-5)

M = 5; % nonmonotone steps (default: 5)

t = 1; % initialization of t (default: 1)

tmin = 1e-20; % tmin parameter (default: 1e-20)

tmax = 1e20; % tmax parameter (default: 1e20)

sigma = 1e-5; % parameter in the line search (default: 1e-5)

eta = 2; % eta factor (default: 2)

stopnum = 3; % number of satisfying stopping criterion (default: 3)

maxinneriter = 20; % number of maximum inner iteration (line search) (default: 20)

% call the function
time_list3 = zeros(run_times,1);
for i = 1:run_times
tic;
 time_list3(i) = toc;
[w3,fun,time,iter] = gistL2SVM(train_instance,train_label,lambda1,theta,...
                              'maxiteration',maxiter,...
                              'regtype',regtype,...
                              'stopcriterion', stopcriterion,...
                              'tolerance',tol,...
                              'startingpoint',w0,...
                              'nonmonotone',M,...
                              'tinitialization',t,...
                              'tmin',tmin,...
                              'tmax',tmax,...
                              'sigma',sigma,...
                              'eta',eta,...
                              'stopnum',stopnum,...
                              'maxinneriter',maxinneriter);
 
 disp(time_list3(i));
end
disp(time_list3');
avertime = mean(time_list3);
fprintf('The average runtime of GIST is:');
disp(avertime);
fprintf('The iteration number of GIST is:');
disp(iter(end));
NZF3 = length(nonzeros(w3))/length(w3)*100;
fprintf('The selected numberis %d totoal number is %d and NZF of GIST solver is:%.2f%%\n\n',length(nonzeros(w3)),length(w3),NZF3);

% ================================= plot the result of admm ====================================

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


% ================================load the test data=========================================
%
% load the test data 
% rcv1_test.binary
fprintf('\n----------------------------------------------------\n');
%filename = 'rcv1_test.binary';
%pathT =['..\dataset\',filename];
%pathT =['..\dataset\',filename,'.t'];
pathT =['..\dataset\',char(filename),'\',char(filename),'.t'];
[test_label, test_instance]=libsvmread(pathT);
%test_instance = train_Instance(57848:72309,:);
%test_label = train_Label(57848:72309);
%convertLabel=1;

% ================================ predict for admm solver =========================================
temp_accuracy = test_accuracy(test_instance, test_label, x, bias,ConvertLable);
fprintf('The test accuracy rate of ADMM solver is:%.2f%%\n\n',temp_accuracy);



%==================================== predict for scadsvm ===========================================
temp_accuracy2 = test_accuracy(test_instance, test_label, w2, bias2,ConvertLable);
fprintf('The test accuracy rate of scadSVM solver is:%.2f%%\n\n',temp_accuracy2);

%==================================== predict for GIST ===========================================
bias3 = 0;
temp_accuracy3 = test_accuracy(test_instance, test_label, w3, bias3,ConvertLable);

fprintf('The test accuracy rate of GIST solver is:%.2f%%\n\n',temp_accuracy3);
end


clear test_label;
clear test_instance;

