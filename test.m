clear;clc;
close all;
filename = 'heart_scale';
%filename = 'rcv1_train.binary';
path =['.\',filename,'.tr'];
%pathTr =['..\dataset\',filename];
%disp('hell0');
%pause();
%pathTr =['..\dataset\',filename,'.tr'];
[train_label, train_instance]=libsvmread(path);
A = train_instance';
b = train_label;
%[train_Label, train_Instance]=libsvmread(pathTr);
%train_instance = train_Instance(1:57847,:);
%train_label = train_Label(1:57847);
% Convert the label to correct {-1, 1} class if it is required
ConvertLable=1;
if(ConvertLable==1)
    for i = 1:length(train_label),
        if(train_label(i,1)~=1)
            train_label(i,1) = -1;
        end
    end
end
disp('------------------------');
lambda =2^(-6); % fixed
theta=3.7;
regtype=3;

% =============================== Train of ADMM solver ==========================
rho1 = 0.01;
rho2 = 0.01;

fprintf('\n---Training the Data using admm \n');
verbose = 1;

disp('------------------------');
tic;
%[x, bias, history, iter] = nonconvexSolver1(A, b, rho1, rho2, lambda,  theta, regtype);
[x, bias, history, iter] = nonconvexSolver(train_instance, b, rho1, rho2, lambda,  theta, regtype, verbose);
toc;
%[x, history, iter] = nonconvexSolver( A, b, alpha, rho1, lambda, theta, regtype);
disp(x);
disp(bias);
 



%display(x);

%K = length(history.objval);


%% ================================= plot the result ====================================

figure;
subplot(2,1,1);
plot(1:iter+1, history.objval(1:iter+1), 'r-', 'MarkerSize', 10, 'LineWidth', 2)
%plot(1:K, history.objval, 'k', 'MarkerSize', 10, 'LineWidth', 2);
ylabel('obj'); xlabel('iter (k)');

subplot(2,1,2);
plot(history.time(1:iter+1), history.objval(1:iter+1), 'r-', 'MarkerSize', 10, 'LineWidth', 2)
ylabel('Objective function value (logscaled)'); xlabel('CPU time (seconds)');
%g = figure;
%subplot(2,1,1);
%semilogy(1:K, max(1e-8, history.r_norm), 'k', ...
   % 1:K, history.eps_pri, 'k--',  'LineWidth', 2);
%ylabel('||r||_2');

%subplot(2,1,2);
%semilogy(1:K, max(1e-8, history.s_norm), 'k', ...
    %1:K, history.eps_dual, 'k--', 'LineWidth', 2);
%ylabel('||s||_2'); xlabel('iter (k)');

clear train_label;
clear train_instance;


