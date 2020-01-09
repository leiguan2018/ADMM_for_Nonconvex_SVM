function [ z, b, history, iter ] = nonconvexSolver( A, y, rho1, rho2, lambda, theta, regtype, verbose )
%tic;
%|---------------------------------------------------------------|
%      Solve the nonconvex SVM via ADMM
%
%     minimize   P(z) + \alpha*\sum_i \xsi_i
%     s.t.       x_i = z  
%                Y_i*£¨A_i*x_i + b) = 1 + xsi_i
%                xsi_i >= 0
%                s_i >= 0
%
% 
%          Author: Lei Guan
%          Update in 2016.12.10
%----------------------------------------------------------------


%Global constants and defaults
%QUIET    = 0;
MAX_ITER = 1000;
%ABSTOL   = 1e-4;
%RELTOL   = 1e-2;
%Data preprocessing
%relax_alpha=1.8;

[n, d] = size(A);
%disp('--------------------------------------------this');
%disp(size(A));
%disp(n)
%ADMM solver

%ALPHA = 1.4

min_s = 1e-20;
max_xsi = 1e10;
x = zeros(d,1);
b = 0;
%disp(x)
%randn('seed',100);
%z = randn(d,1); % starting point (default: zero vector)
z = zeros(d,1);
%z = zeros(d,1);
xsi = zeros(n,1);
s = zeros(n,1);
u1 = zeros(d,1);
u2 = zeros(n,1);
%s=zeros(n,1);
%u1=zeros(n,1);
%u2=zeros(d,1);
%vector_onen=ones(n,1);
%vector_oned=ones(d,1);
%u = zeros(n,1);

history.obj = zeros(MAX_ITER+1,1);
history.time = history.obj;

%if ~QUIET
  %  fprintf('%3s\t%10s\t%10s\t%10s\t%10s\t%10s\t%10s\t%10s\t%10s\t%10s\n', 'iter', ...
   %   'r norm1', 'eps pri1', 's norm1', 'eps dual1','r norm2', 'eps pri2', 's norm2', 'eps dual2', 'objective');
%end


% precompute static variables for x-update (projection on to Ax=b)
Y=sparse(1:n,1:n,y,n,n);
H=Y*A;

%HtH=H'*H;
yty=y'*y;
%disp(HtH);

[H_m, H_n] = size(H);
rho = rho1/rho2;
[L U] = factor(H, rho);
%toc;

history.objval(1) = 1./n*sum(max(0,1-Y*(A*x+b))) + funRegC(z,d,lambda,theta,regtype);
if verbose ==1
    fprintf('%3d\t %10.4f\n', 0, history.objval(1));
end

history.time(1)=0;
%tic;
for k = 1:MAX_ITER
    %tic;
    
     %w-update
    d2=rho*(z-u1)+ H'*(s+1-xsi-u2-b*y);
    if (H_m >= H_n)
        x = U \ (L \ d2);
    else
        x = d2/rho - (H'*(U \ ( L \ (H*d2) )))/rho^2;
    end
    
    %b-update
    
    b=y'*(s+1-xsi-u2-H*x)/yty;
   
   
   
  %disp('=============================');
 % disp(x);
    % z-update with relaxation   
    %zold = z;
    %x_hat = ALPHA*x + (1-ALPHA)*zold;
    z= proximalRegC(x + u1, d, lambda/rho1, theta, regtype);
    %disp('>>>>-------------------------');
   % disp(z);
 
    % xsi updation
    xsi=-1/(rho2*n)-H*x+1+s-u2-b*y;
    %disp('old xsi---------------------');
    %disp(xsi);
    xsi= min(max(0, xsi), max_xsi);
    
    
    %disp('xsi------------------------');
    %disp(xsi);
    %s updation
    %sold=s;
    %xsi_hat = ALPHA*xsi +(1-ALPHA)*(1-H*x+s);
   % disp('ss-------------------');
    s=H*x+xsi-1+u2+b*y;
    s= max(min_s, s);
   % disp(s);
    % dual variable updation
    u1=u1+(x-z);
   % disp('us 0ld-----');
   % disp(u2);
    %disp('H------');
    %disp('H1------');
    %disp(H(1,:));
    %disp(H);
    %disp('x-------------');
    %disp(x);
    %disp('H*x----------------');
    %disp(H*x);
    u2=u2+(H*x+xsi-1-s+b*y);
   
    %disp(u1);
    
    %disp('u2-------------------');
    %disp(u2);

    %history.time(k+1) = history.time(k) + toc;
    % diagnostics, reporting, termination checks
    history.objval(k+1)= 1./n*sum(max(0,1-Y*(A*x+b))) + funRegC(z,d,lambda,theta,regtype);
    if (verbose == 1)
        fprintf('%3d\t %10.4f\n', k, history.objval(k+1));
    end
    
     relative_change = (history.objval(k+1)-history.objval(k))/history.objval(k);
    %disp(relative_change);
    if (abs(relative_change)<1e-4)
        break;
    end

end
%toc;
%if ~QUIET
  %  toc(t_start);
%end
history.objval = history.objval(1: min(MAX_ITER,k)+1);
history.time = history.time(1: min(MAX_ITER,k)+1);

iter = k;
%disp(b);
end


function [L U] = factor(A, rho)
    [m, n] = size(A);
    if ( m >= n )    % if skinny
       L = chol( A'*A + rho*speye(n), 'lower' );
    else            % if fat
       L = chol( speye(m) + 1/rho*(A*A'), 'lower' );
    end

    % force matlab to recognize the upper / lower triangular structure
    L = sparse(L);
    U = sparse(L');
end






