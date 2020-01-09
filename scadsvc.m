function [w, b, xind, iter] = scadsvc(xtrain, ytrain, a, lambda, tol)
% this implements SCAD SVM classification
% Input:
%   xtrain : n-by-d data matrix to train
%   ytrain : column vector of target {-1, 1}'s
%   a : tuning parameter in scad function (default: 3.7 or whatever the paper uses)
%   lambda : tuning parameter in scad function (default : 2 or whatever the paper uses)
%   tol: the cut-off value to be taken as 0
% Output:
%   w : direction vector of separating hyperplane
%   b : the bias
%   Ind : Indices of remained variables
% You have to have SVM software of OSU to run this program

if nargin < 2;
    disp('not enough input');
elseif nargin == 2;
    a = 3.7; lambda = 1; tol= 10^(-6);
elseif nargin == 3;
    lambda = 2; tol= 10^(-4);
elseif nargin == 4;
    tol= 10^(-4);
end;

%[SVs, Bias, Parameters, nSV, nLabel] = LinearSVC(xtrain', ytrain');
%[AlphaY, SVs, Bias, Parameters, nSV, nLabel] = LinearSVC(xtrain', ytrain');
d = size(xtrain,2);    
%w = sum(vec2matSM(AlphaY,d) .* SVs,2) ;
%b = Bias;
%w = zeros(d,1);
w = 1e-3*ones(d,1);
%w = randn(d,1);
b = 0;
diff = 1000;
ntrain = size(xtrain,1);
xind = 1:d;
MAX_ITER = 1000;
iter = 0;
while diff > tol;
    x = [ones(ntrain,1) xtrain];
    y1 = ytrain;
    y0 = y1./abs(y1 - x * [b ; w]);
    sgnres = y1 -x*[b;w];
    res = abs(y1 - x * [b ; w]);
    D = 1/(2*ntrain)*diag(1./res) ;
    aw = abs(w);
    dp = zeros(size(xtrain, 2), 1);
    dp = lambda*(aw<=lambda)+(a*lambda-aw)/(a-1).*(aw>lambda&aw<=a*lambda);
    Q1 = diag([0; dp./aw]);
    Q = x'* D * x + Q1;
    P = 0.5*(y1 + y0)' * x /ntrain;
    nwb = pinv(Q) * P';
    nw = nwb(2:end);
    nb = nwb(1);
    diff = norm(nwb - [b;w]);
    ind = abs(nw)>0.001;
    if (sum(ind)>0) 
      w = nw(ind);
      xtrain = xtrain(:,ind);
      xind = xind(ind);
      b = nb;
    else 
      diff=tol/2;
      xind = 0;
    end 
    iter = iter +1;
    if(iter>999)
        break;
    end
end;
    ind = abs(nw)>0.001;
    if (sum(ind)>0) 
      w = nw(ind);
    else
      w = zeros(size(xtrain,2),1);
    end 
    b = nb;
    f = xtrain*w+b;
    xqx = 0.5*x*pinv(Q)*x';
