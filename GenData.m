function [ A, b ] = GenData( m, n )

%m  number of examples
%n  number of features
rand('seed', 0);
randn('seed', 0);
x0 = sprandn(n,1,0.05);

randn('seed', 0);
A = randn(m,n);
A = A*spdiags(1./sqrt(sum(A.^2))',0,n,n); % normalize columns

randn('seed', 0);
v = sqrt(0.001)*randn(m,1);
b = A*x0 + v;

fprintf('solving instance with %d examples, %d variables\n', m, n);
fprintf('nnz(x0) = %d; signal-to-noise ratio: %.2f\n', nnz(x0), norm(A*x0)^2/norm(v)^2);

A=sparse(A);
%train_inst = A(1:40000,:);
%train_label = b(1:40000);
%test_inst = A(40001:50000,:);
%test_label= b(40001:50000);
%libsvmwrite('synthetic3.tr', b(1:40000), A(1:40000,:));
%libsvmwrite('synthetic3.t', b(40001:50000), A(40001:50000,:));
fprintf('Data generated over\n');
disp(size(b));

end

