function [ accuracy ] = test_accuracy( test_inst, test_label, w, bias, ConvertLable )

%ConvertLable=1;
if(ConvertLable==1)
    for i = 1:length(test_label),
        if(test_label(i,1)~=1)
            test_label(i,1) = -1;
        end
    end
end

%disp(size(test_inst));
%disp(size(w));
%disp(size(test_label));
[n,d]=size(test_inst);
try
if d~=length(w)
    w1=w(1:d);
else
    w1=w;
end
result = (test_inst * w1 + bias).*test_label;
catch
    disp('********************************');
    disp(' The matrix and vecor did not match');
    disp('********************************');
end
right_num = sum(result>0);
total_num = length(test_label);
accuracy = right_num/total_num*100;
fprintf('(%d/%d)\n',right_num,total_num);
%disp(accuracy);
end

