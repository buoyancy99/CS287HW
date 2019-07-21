% Decompose belief vector into mean and square root of covariance

function [x, SqrtSigma] = decompose_belief(b, model)

xDim = model.xDim;

x = b(1:xDim);
idx = xDim+1;
if (isa(b, 'double'))
    SqrtSigma = zeros(xDim, xDim);
end
for j = 1:xDim
    for i = j:xDim
        SqrtSigma(i,j) = b(idx);
        SqrtSigma(j,i) = SqrtSigma(i,j);
        idx = idx+1;
    end
end
end
