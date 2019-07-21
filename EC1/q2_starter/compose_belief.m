% Compose belief vector as [state; square root of covariance matrix]
% Since the principal square root of the covariance matrix is symmetric, we only store the 
% lower (or upper) triangular portion of the square root to eliminate redundancy 

function b = compose_belief(x, SqrtSigma, model)
xDim = model.xDim;
bDim = model.bDim;

b = zeros(bDim,1);
b(1:xDim) = x;
idx = xDim+1;
for j = 1:xDim
    for i = j:xDim
		b(idx) = 0.5 * (SqrtSigma(i,j) + SqrtSigma(j,i));
		idx = idx+1;
    end
end
end