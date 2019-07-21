% Numerical Jacobian of func 
% idx specifies the index of argument w.r.t which the Jacobian is computed
% varargin encodes arguments passed to func

% For instance, for y = f(x1, x2, ..., xN)
% numerical_jac(@f, 2, x1, x2, ..., xN) computes the Jacobian df/dx2

function J = numerical_jac(func, idx, varargin)

step = 1e-6;

x = varargin{idx};
y = feval(func, varargin{:});
lenx = length(x);
leny = length(y);
J = zeros(leny, lenx);

for i=1:lenx
    xhi = x(i) + step;
    xlo = x(i) - step;

    varargin{idx}(i) = xhi;
    yhi = feval(func, varargin{:}); 
    varargin{idx}(i) = xlo;
    ylo = feval(func, varargin{:});    
    varargin{idx}(i) = x(i);
    
    J(:,i) = (yhi - ylo)/(xhi - xlo);  
end
end