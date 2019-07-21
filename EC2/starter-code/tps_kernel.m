function K = tps_kernel(X)
  [d,n] = size(X);
  K = zeros(n);
  for i = 1:n
    for j = 1:n
      r = norm(X(:,i) - X(:,j));
      if r == 0
	K(i,j) = 0;
      else
	K(i,j) = (r^2)*log(r);
      end
    end
  end
end
