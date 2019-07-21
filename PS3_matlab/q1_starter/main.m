% pabbeel@cs.berkeley.edu
%
% main.m
%
% In this question we write our own interior point, feasible step Newton
% method.
%
% results.mat contains some of my results for you to compare with




% part (a): let's make sure all our numerical derivatives are correctly
% implemented
%
%

finite_diff_eps = 1e-4;
x0 = [   0.418649467727506; 0.846221417824324; 0.525152496305172];

g1 = gradient(@f_linear, x0, finite_diff_eps); % YOURS to implement
% check your result by comparing with the exact result, which is easily found by inspecting f_linear.
% my result is upto accuracy 1e-12 equal to the exact gradient


g2 = gradient(@f_quadratic, x0, finite_diff_eps); 
% check your result by comparing with the exact result, which is easily found by inspecting f_quadratic.
% my result is upto accuracy 1e-11 equal to the exact gradient

H2 = hessian(@f_quadratic, x0, finite_diff_eps);  % YOURS to implement



% REPORT the gradient, the Hessian, and the condition number (the ratio of
% the highest eigenvalue over the lowest eigenvalue) at x_test

x_test = [   0.950129285147175;   0.231138513574288;   0.606842583541787;   0.485982468709300;   0.891298966148902];

g_test = gradient(@f_test_exponential, x_test, finite_diff_eps);
H_test = hessian(@f_test_exponential, x_test, finite_diff_eps);

eigs = eig(H_test);
condition_number = max(eigs)/min(eigs);



% part (b): let's start with unconstrained cvx optimization:
% min_x  f(x)  (for f convex)


stopping_eps = 1e-6; % gradient descent: norm(g,2) < stopping_eps; newton descent: delta_x'*H*delta_x < stopping_eps
alpha =  0.3; % in (0, 0.5)  -- backtracking line search parameter
beta = 0.9; % in (0,1)       -- backtracking line search parameter

[x_q_g_min, x_q_g_iters, f_q_g_iters, stoppingvals_q_g_iters] = gradient_descent(@f_quadratic, x0, finite_diff_eps, stopping_eps, alpha, beta); % YOURS to implement
figure; plot(f_q_g_iters);


[x_q_newton_min, x_q_newton_iters, f_q_newton_iters, stoppingvals_q_newton_iters] = newton_descent( @f_quadratic, x0, finite_diff_eps, stopping_eps, alpha, beta); % YOURS to implement
figure; plot(f_q_newton_iters);
  
% check your results by comparing against mine



% REPORT plot of f_test_g_iters, f_test_newton_iters, the x value
% achieving the minimum of f_test

x_test0 = [   0.950129285147175;   0.231138513574288;   0.606842583541787;   0.485982468709300;   0.891298966148902];

% stopping_eps * 1000 for gradient descent b/c gradient descent tends to be really slow at
% getting to the minimum

[x_test_g_min, x_test_g_iters, f_test_g_iters, stoppingvals_test_g_iters] = gradient_descent(@f_test_exponential, x_test0, finite_diff_eps, stopping_eps*1000, alpha, beta); 
figure; plot(f_test_g_iters);

[x_test_newton_min, x_test_newton_iters, f_test_newton_iters, stoppingvals_test_newton_iters] = newton_descent( @f_test_exponential, x_test0, finite_diff_eps, stopping_eps, alpha, beta);
figure; plot(f_test_newton_iters);




% part (c): let's add equality constraints:
% min_x f(x)   (for f convex)
%  s.t. Cx = d

C = [ 0.189653747547175   0.682223223591384   0.541673853898088];

d = [ 0.697898481859863];

x0_feasible = C\d;
  
[x_eq_min, x_eq_iters, f_eq_iters, stoppingvals_eq_iters] = feasible_newton_descent_w_equality_constraints(@f_quadratic, C, d, x0_feasible, finite_diff_eps, stopping_eps, alpha, beta); % YOURS to implement
  
% check your results by comparing against mine



C_test = [    0.9501    0.6068    0.8913];

d_test = [    0.6154];


x0_feasible = C_test \ d_test;

[x_test_eq_min, x_test_eq_iters, f_test_eq_iters, stoppingvals_test_eq_iters] = feasible_newton_descent_w_equality_constraints(@f_test, C_test, d_test, x0_feasible, finite_diff_eps, stopping_eps, alpha, beta);

plot(f_test_eq_iters)



C_test_exp = [  ...
	-0.5883    0.1139   -0.0956   -1.3362   -0.6918
   -0.1364    0.0593    0.2944    1.6236    1.2540];

d_test_exp = [ ...
	   -1.5937
	    0.5711];

x0_feasible_test_exp = C_test_exp \ d_test_exp;


[x_test_exponential_eq_min, x_test_exponential_eq_iters, f_test_exponential_eq_iters, stoppingvals_test_exponential_eq_iters] = ...
	feasible_newton_descent_w_equality_constraints(@f_test_exponential, C_test_exp, d_test_exp, x0_feasible_test_exp, finite_diff_eps, stopping_eps, alpha, beta);

plot(f_test_exponential_eq_iters)


% REPORT: plots of f_test_eq_iters, and
% f_test_exponential_iters




% part (d): let's add inequality constraints
% min_x f(x) 
%  s.t. Cx = d
%       f_i(x) <= 0



fi{1} = @(x) (exp(x(1)*3) + exp(x(2)*2) - 6);
x_low = -0.8*ones(3,1); %[-0.5; -0.5; -0.5]; % we could also encode x_low and x_high with additional entries in f_i, but I decided to treat them separately as they are easy to directly incorporate and a bit easier to type out this way
x_high = 0.8*ones(3,1); %[0.5; 0.5; 0.5];




x0_feasible = C\d;



% we need to first find a feasible point to initialize our actual problem
% with:


[x_q_eq_and_ineq_init_min, feasibility] = init_feasible_newton_descent_w_eq_and_ineq(fi, x_low, x_high, C, d, x0_feasible, finite_diff_eps, stopping_eps, alpha, beta);% YOURS to implement


% now let's solve our actual problem:


[x_q_eq_and_ineq_min, x_q_eq_and_ineq_iters, f_q_eq_and_ineq_iters] = feasible_newton_descent_w_equality_and_inequality_constraints(@f_quadratic,...
	fi, x_low, x_high, C, d, x_q_eq_and_ineq_init_min, finite_diff_eps, stopping_eps, alpha, beta);% YOURS to implement


% compare with my solution


% REPORT on the one below:

fi{1} = @(x) (exp(x(1)*3) + exp(x(2)*2) - 6);
x_low = -0.7*ones(3,1); % we could also encode x_low and x_high with additional entries in f_i, but I decided to treat them separately as they are easy to directly incorporate and a bit easier to type out this way
x_high = 0.7*ones(3,1); 




x0_feasible = C\d;



% we need to first find a feasible point to initialize our actual problem
% with:


[x_test_q_eq_and_ineq_init_min, feasibility] = init_feasible_newton_descent_w_eq_and_ineq(fi, x_low, x_high, C, d, x0_feasible, finite_diff_eps, stopping_eps, alpha, beta);



% now let's solve our actual problem:


[x_test_q_eq_and_ineq_min, x_test_q_eq_and_ineq_iters, f_test_q_eq_and_ineq_iters] = feasible_newton_descent_w_equality_and_inequality_constraints(@f_quadratic,...
	fi, x_low, x_high, C, d, x_test_q_eq_and_ineq_init_min, finite_diff_eps, stopping_eps, alpha, beta);


% REPORT: the min and the argmin of this constrained optimization problem









%% saving my results:

%save q2_results.mat f_q_g_iters x_q_g_iters f_q_newton_iters x_q_newton_iters f_eq_iters x_eq_iters x_q_eq_and_ineq_init_min  x_q_eq_and_ineq_min
% 





% Congratulations!!  You just finalized your own generic solver for convex optimization problems!!







