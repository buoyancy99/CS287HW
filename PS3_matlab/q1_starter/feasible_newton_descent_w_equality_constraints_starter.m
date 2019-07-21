function [x_min, x_iters, f_iters, stoppingval_iters] = feasible_newton_descent_w_equality_constraints(f, C, d, x0, finite_diff_eps, stopping_eps, alpha, beta)

% min_x f(x)
% s.t.  Cx = d
%
% newton descent with backtracking line search

% returned values:
% x_min: the minimum found
% x_iters: x values found in each iteration
% f_iters: f values found in each iteration
% stoppingval_iters: delta_x'*H*delta_x from every iteration

