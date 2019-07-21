function [x_target, u_target] = find_target_trajectory_through_SCP(f, dt, x_tentative, u_tentative, x_init, u_min, u_max);

nX = length(x_init);
nU = length(u_min);
T = size(x_tentative,2);
nXUT = (nX+nU)*T;


user_cfg = struct();
user_cfg.min_approx_improve = 1e-1;
user_cfg.min_trust_box_size = 1e-3;
user_cfg.full_hessian = false;
user_cfg.h_use_numerical = false; %%for speed, you'll want to be provide an implementation of the gradient computation for h 
user_cfg.initial_trust_box_size = .1;
user_cfg.max_merit_coeff_increases = 3;
user_cfg.initial_penalty_coeff = 1000;

traj_dynamics_cfg = struct();
traj_dynamics_cfg.nX = nX;
traj_dynamics_cfg.nU = nU;
traj_dynamics_cfg.T = T;
traj_dynamics_cfg.f = f;
traj_dynamics_cfg.dt = dt;

%% YOUR CODE HERE

q = % YOUR CODE 
Q = % YOUR CODE

x0 = % YOUR CODE


f0 = @(x) 0;
A_ineq = % YOUR CODE
b_ineq = % YOUR CODE
A_eq = % YOUR CODE
b_eq = % YOUR CODE
g = @(x) -1e5;

h = @(x) h_trajectory_dynamics(x, traj_dynamics_cfg); %YOURS TO IMPLEMENT


[xu_trajectory, success] = penalty_sqp(x0, Q, q, f0, A_ineq, b_ineq, A_eq, b_eq, g, h, user_cfg);

% assuming your xu_trajectory has first all states and then all control
% inputs, code below will get back out x_target and u_target in desired
% format
x_target = reshape(xu_trajectory(1:T*nX),nX, T);
u_target = reshape(xu_trajectory(T*nX+1:end), nU, T);
