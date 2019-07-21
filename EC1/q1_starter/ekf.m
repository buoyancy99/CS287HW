%% EKF step

function [x_tp1, Sigma_tp1] = ekf(x_t, Sigma_t, u_t, z_tp1, model)

% Extract function handles and useful definitions from model
dynamics_func = model.dynamics_func;
obs_func = model.obs_func;
xDim = model.xDim;
qDim = model.qDim;
rDim = model.rDim;
Q = model.Q;
R = model.R;

%% YOUR_CODE_HERE

%% Implement the EKF update equations here as specified in the problem set in Q 1

end