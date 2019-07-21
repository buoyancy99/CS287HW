% Simulate execution of controls Uopt along locally optimal trajectory
% starting from initial belief state b1
function simulate_bsp_trajectory(b1, Uopt, model)

% Use legacy random number generation to be compatible across MATLAB versions
%randn('state',123456);

dynamics_func = model.dynamics_func;
obs_func = model.obs_func;
T = model.T;

% True state (hidden; used for simulating observations)
[x0, SqrtSigma0] = decompose_belief(b1, model);
x_true = x0 + SqrtSigma0*randn(model.xDim,1);

B = zeros(model.bDim, T);
B(:,1) = b1;

for t=1:model.T-1
    x_true = dynamics_func(x_true, Uopt(:,t), chol(model.Q)'*randn(model.qDim,1), model);
    z_tp1 = obs_func(x_true, chol(model.R)'*randn(model.rDim,1), model);
    
    B(:,t+1) = belief_dynamics(B(:,t), Uopt(:,t), z_tp1, model);
end

hold on;
plot_belief_trajectory(B, Uopt, model);

end
