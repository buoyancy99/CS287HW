% Belief dynamics: Given belief and control at time t, compute the belief
% at time (t+1) using EKF

function b_tp1 = belief_dynamics(b_t, u_t, z_tp1, model)

dynamics_func = model.dynamics_func;
obs_func = model.obs_func;
qDim = model.qDim;
rDim = model.rDim;

[x_t, SqrtSigma_t] = decompose_belief(b_t, model);
Sigma_t = SqrtSigma_t*SqrtSigma_t;

if isempty(z_tp1)
    % Maximum likelihood observation assumption
    z_tp1 = obs_func(dynamics_func(x_t, u_t, zeros(qDim,1), model), zeros(rDim,1), model);
end

[x_tp1, Sigma_tp1] = ekf(x_t, Sigma_t, u_t, z_tp1, model);

% Compute square root for storage 
% Several different choices available -- we use the principal square root
[V,D] = eigs(Sigma_tp1);
SqrtSigma_tp1 = V*sqrt(D)*V';

b_tp1 = compose_belief(x_tp1, SqrtSigma_tp1, model);

end