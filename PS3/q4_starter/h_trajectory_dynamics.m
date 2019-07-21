function [h, jach] = h_trajectory_dynamics(xu_trajectory, cfg)

T = cfg.T; nX = cfg.nX; nU = cfg.nU; dt = cfg.dt;

x_traj = reshape(xu_trajectory(1:T*nX),nX, T);
u_traj = reshape(xu_trajectory(T*nX+1:end), nU, T);

f = cfg.f; %dynamics model x_t_plus_1 = f(x_t, u_t, dt);

%% YOUR CODE HERE

%% NOTE: Most equality constraints depend on only a subset of the
%% variables; this makes "blind" numerical computation of the jacobian by
%% the SQP solver very inefficient -- as it'd compute a ton of entries that
%% are known to be zero (but still do all the work for it).  Hence this
%% function also provides the SQP solver with the jacobian of h.  Note that
%% in our experience it is fast enough to compute the non-zero entries in
%% jach through numerical differentation.

