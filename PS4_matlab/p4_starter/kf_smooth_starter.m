function [xfilt,varargout] = kf_smooth_starter(y, A, B, C, d, u, Q, R, init_x, init_V)
%
% function [xfilt, xpred, Vfilt, loglik, xsmooth, Vsmooth, Q, R] = 
%           kf_smooth(y, A, B, C, d, u, Q, R, init_x, init_V)
%
%
% Kalman filter
% [xfilt, xpred, Vfilt] = ekf_smooth(y_all, A, B, C, d, Q, R, init_x, init_V);
%
% Kalman filter with Smoother
% [xfilt, xpred, Vfilt, loglik, xsmooth, Vsmooth] = ekf_smooth(y_all, A, B, C, d, Q, R, init_x, init_V);
%
% Kalman filter with Smoother and EM algorithm
% [xfilt, xpred, Vfilt, loglik, xsmooth, Vsmooth, Q, R] = ekf_smooth(y_all, A, B, C, d, Q, R, init_x, init_V);
%
%
% INPUTS:
% y - observations
% A, B, C, d:  x(:,t+1) = A x(:,t) + B u(:,t) + w(:,t) 
%              y(:,t)   = C x(:,t) + d        + v(:,t)
% Q - covariance matrix of system x(t+1)=A*x(t)+w(t) , w(t)~N(0,Q)
% R - covariance matrix of output y(t)=C*x(t)+v(t) , v(t)~N(0,R)
% init_x -
% init_V -
%
%
% OUTPUTS:
% xfilt = E[X_t|t]
% varargout(1) = xpred - the filtered values at time t before measurement
% at time t has been accounted for
% varargout(2) = Vfilt - Cov[X_t|0:t]
% varargout(3) = loglik - loglikelihood
% varargout(4) = xsmooth - E[X_t|0:T]
% varargout(5) = Vsmooth - Cov[X_t|0:T]
% varargout(6) = Q - estimated system covariance according to 1 M step (of EM)
% varargout(7) = R - estimated output covariance according to 1 M step (of EM)


n_var_out = max(nargout,1)-1; % number of variable number of outputs

T = size(y,2);
ss = size(Q,1); % size of state space

%% Forward pass (Filter)

%YOUR code here

if(n_var_out >= 1), varargout(1) = {xpred}; end
if(n_var_out >= 2), varargout(2) = {Vfilt}; end
if(n_var_out >= 3), varargout(3) = {loglik}; end


%% Backward pass (RTS Smoother and EM algorithm)
if(n_var_out >= 4)

	%YOUR code here
	
	varargout(4) = {xsmooth};
   if(n_var_out >= 5), varargout(5) = {Vsmooth}; end
   if(n_var_out >= 6), varargout(6) = {Q}; end
   if(n_var_out == 7), varargout(7) = {R}; end
end





















