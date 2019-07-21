%% Copyright Pieter Abbeel
%%
%% How I generated data:

% 
% T = 500;
% 
% 
% A = [...	
%     0.9040   -0.0342   -0.0095    0.0318    0.0485
%    -0.0201    0.9356   -0.0014    0.0428    0.0054
%    -0.0931   -0.0525    0.9220    0.0215    0.1242
%    -0.0245   -0.0350    0.0249    1.0414   -0.0451
%    -0.0775   -0.0462   -0.0194    0.0386    1.0371
% ];
% 
% 
% B = [ ...
% 	   1.0347    0.8884
%     0.7269   -1.1471
%    -0.3034   -1.0689
%     0.2939   -0.8095
%    -0.7873   -2.9443
% ];
% 
% 
% 
% C = [ ...
%     0.0859   -0.7423    2.3505    0.7481    0.8886
%    -1.4916   -1.0616   -0.6156   -0.1924   -0.7648
% ];
% 
% 
% d = [ -2.1384   -0.8396]';
% 
% 
% x(:,1) = [   -0.6003    0.4900    0.7394    1.7119   -0.1941 ]';
% 
% 
% Q = % not revealing
% R = % not revealing
% 
% nX = size(A,1); nZ = size(C,1);
% nU = size(B,2);
% 
% w = randn(nX,T); w = sqrtm(Sigma_w)*w;
% v = randn(nZ,T); v = sqrtm(Sigma_v)*v;
% 
% u = randn(nU, T);
% 
% y(:,1) = C*x(:,1) + d + v(:,1);
% 
% for t=1:T-1
% 	x(:,t+1) = A * x(:,t) + B * u(:,t) + w(:,t);
% 	y(:,t+1) = C*x(:,t+1) + d + v(:,t+1);
% end
% y(:,T) = C*x(:,T) + v(:,T);
% 
% 
% save p3_a_data.mat T A B C d u y x

clear; clc; close all;
load p3_a_data.mat


x_init = zeros(5,1); % mean at time t=1 before measurement at time t=1
P_init = eye(5); % covariance at time t=1 before measurement at time t=1


% I found initially overestimating Q and R gives better learning of Q and R
% during EM

Q = 10*eye(5); R = 10*eye(2);


for i=1:50 % 50 is an arbitrary number of EM iterations, but is reasonable for this example
	[xfilt, xpred, Vfilt, loglik, xsmooth, Vsmooth, Q, R] = kf_smooth(y, A, B, C, d, u, Q, R, x_init, P_init); % YOURS to implement
	% I saved my results for i=1 into this file, this could help you debug:
	%if(i==1)
	%	save p3_a_solution__1kf_smooth_run__xfilt_xsmooth_Vfilt_Vsmooth_loglik.mat xfilt xpred Vfilt loglik xsmooth Vsmooth loglik Q R
	%end
	ll(i) = loglik;
end


figure; plot(ll);

figure; plot(x');
hold on; 
plot(xfilt', '-.');
plot(xsmooth', '--');



