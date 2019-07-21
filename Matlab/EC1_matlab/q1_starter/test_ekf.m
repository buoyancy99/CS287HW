% Test script for extended Kalman filtering (EKF)

function test_ekf()

close all;
clear all;

% Setup model
model = {};

% Setup model dimensions
model.xDim = 2; % state space dimension
model.uDim = 2; % control input dimension
model.qDim = 2; % dynamics noise dimension
model.zDim = 2; % observation dimension
model.rDim = 2; % observation noise dimension

model.Q = 2*eye(model.qDim); % dynamics noise variance
model.R = eye(model.rDim); % observation noise variance
model.R(2,2) = 10;

model.T = 50; % number of time steps in trajectory

model.dynamics_func = @dynamics_func; % function handle to dynamics function
model.obs_func = @obs_func; % function handle to observation function

x0 = [10;10];
Sigma0 = eye(model.xDim,model.xDim);

% Load true (hidden) states and simulated observations
[X, Z] = load_states_observations();

% Mean and covariances for plotting
mean_ekf = zeros(model.xDim, model.T);
cov_ekf = zeros(model.xDim, model.xDim, model.T);

mean_ekf(:,1) = x0;
cov_ekf(:,:,1) = Sigma0;

for t=1:model.T-1
    
    % EKF step
    [mean_ekf(:,t+1), cov_ekf(:,:,t+1)] = ekf(mean_ekf(:,t), cov_ekf(:,:,t), zeros(model.uDim,1), Z(:,t+1), model);
   
end
    
plot_1d_trajectory(mean_ekf, cov_ekf, X, model);

filename = 'ekf.png';
saveas(gcf, filename);

format longg;
fprintf('Mean at last timestep:\n');
disp(mean_ekf(:,model.T));

fprintf('Covariance matrix at last timestep:\n');
disp(cov_ekf(:,:,model.T));

fprintf('For verification (allow for numerical inaccuracy):\n');
fprintf('(Answer) Mean at last timestep:\n');
disp([7.42875441335378; 11.1154584205147]);

fprintf('(Answer) Covariance matrix at last timestep:\n');
disp([0.123066918908846 -0.0794197546935217; -0.0794197546935217 0.0924640118620255]);

end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Dynamics function: x_t+1 = dynamics_func(x_t, u_t, q_t, model)

function x_tp1 = dynamics_func(x_t, u_t, q_t, model)

x_tp1 = zeros(model.xDim,1);

x_tp1(1) = 0.1*(x_t(1)*x_t(1)) - 2*x_t(1) + 20 + q_t(1);
x_tp1(2) = x_t(1) + 0.3*x_t(2) - 3 + q_t(2)*3;

end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Observation function: z_t = obs_func(x_t, r_t, model)

function z_t = obs_func(x_t, r_t, model)

z_t = zeros(model.zDim,1);

z_t(1) = (x_t'*x_t) + sin(5*r_t(1));
z_t(2) = 3*(x_t(2)*x_t(2))/x_t(1) + r_t(2);

end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function plot_1d_trajectory(mean_ekf, cov_ekf, X, model)

figure('units','pixel','outerposition',  [0 0 1600 1200]);
hor = 1:model.T;
clf; 

% Iterate over dimensions
for d=1:model.xDim
    x_td = mean_ekf(d,:)';
    Sigma_td = squeeze(cov_ekf(d,d,:));
    
    ff1 = [hor';  flipdim(hor',1)];
    ff2 = [x_td + 3*sqrt(Sigma_td); flipdim(x_td - 3*sqrt(Sigma_td),1)];
    subplot(model.xDim,1,d);
    hold on;
    fill(ff1, ff2, [7 7 8]/8, 'EdgeColor', [7 7 8]/8);
    plot(hor, X(d,:), 'rs-','linewidth',3);
    plot(hor, x_td, 'b*-.','linewidth',1);
    plot(hor, x_td + 3*sqrt(Sigma_td), 'color', [0 4 0]/8);
    plot(hor, x_td - 3*sqrt(Sigma_td), 'color', [0 4 0]/8);
    
    grid on;
    set(gcf,'PaperSize', [10 5]);
    set(gcf,'PaperPosition',[0.1 0.1 10 5]);
    xlabel('time steps');
    ylabel('state');
    axis([1 model.T 5 18]);
    legend(strcat('EKF dimension ',num2str(d)),'ground truth','location','southwest');
end
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [X, Z] = load_states_observations()

X = load('X.mat', '-ascii');
Z = load('Z.mat', '-ascii');

% % Default random seed
% rng('default');
% 
% X = zeros(model.xDim, model.T); % true states (not known)
% Z = zeros(model.zDim, model.T); % observations received
% 
% X(:,1) = x0 + chol(Sigma0)'*randn(model.xDim,1);
% Z(:,1) = obs_func(X(:,1), chol(model.R)'*randn(model.rDim,1), model);
% 
% for t=1:model.T-1
%   X(:,t+1) = dynamics_func(X(:,t), zeros(model.uDim,1), chol(model.Q)'*randn(model.qDim,1), model);
%   Z(:,t+1) = obs_func(X(:,t+1), chol(model.R)'*randn(model.rDim,1), model);
% end
% save('X.mat','-ascii','-double', 'X');
% save('Z.mat','-ascii','-double', 'Z');

end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%