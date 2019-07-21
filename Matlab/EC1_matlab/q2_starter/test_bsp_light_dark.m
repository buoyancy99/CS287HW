% Test suite for belief space planning for the light dark example (Platt et al. RSS 2010)

function test_bsp_light_dark()

close all;
clear all;

addpath ../q1_starter  %TODO: your path might be different depending on where you solved q1

% Setup model
model = {};

% Setup model dimensions
model.xDim = 2; % state space dimension
model.uDim = 2; % control input dimension
model.qDim = 2; % dynamics noise dimension
model.zDim = 2; % observation dimension
model.rDim = 2; % observation noise dimension

model.bDim = model.xDim + (model.xDim*(model.xDim+1)/2); % belief space dimension
% note that we only store the lower (or upper) triangular portion of the
% covariance matrix to eliminate redundancy

model.dT = 1; % time step for dynamics function

model.T = 15; % number of time steps in trajectory

model.alpha_belief = 10; % weighting factor for penalizing uncertainty at intermediate time steps
model.alpha_final_belief = 10; % weighting factor for penalizing uncertainty at final time step
model.alpha_control = 1; % weighting factor for penalizing control cost

model.xMin = [-5; -3]; % minimum limits on state (xMin <= x)
model.xMax = [5; 3]; % maximum limits on state (x <= xMax)
model.uMin = [-1; -1]; % minimum limits on control (uMin <= u)
model.uMax = [1; 1]; % maximum limits on control (u <= uMax)

model.Q = eye(model.qDim); % dynamics noise variance
model.R = eye(model.rDim); % observation noise variance

model.dynamics_func = @dynamics_func; % function handle to dynamics function
model.obs_func = @obs_func; % function handle to observation function
model.plot_domain = @plot_domain; % function handle to plotting experiment specific stuff
model.setup_sqp_params = @setup_sqp_params; 

X1 = [-3.5  2; -3.5 -2; -4 0; 2  2; -4  2]';
G =  [-3.5 -2; -3.5  2; -1 0; 2 -2; -1 -2]';

% [Reference] final belief trajectory costs for verification 
% Allow for numerical inaccuracy
verify_cost = [45.181701; 45.181643; 49.430339; 27.687003; 56.720314];

for i_problem = 1:5
    % Setup initial conditions for problem
    x1 = X1(:,i_problem); % start (mean of initial belief)
    SqrtSigma1 = eye(2); % initial covariance
    goal = G(:,i_problem); % goal
    
    model.start = x1;
    model.goal = goal;

    % setup initial control vector -- straight line initialization from start to goal
    U = repmat(((model.goal - model.start)/(model.T-1)), 1, model.T-1);
    
    B = zeros(model.bDim,model.T);
    B(:,1) = compose_belief(x1, SqrtSigma1, model);
    for t=1:model.T-1
        B(:,t+1) = belief_dynamics(B(:,t), U(:,t), [], model);
    end
    clf;
    plot_belief_trajectory(B, U, model); % display initialization 
    pause(0.1);
    
    [Bopt, Uopt, ~] = belief_opt_penalty_sqp(B, U, model);
    plot_belief_trajectory(Bopt, Uopt, model);
    
    cost = compute_forward_simulated_cost(compose_belief(x1, SqrtSigma1, model), Uopt, model);
    fprintf('Total cost of optimized trajectory: %f\n', cost);
    
    % save trajectory to png file 
    saveas(gcf, sprintf('bsp-light-dark-plan-%i.png',i_problem));
    
    fprintf('For verification (allow for numerical inaccuracy):\n');
    fprintf('(Reference) Total cost of optimized trajectory: %2.6f\n', verify_cost(i_problem));
    
    % Simulate execution of trajectory (test to see why the maximum likelihood observation assumption does not hold)
    %simulate_bsp_trajectory(B(:,1), Uopt, model);
    
    disp('press enter to continue to the next problem')
    pause();
end

end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Dynamics function: x_t+1 = dynamics_func(x_t, u_t, q_t, dt)

function x_tp1 = dynamics_func(x_t, u_t, q_t, model)

x_tp1 = x_t + u_t*model.dT + 0.01*q_t;

end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Observation function: z_t = obs_func(x_t, r_t)

function z_t = obs_func(x_t, r_t, model)

intensity = 0.5*0.5*(x_t(1))*(x_t(1)) + 1e-6;
z_t = x_t + sqrt(intensity)*r_t;

end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Plot problem specific stuff -- background gradient to show light/dark domain
function plot_domain(B, model)
[imx, imy] = meshgrid(-5:0.025:3,-3:0.025:3);
sx = size(imx,1);
sy = size(imy,2);
imz = ones(sx,sy);

for i = 1:sx
   for j = 1:sy
       imz(i,j) = (1.0/((imx(i,j)).^2 + 1));
   end
end

% Plot
hold on;
imagesc(imx(1,:),imy(:,1),imz);
set(gca,'position',[0 0 1 1],'ydir','normal');
colormap(gray);
axis off;
axis image;

end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Setup parameters for trajectory optimization in belief space using SQP
function cfg = setup_sqp_params()

% penalty sqp config parameters
cfg = {};
cfg.improve_ratio_threshold = .1;
cfg.min_trust_box_size = 1e-3;
cfg.min_approx_improve = 1e-4;
cfg.max_iter = 50;
cfg.trust_shrink_ratio = .1;
cfg.trust_expand_ratio = 1.5;
cfg.cnt_tolerance = 1e-4;
cfg.max_penalty_coeff_increases = 3;
cfg.penalty_coeff_increase_ratio = 10;
cfg.initial_trust_box_size = 1;
cfg.initial_penalty_coeff = 50;

disp('Optimizer parameters:');
disp(cfg)

end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%