function test_bsp_spotlight()

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

model.T = 10; % number of time steps in trajectory

model.alpha_belief = 1; % weighting factor for penalizing uncertainty at intermediate time steps
model.alpha_final_belief = 100; % weighting factor for penalizing uncertainty at final time step
model.alpha_control = 1; % weighting factor for penalizing control cost

model.xMin = [-4; -3]; % minimum limits on state (xMin <= x)
model.xMax = [4; 3]; % maximum limits on state (x <= xMax)
model.uMin = [-1; -1]; % minimum limits on control (uMin <= u)
model.uMax = [1; 1]; % maximum limits on control (u <= uMax)

model.Q = eye(model.qDim); % dynamics noise variance
model.R = eye(model.rDim); % observation noise variance

model.lightcoords = zeros(2,3); % Coordinates of the triangular spotlight (specified in counter-clockwise order)
model.lightcoords(:,1) = [0;0];
model.lightcoords(:,2) = [1.5;-0.75];
model.lightcoords(:,3) = [1.5;0.75];

model.displaybg = 0; % Flag to toggle background gradient display
model.display_spotlight_trajectory = 0; % Flag to toggle display of spotlight trajectory for Q 2 (iii)

model.dynamics_func = @dynamics_func; % function handle to dynamics function
model.obs_func = @obs_func; % function handle to observation function
model.plot_domain = @plot_domain; % function handle to plotting experiment specific stuff
model.setup_sqp_params = @setup_sqp_params;

% Setup initial conditions for problem
x1 = [-3;2];
goal = [-3;-2]; 
SqrtSigma1 = eye(2);

model.start = x1; % start (mean of initial belief)
model.goal = goal; % goal
Theta = [0; pi/2; pi];

verify_cost = [17.355568; 14.027668; 10.541347];

for i_problem=1:3
    
    model.theta = Theta(i_problem);
    model.light = rotate_spotlight(model.lightcoords, model.theta);
    
    % setup initial control vector -- straight line initialization from start to goal
    U = repmat(((model.goal - model.start)/(model.T-1)), 1, model.T-1);
    
    B = zeros(model.bDim,model.T);
    B(:,1) = compose_belief(x1, SqrtSigma1, model);
    for t=1:model.T-1
        B(:,t+1) = belief_dynamics(B(:,t), U(:,t), [], model);
    end
    clf;
    model.displaybg = 0;
    plot_belief_trajectory(B, U, model);
    pause(0.1);
    
    [Bopt, Uopt, ~] = belief_opt_penalty_sqp(B, U, model);
    clf;
    model.displaybg = 1;
    plot_belief_trajectory(Bopt, Uopt, model);
    
    % save trajectory to png file 
    saveas(gcf, sprintf('bsp-static-spotlight-plan-%i.png',i_problem));
    
    cost = compute_cost(compose_belief(x1, SqrtSigma1, model), Uopt, model);
    fprintf('Total cost of optimized trajectory: %f\n', cost);
    
    fprintf('For verification (allow for numerical inaccuracy):\n');
    fprintf('(Reference) Total cost of optimized trajectory: %2.6f\n', verify_cost(i_problem));
    
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

sd = signed_dist_spotlight(x_t(1:2), model.light);
intensity = (1/(1 + exp(-sd))).^2;

%fprintf('Signed dist: %f; Noise: %f\n',sd,noise);

z_t = x_t + intensity*r_t;

end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function plot_domain(B, model)
[imx, imy] = meshgrid(-5:0.025:2.5,-2.5:0.025:3);
sx = size(imx,1);
sy = size(imy,2);
imz = ones(sx,sy);

if (model.displaybg)
	if (model.display_spotlight_trajectory)
		[x_T, ~] = decompose_belief(B(:,model.T), model);
		light = rotate_spotlight(model.lightcoords, x_T(3));
	else
		light = model.light;
	end
		
    for i = 1:sx
        for j = 1:sy
            sd = signed_dist_spotlight([imx(i,j);imy(i,j)], light);
            imz(i,j) = 1 - 1/(1 + exp(-sd));
        end
    end
end

% Plot
hold on;
imagesc(imx(1,:),imy(:,1),imz);
set(gca,'position',[0 0 1 1],'ydir','normal');
colormap(gray);
axis off;
axis image;

if (model.display_spotlight_trajectory)
    for t=1:model.T
        [x_t, ~] = decompose_belief(B(:,t), model);
        light = rotate_spotlight(model.lightcoords, x_t(3));
        Lplot = [light light(:,1)];
        if (t < model.T)
            plot(Lplot(1,:),Lplot(2,:),'Color',[(model.T - t)/model.T t/model.T t/model.T],'Marker','*','LineStyle','--','LineWidth',2.0);
        else
            plot(Lplot(1,:),Lplot(2,:),'Color',[0.0 0.0 1.0],'Marker','*','LineWidth',3.0);
        end
    end
else
    Lplot = [model.light model.light(:,1)];
    plot(Lplot(1,:),Lplot(2,:),'b*-','LineWidth',3.0);
end
plot(Lplot(1,1), Lplot(2,1), 'rs', 'LineWidth', 5.0);

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
% Computes the signed distance from a point to a triangular spotlight

function dist = signed_dist_spotlight(pt, T)
T = shrink_spotlight(T);
d12 = line_point_signed_dist(T(:,1),T(:,2),pt);
d23 = line_point_signed_dist(T(:,2),T(:,3),pt);
d31 = line_point_signed_dist(T(:,3),T(:,1),pt);
if(d12 <= 0.0 && d23 <= 0.0 && d31 <= 0.0)
    dist = max([d12, d23, d31]);
else
    d12 = segment_point_dist(T(:,1), T(:,2), pt);
    d23 = segment_point_dist(T(:,2), T(:,3), pt);
    d31 = segment_point_dist(T(:,3), T(:,1), pt);
    dist = min([d12 d23 d31]);
end
end

function dist = line_point_signed_dist(p1, p2, pt)
a = p2(2) - p1(2);
b = p1(1) - p2(1);
c = p2(1) * p1(2) - p1(1) * p2(2);
dist = (a*pt(1) + b*pt(2) + c)/sqrt(a*a + b*b);
end

function dist = segment_point_dist(p1, p2, pt)
len = sum ((p2(:,1) - p1(:,1)).^2);
t = (pt(:,1) - p1(:,1))' * (p2(:,1) - p1(:,1))/len;
t = max(t,0.0);
t = min(t,1.0);
pn(:,1) = p1(:,1) + t*(p2(:,1) - p1(:,1));
dist = sqrt(sum((pn(:,1) - pt(:,1)).^2));
end

function T = shrink_spotlight(T)
n13 = [T(2,1) - T(2,3); T(1,3) - T(1,1)];
n32 = [T(2,3) - T(2,2); T(1,2) - T(1,3)];
n21 = [T(2,2) - T(2,1); T(1,1) - T(1,2)];
T(:,1) = T(:,1) + 0.75*n32/norm(n32);
T(:,2) = T(:,2) + 0.75*n13/norm(n13);
T(:,3) = T(:,3) + 0.75*n21/norm(n21);
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Rotate triangular spotlight

function light = rotate_spotlight(light, theta)
R = [cos(theta) -sin(theta); sin(theta) cos(theta)];
base = repmat(light(:,1),1,3);
light = R*(light - base) + base;
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%