function [ ] = policy_gradient_ppo( mdp, policy)
    % Uses Proximal Policy Optimization (PPO) [1] to directlys earch for optimal
    % policy parameters.
    %
    % INPUTS:
    % mdp - an object of the class mujoco_mdp (see mujoco_mdp.m)
    % policy - an object of the class mujoco_policy (see mujoco_policy.m)
    %
    % [1] PPO is a varient of TRPO, described in http://arxiv.org/abs/1502.05477
    %     The key difference is that PPO solves an unconstrained optimization problem.

    global policy_theta_avg_reward

    addpath('minFunc_2012/minFunc/');
    addpath('minFunc_2012/autoDif/');

    policy.scale_factor = 100;
    rng(1)  % Sets the random seed to 1

    % Set initial theta and log_sigma
    theta = normrnd(zeros(policy.theta_dim, 1), ...
                    0.08 * ones(policy.theta_dim, 1));
    log_sigma = zeros(mdp.action_dim, 1);
    policy.set_theta_log_sigma(theta, log_sigma);

    discount = 0.999;  % Discount factor, for calculating the total discounted reward
    max_timesteps = 10000;  % Maximum number of timesteps allowed for a single rollout of hte policy
    timesteps_per_batch = 50000;  % Number of total timesteps of rollouts to collect per iteration
    pause_after = 49500;  % The timesteps at which to slow down rollouts, so it is possible to watch the rest for that iteration
    n_iterations = 200;  % number of iterations of policy gradient to do

    vf = linear_value_function();  % Linear function; a baseline that approximates the value function
    
    tic
    for t=1:n_iterations
        % TODO: Perform rollouts to gather a set of observations (all_obs), actions
        % (all_actions), and rewards (all_rewards). Compute the discounted total
        % rewards(all_discounted_rewards) and the discounted total advantages (all_advantages).
        %
        % YOUR CODE HERE

        % The output of the current neural net, with the rollouts' observations all_obs as input
        mean_actions = policy.net.forward_mat(all_obs', false)';

        % Use LBFGS to find the new theta and log_sigma that optimize the PPO objective function
        ppo_f = @(x) policy.ppo_objective_grad(x, mean_actions, policy.log_sigma, all_obs, all_actions, all_advantages);
        options = [];
        options.display = 'none';
        options.maxFunEvals = 20;
        options.Method = 'lbfgs';
        options.Display = 'full';
        options.useMex = 0;
        new_theta_logsigma = minFunc(ppo_f,[policy.theta; policy.log_sigma], options); 
        new_theta = new_theta_logsigma(1:policy.theta_dim);
        new_log_sigma = new_theta_logsigma(policy.theta_dim+1:end);

        % Update the policy's parameters with the new theta and log_sigma
        policy.set_theta_log_sigma(new_theta, new_log_sigma);

        % TODO: Fit linear value function vf with the observations, and corresponding
        %       *discounted* total rewards (see the function vf.fit for the equation)
        % YOUR CODE HERE
        
        toc
    end
end
