function [ ] = cross_entropy( mdp, policy )
    % Uses the Cross Entropy Method (CEM) [1] to directly search for optimal
    % policy parameters.
    %
    % INPUTS:
    % mdp - an object of the class mujoco_mdp (see mujoco_mdp.m)
    % policy - an object of the class mujoco_policy (see mujoco_policy.m)
    %
    % [1] The CEM algorithm is described in Thiery and Scherrer, "Improvements
    % on Learning Tetris with Cross Entropy."
    % https://hal.inria.fr/inria-00418930/document
    global ce_avg_reward_for_samples ce_max_reward_for_samples

    rng(1)  % Sets the random seed to 1

    N = 1000;  % Number of sample params generated at each iteration
    p = 10;   % Number of best params selected from each set of N
    z = 1;  % Constant noise term to add to the variance
    n_rollouts = 1;  % Number of rollouts for evaluating each set of policy parameters
    n_iterations = 200;  % number of iterations to run
    max_timesteps = 1000;  % Maximum number of timesteps allowed for a single rollout of the policy
    discount = 0.99;  % Discount factor, for calculating the total discounted reward
    policy_log_sigma = -10 * ones(policy.output_dim,1);  % Controls the amount of stochasticity in the policy
    
    tic
    % Initialize the mean parameters to zero. params_mean should be updated for each iteration of CEM.
    params_mean = zeros(numel(policy.theta),1)';
    % Initialize the variance of the parameters to one. (We assume there is zero
    % covariance across the policy parameters.) params_variance should be updated for each iteration of CEM.
    params_variance = ones(numel(policy.theta),1);
   
    % Perform n_iterations of CEM
    for t=1:n_iterations
        fprintf('Iteration %d:\n', t);

        % Set parameters of the neural net policy to params_mean and policy_log_sigma
        policy.set_theta_log_sigma(reshape(params_mean, policy.theta_dim, 1), policy_log_sigma);

        % Sample N weights from the multivariate Gaussian with mean params_mean
        % and covariance diag(params_variance)
        params = mvnrnd(params_mean,diag(params_variance),N);

        % TODO: Evaluate each potential set of weights; store *average* and *maximum*
        %       total discounted reward for each rollout in ce_avg_reward_for_samples{t}
        %       and ce_max_reward_for_samples{t}, respectively.
        discounted_reward = zeros(N,1);
        % YOUR CODE HERE

        ce_avg_reward_for_samples(t) = sum(discounted_reward) / N;
        ce_max_reward_for_samples(t) = max(discounted_reward);
        fprintf('\tAverage total reward for sampled parameters: %f\n', ce_avg_reward_for_samples(t));
        fprintf('\tMaximum total reward for sampled parameters: %f\n', ce_max_reward_for_samples(t));

        % TODO: Update the mean (params_mean) and variance (params_variance)
        %       based on the p best-performing policy parameters
        % YOUR CODE HERE

        toc
    end
end

