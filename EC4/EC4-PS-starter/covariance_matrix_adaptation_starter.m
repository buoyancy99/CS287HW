function [ ] = covariance_matrix_adaptation( mdp, policy )
    % Uses Covariance Matrix Adaptation (CMA) [1] to directly search for
    % optimal policy parameters.
    %
    % INPUTS:
    % mdp - an object of the class mujoco_mdp (see mujoco_mdp.m)
    % policy - an object of the class mujoco_policy (see mujoco_policy.m)
    %
    % [1] The standard CMA algorithm is concisely described in Section 4.1 of
    % Wampler and Popovic, "Optimal Gait and Form for Animal Locomotion."
    % http://grail.cs.washington.edu/projects/animal-morphology/s2009/Optimal_Gait_and_Form_for_Animal_Locomotion.pdf

    global cma_avg_reward_for_samples cma_max_reward_for_samples

    rng(1)  % Sets the random seed to 1

    N = 1000;  % Number of sample params generated at each iteration (= \lambda in the Wampler & Popovic paper)
    p = 10;   % Number of best params selected from each set of N (= \mu in the Wampler & Popovic paper)
    step = 0.3;  % Step size to take towards the new covariance matrix; must be in the range (0,1]
                 % (= c_{cov} in the Wampler & Popovic paper)
    n_rollouts = 1;  % Number of rollouts for evaluating each set of policy parameters
    n_iterations = 200;  % Number of iterations to run
    max_timesteps = 1000;  % Maximum number of timesteps allowed for a single rollout of the policy
    discount = 0.999;  % Discount factor, for calculating the total discounted reward
    policy_log_sigma = -10 * ones(policy.output_dim,1);  % Controls the amount of stochasticity in the policy
    
    cma_weights = zeros(1,p);
    % TODO: Pre-compute the weights used to update the mean and covariance matrix estimates.
    %       Save these in cma_weights.
    % YOUR CODE HERE

    tic
    % Initialize the mean parameters to zero. params_mean should be updated for each iteration of CMA.
    params_mean = zeros(numel(policy.theta),1)';
    % Initialize the covariance matrix to 100*I. cov_matrix should be updated for each iteration of CMA.
    cov_matrix = 100 * eye(numel(policy.theta));

    % Perform n_iterations of CMA
    for t=1:n_iterations
        fprintf('Iteration %d:\n', t);

        % Set parameters of the neural net policy to params_mean and policy_log_sigma
        policy.set_theta_log_sigma(reshape(params_mean, policy.theta_dim, 1), policy_log_sigma);

        % Sample N weights from the multivariate Gaussian with mean params_mean and covariance cov_matrix
        params = mvnrnd(params_mean,cov_matrix,N);

        % TODO: Evaluate each potential set of weights; store *average* and *maximum*
        %       total discounted reward for each rollout in cma_avg_reward_for_samples{t}
        %       and cma_max_reward_for_samples{t}, respectively.
        discounted_reward = zeros(N,1);
        % YOUR CODE HERE

        cma_avg_reward_for_samples(t) = sum(discounted_reward) / N;
        cma_max_reward_for_samples(t) = max(discounted_reward);
        fprintf('\tAverage total reward for sampled parameters: %f\n', cma_avg_reward_for_samples(t));
        fprintf('\tMaximum total reward for sampled parameters: %f\n', cma_max_reward_for_samples(t));

        % TODO: Update the mean (params_mean) and covariance matrix (cov_matrix)
        %       based on the p best-performing policy parameters
        % YOUR CODE HERE

        toc
    end
end
