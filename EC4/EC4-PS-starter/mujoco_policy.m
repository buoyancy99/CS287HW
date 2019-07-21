classdef mujoco_policy < handle
    % This class is a stochastic policy parameterized by a neural network.
    % - The architecture of the neural network is defined in the function construct_net.
    % - A policy's parameters theta consist of the (flattened and concatenated) weights of the nueral network.
    % - The input to the neural network is the current observation; i.e. a vector
    %   of size input_dim x 1. The output of the neural network is a potential action,
    %   a vector of size output_dim x 1.
    % - The policy is stochastic. The chosen action is drawn from a Gaussian
    %   distribution centered at the output of the neural network, with covariance
    %   diag(exp(log_sigma.^2)), and then this is multipled by the scale_factor.

    properties
        net  % A neural network instantiation of the type matlab_nn/neural_network.m
        input_dim  % The dimension of observations, which are input to the neural network
        output_dim  % The dimension of actions, which are output from the neural network
        action_limits  % output_dim x 2 array of the minimum and maximum control limits for each action dimension
        theta_dim  % The dimension of the parameter vector of this policy (i.e., the parameter vector of the neural network policy.net)
        theta  % The values of the neural network parameters. A vector of size theta_dim x 1
        log_sigma  % The log standard deviation of the Gaussian distribution that the action is selected from
        scale_factor  % The scale to multiply each neural network output by
        penalty_coeff  % For PPO: The penalty coefficient on the KL divergence term in the objective function
        max_kl  % For PPO: The maximum allowed KL divergence
    end

    methods (Abstract)
    end

    methods
        function policy = mujoco_policy(input_dim, output_dim, action_limits)
            % Constructs a mujoco_policy object. See comment at the top of the class
            % for more information about this policy.
            %
            % INPUTS:
            %     input_dim - the dimension of observations
            %     output_dim - the dimension of actions / outpus
            %     action_limits - a output_dim x 2 vector of (min,max) limits for each action dimension
            policy.input_dim = input_dim;
            policy.output_dim = output_dim;
            policy.action_limits = action_limits;
            policy.net = policy.construct_net(input_dim, output_dim);

            % Calculate the dimensionality of the policy parameters (i.e., theta)
            % by summing up the dimensions of all the parameters in the layers
            % of the neural network
            dim = 0;
            for l = 1:size(policy.net.layers, 2)
                dim = dim + size(policy.net.layers{l}.get_paramvec(), 1);
            end
            policy.theta_dim = dim;

            % Initialize policy parameters randomly from a normal distribution
            % centered at zero
            random_theta = normrnd(zeros(policy.theta_dim, 1), ...
                                   0.08 * ones(policy.theta_dim, 1));
            policy.set_theta_log_sigma(random_theta, zeros(output_dim,1));

            policy.penalty_coeff = 0.5;
            policy.max_kl = 0.1;
        end
            
        function policy = set_theta_log_sigma(policy, theta, log_sigma)
            % Sets the parameters of the policy - both theta (the weights and
            % bias values of the neural network) and log_sigma, the log of the
            % variance for choosing the final action.
            % Note: Do not set theta and sigma directly by assigning to policy.theta
            % and policy.log_sigma directly, since that will not update the
            % underlying neural network correctly,
            policy.log_sigma = reshape(log_sigma, [], 1);
            policy.theta = reshape(theta, [], 1);

            curr_idx = 1;
            for l = 1:size(policy.net.layers, 2)
                paramvec = policy.net.layers{l}.get_paramvec();
                if size(paramvec, 1) > 0
                    policy.net.layers{l}.set_paramvec(policy.theta(curr_idx:curr_idx+size(paramvec,1)-1));
                    curr_idx = curr_idx + size(paramvec,1);
                end
            end
        end

        function action = step(policy, obs)
            % Chooses the action to take, given the current observation obs.
            % See the class comment for more details on how the action is chosen.

            obs = reshape(obs, [], 1);
            mean_action = policy.net.forward(obs, false);
            action = mujoco_policy.meanstd_sample(mean_action, policy.log_sigma);
            action = action * policy.scale_factor;

            % Clips actions to their maximum and minimum accepted values
            % (these are defined in the model xml file that is passed to MuJoCo)
            action = min(action, policy.action_limits(:,2));
            action = max(action, policy.action_limits(:,1));
        end

        function [obj, grad] = ppo_objective_grad( policy, theta_logsigma, oldmean_actions, oldlogsigma, obs, actions, advantages )
            % Returns both the value of the Proximal Policy Optimization (PPO) [1]
            % objective and the gradient of the objective with respect to theta and log_sigma.
            %
            % INPUTS:
            %     theta_logsigma - the new theta and log_sigma parameters of the
            %                      policy, at which to calculate the values of the
            %                      PPO objective and gradient
            %     oldmean_actions - a N x action_dim array; output of neural net
            %                       with previous policy's theta parameters,
            %                       from passing in the obs array
            %     oldlogsigma - previous policy's log_sigma parameters
            %     (N = total number of timesteps across all rollouts)
            %     obs - a N x obs_dim array of observations from the rollouts
            %     actions - a N x action_dim array of actions from the rollouts
            %     advantages - a N x 1 vector of discounted total estimated advantages (i.e., the
            %                  difference between the q-value associated with 
            %                  taking action a in state s, minus the estimated
            %                  value function at state s under the current policy)
            %
            % Let r be the vector of rewards (from each of the N timesteps). Then
            %     advantages(i) =  \sum_{j=i}^{T} discount^{j-i} * a_n(j)
            % where
            %     a_n(j) = r(j) + discount * estimated_V(obs(j+1)) - estimated_V(obs(j))
            %     T = number of timesteps in the rollout
            %
            % OUTPUTS:
            %     obj - a real value, equal to the PPO objective given the new
            %           policy's parameters theta_logsigma and the previous policy's
            %           oldlogsigma and outputs oldmean_actions
            %     grad - a length dim_theta + dim_logsima vector with the gradients
            %            with respect to theta first, and then the gradients with
            %            respect to log_sigma
            %
            % [1] PPO is a varient of TRPO, described in http://arxiv.org/abs/1502.05477
            %     The key difference is that PPO solves an unconstrained optimization problem.

            obj = policy.ppo_objective(theta_logsigma, oldmean_actions, oldlogsigma, obs, actions, advantages);
            grad = policy.ppo_grad(theta_logsigma, oldmean_actions, oldlogsigma, obs, actions, advantages);
        end

        function obj = ppo_objective( policy, theta_logsigma, oldmean_actions, oldlogsigma, obs, actions, advantages )
            % Returns the value of the PPO objective with a pre-specified previous
            % neural net policy, and a new setting of policy parameters specified
            % by the input theta_logsigma.
            % For detailed comments on the inputs, please refer to the function
            % ppo_objective_grad above.

            % Save the current policy's parameters theta and log_sigma, and reset
            % to these values at the end of the function
            prev_theta = policy.theta;
            prev_logsigma = policy.log_sigma;
            policy.set_theta_log_sigma(theta_logsigma(1:policy.theta_dim), theta_logsigma(policy.theta_dim+1:end));

            num_timesteps = length(obs);
            mean_actions = policy.net.forward_mat(obs', false)';

            logprob_actions = -0.5 * sum((((actions ./ policy.scale_factor) - mean_actions) ./ repmat(exp(policy.log_sigma)', num_timesteps, 1)) .^ 2, 2) - sum(policy.log_sigma);
            oldlogprob_actions = -0.5 * sum((((actions ./ policy.scale_factor)  - oldmean_actions) ./ repmat(exp(oldlogsigma)', num_timesteps, 1)) .^ 2, 2) - sum(oldlogsigma);
            ratio = exp(logprob_actions - oldlogprob_actions);
            surr = mean(ratio .* advantages);
            kl = -0.5 * policy.output_dim + sum(policy.log_sigma - oldlogsigma) + mean(sum((repmat(exp(oldlogsigma).^2', num_timesteps, 1) + (oldmean_actions - mean_actions).^2) ./ repmat((2 * exp(policy.log_sigma).^2)', num_timesteps, 1), 2));

            obj = policy.penalty_coeff * kl - surr;
            if kl > policy.max_kl || isinf(obj)
                obj = 1e10;
            end

            policy.set_theta_log_sigma(prev_theta, prev_logsigma);
        end

        function grad = ppo_grad( policy, theta_logsigma, oldmean_actions, oldlogsigma, obs, actions, advantages )
            % Returns the gradient of the PPO objective.
            % For detailed comments on the inputs and output, please refer to the function
            % ppo_objective_grad above.

            % Save the current policy's parameters theta and log_sigma, and reset
            % to these values at the end of the function
            prev_theta = policy.theta;
            prev_logsigma = policy.log_sigma;
            policy.set_theta_log_sigma(theta_logsigma(1:policy.theta_dim), theta_logsigma(policy.theta_dim+1:end));

            num_timesteps = size(obs, 1);
            % Partial derivatives of surr with respect to [theta; log_sigma]
            policy_gradients = policy.net.forward_backward_pg_mat(obs');
            single_gradient = [];
            for g=1:length(policy_gradients)
                single_gradient = [single_gradient, policy_gradients{g}];
            end
            mean_actions = policy.net.forward_mat(obs', false)';

            prob_actions = exp(-0.5 * sum(bsxfun(@rdivide, ((actions ./ policy.scale_factor) - mean_actions), exp(policy.log_sigma')).^2, 2) - sum(policy.log_sigma));
            oldprob_actions = exp(-0.5 * sum(bsxfun(@rdivide, ((actions ./ policy.scale_factor) - oldmean_actions), exp(oldlogsigma')).^2, 2) - sum(oldlogsigma));
            grad_surr_coeff = advantages .* (prob_actions ./ oldprob_actions);
            grad_theta_actionprob = bsxfun(@rdivide, (actions ./ policy.scale_factor) - mean_actions, exp(policy.log_sigma') .^ 2);
            grad_theta = sum(bsxfun(@times, grad_surr_coeff, ...
                             permute(mtimesx(reshape(grad_theta_actionprob', 1, ...
                                                     policy.output_dim, num_timesteps), ...
                                             single_gradient), ...
                                     [3,2,1])), ...
                             1);

            grad_logsigma = sum(bsxfun(@times, grad_surr_coeff, ...
                                -1 + bsxfun(@rdivide, (actions ./ policy.scale_factor) - mean_actions, exp(policy.log_sigma')).^2), 1);

            % Partial derivatives of -penalty_coeff * kl with respect to [theta; log_sigma]
            frac = bsxfun(@rdivide, (oldmean_actions - mean_actions), exp(policy.log_sigma').^2);
            grad_theta = grad_theta + policy.penalty_coeff * sum(permute(mtimesx(reshape(frac', 1, policy.output_dim, num_timesteps), single_gradient), [3,2,1]), 1);
            grad_logsigma = grad_logsigma + policy.penalty_coeff * ...
                            sum(bsxfun(@plus, exp(oldlogsigma').^2, (oldmean_actions - mean_actions).^2), 1) ./ (exp(policy.log_sigma').^2);

            grad_theta = grad_theta ./ num_timesteps;
            grad_logsigma = grad_logsigma ./ num_timesteps;
            grad_logsigma = grad_logsigma + -1 * policy.penalty_coeff;

            grad = [grad_theta'; grad_logsigma'];
            grad = -1 * grad;
            policy.set_theta_log_sigma(prev_theta, prev_logsigma);
        end

        function correct = check_ppo_gradient_with_rollout( policy, obs, actions, advantages )
            % Check the PPO gradient numerically, to make sure it returns the correct
            % gradient.
            %
            % INPUTS:
            %     obs, actions, advantages - see explanation in ppo_objective_grad above
            %
            % OUTPUTS:
            %     correct - return True if the analytical and numerical gradients match

            stepdir_theta = rand(size(policy.theta,1), size(policy.theta,2));
            stepdir_theta2 = rand(size(policy.theta,1), size(policy.theta,2));
            stepdir_log_sigma = rand(size(policy.log_sigma,1), size(policy.log_sigma,2));

            num_timesteps = size(obs,1);
            mean_actions = policy.net.forward_mat(obs', false)';

            curr_theta = policy.theta + stepdir_theta2 * 0.0001;

            f = @(x) policy.ppo_objective([curr_theta+stepdir_theta*x; ...
                                           policy.log_sigma+stepdir_log_sigma*x], ...
                                           mean_actions, policy.log_sigma, ...
                                           obs, actions, advantages);
            numgrad = (f(1e-5) - f(-1 * 1e-5)) / (2*1e-5)
            g_theta_log_sigma = policy.ppo_grad([curr_theta; policy.log_sigma], ...
                                                mean_actions, policy.log_sigma, ...
                                                obs, actions, advantages);
            g_theta = g_theta_log_sigma(1:policy.theta_dim);
            g_log_sigma = g_theta_log_sigma(policy.theta_dim+1:end);
            anagrad = sum(sum(stepdir_theta .* g_theta)) + sum(stepdir_log_sigma .* g_log_sigma);
            correct = false;
            if abs(numgrad - anagrad) < 1e-8
                correct = true;
            end
        end
    end

    methods(Static)
        function sample = meanstd_sample( mean_action, log_std_action )
            % Samples a vector, with each element sampled from a separate
            % univariate normal distribution.
            %
            % INPUTS:
            %     mean_action - a N x 1 vector; element i is the mean of the i'th normal distribution
            %     log_std_action - a N x 1 vector; element i is the log standard deviation of the i'th normal distribution

            sample = mean_action + exp(log_std_action) .* normrnd(zeros(size(mean_action)), ...
                     ones(size(mean_action)));
        end
    end
end
