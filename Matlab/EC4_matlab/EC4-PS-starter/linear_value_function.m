classdef linear_value_function < handle
    % This class is a simple value function estimator, that approximates the value
    % function with a linear combination of features - the weights of which are
    % fit using least squares.

    properties
        coeffs  % weights for the linear combination of features
    end

    methods
        function vf = linear_value_function()
            vf.coeffs = [];
        end

        function vf = fit(vf, all_obs, discounted_rewards)
            % Uses least squares to find the best feature coefficients, to fit
            % the value function. Updates the coeffs property of vf to be the new
            % set of coefficients.
            % INPUTS:
            %     all_obs - N x obs_dim array of observations from all rollouts
            %               (N = total number of timesteps)
            %     discounted_rewards - a Nx1 vector of all total discounted rewards
            %                          element i = \sum_{j=i}^T \gamma^{j-i} * reward(j)
            all_obs_features = linear_value_function.get_features(all_obs);
            vf.coeffs = all_obs_features \ discounted_rewards;
        end

        function value = predict(vf, obs)
            % INPUTS:
            %     obs - N x obs_dim array of observations from rollouts
            % OUTPUTS:
            %     value - N x 1 vector of predicted values for each of the obeservations
            if length(vf.coeffs) == 0
                value = zeros(size(obs,1),1);
            else
                value = linear_value_function.get_features(obs) * vf.coeffs;
            end
        end
    end

    methods(Static)
        function features = get_features(obs)
            % INPUTS:
            %     obs - array of N x obs_dim observations
            % OUTPUTS:
            %     features - array of N x feature_dim features; row i is the
            %                features for the observation in row i of obs
            l = size(obs, 1);
            obs = min(10,max(-10,obs));
            al = reshape(1:l, [], 1) / 100;
            features = [obs, obs .^ 2, al, al .^ 2, al .^ 3, ones(l,1)];
        end
    end
end
