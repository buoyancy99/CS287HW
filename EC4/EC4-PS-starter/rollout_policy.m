function [ obs, actions, rewards ] = rollout_policy( mdp, policy, max_length, pause_length )
% Executes a single rollout of the policy
% Returns T x D array of observations, actions, and rewards.
%   T = number of timesteps
%   D = mdp.input_dim, mdp.output_dim, and 1 for obs, actions, and rewards, respectively
%
% INPUTS:
%     mdp - a mujoco_mdp object, or any MDP object with reset() and step() functions
%     policy - a mujoco_policy object, or any policy object with a step() function
%     max_length - the maximum length of rollout accepted
%     pause_length (optional) - specify an amount to pause after each step in
%                               the rollout; helps for slowing down the simulation
%                               to watch it. Default is 0.

    if nargin < 4
        pause_length = 0;
    end
    ob = mdp.reset();
    obs = [];
    actions = [];
    rewards = [];
    for i = 1:max_length
        obs = [obs; ob'];
        action = policy.step(ob);
        [next_ob, reward, done] = mdp.step(action);
        actions = [actions;action'];
        rewards = [rewards;reward];
        if done
            break
        end
        pause(pause_length);
        ob = next_ob;
    end
end

