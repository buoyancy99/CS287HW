classdef mujoco_mdp
    % Standard MDP class for MuJoCo models
    properties
        model  % Contains model parameters from the specified model_file in the constructor
        ob_dim  % Dimension of the observation; i.e. the MDP state
        action_dim  % Dimension of the action = number of joints of the robot that can be controlled
        action_limits  % action_dim x 2 array of the minimum and maximum control limits for each action dimension
        timestep  % The amount of time each step in MuJoCo corresponds to
        frame_skip  % The number of times to apply each action in a row
        init_qpos  % The initial robot link angles; used for reseting MuJoCo after each rollout
        init_qvel  % The intiial robot link velocities; used for reseting MuJoCo after each rollout
        init_qfrc_constraint  % Other initial robot settings; used for resetting MuJoCo after each rollout
        init_xpos  % The initial robot link positions; used for resetting MuJoCo after each rollout
        init_ctrl  % The initial control vector for the robot; used for resetting MuJoCo after each rollout
    end
    methods (Abstract)
        ob = get_observation(mdp)  % Converts the current MuJoCo simulator state into a MDP state
        reward = get_reward(mdp, xpos_before, xpos_after, qpos_before, qpos_after) % Calculates the reward,
                                                 % where [xpos_before, qpos_before] is the previous state and
                                                 % [xpos_after, qpos_after] describes the state after taking the action
        done = check_if_done(mdp)  % Returns true if the MDP is in an unrecoverable state. Otherwise returns false.
    end

    methods
        function mdp = mujoco_mdp(model_file)
            % Loads the model_file into MuJoCo, and sets all the object properties
            % except for timestep and frame_skip
            % INPUTS:
            %     model_file - XML model file, i.e. the ones in the folder mujoco_models
            mj clear
            mj('load', model_file)
            mdp.model = mj('getmodel');
            mdp.ob_dim = length(mdp.get_observation());
            data = mj('getdata');
            mdp.action_limits = mdp.model.actuator_ctrlrange;
            mdp.action_dim = size(mdp.action_limits, 1);
            mdp.init_qpos = data.qpos;
            mdp.init_qvel = data.qvel;
            mdp.init_xpos = data.xpos;
            mdp.init_qfrc_constraint = data.qfrc_constraint;
            mdp.init_ctrl = data.ctrl;
        end

        function obs = reset(mdp)
            % Reset function. Must call at the start of each rollout.
            mj('set', 'qpos', mdp.init_qpos, ...
                      'qvel', mdp.init_qvel, ...
                      'qfrc_constraint', mdp.init_qfrc_constraint, ...
                      'xpos', mdp.init_xpos, ...
                      'ctrl', mdp.init_ctrl);
            mj('forward')
            obs = mdp.get_observation();
        end

        function [next_obs, reward, done] = step(mdp, action)
            % Transition function.
            % INPUTS:
            %     action - action to take; vector of dimension mdp.action_dim x 1
            % OUTPUTS:
            %     next_obs - MDP state after taking action; vector of dimension mdp.ob_dim x 1
            %     reward - the reward from taking the action in the current state of the MDP; is a real number
            %     done - true if the MDP is in an unrecoverable state and the rollout should be stopped; false otherwise
            xpos_before = mj('get', 'xpos');
            qpos_before = mj('get', 'qpos');
            mj('set', 'ctrl', action);
            for i=1:mdp.frame_skip
                mj step
            end
            xpos_after = mj('get', 'xpos');
            qpos_after = mj('get', 'qpos');
            reward = mdp.get_reward(xpos_before, xpos_after, qpos_before, qpos_after);
            next_obs = mdp.get_observation();
            done = mdp.check_if_done();
        end
    end
end
