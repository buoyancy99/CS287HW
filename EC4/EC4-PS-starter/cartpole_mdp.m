classdef cartpole_mdp < mujoco_mdp
    % Implements a MDP for the MuJoCo Cartpole. The goal is to keep the cartpole balanced.
    % See the mujoco_mdp class for detailed comments on class properties and methods.

    properties
    end
    
    methods
        function mdp = cartpole_mdp(model_file)
            mdp = mdp@mujoco_mdp(model_file);
            mdp.timestep = 0.02;
            mdp.frame_skip = 5;
        end

        function ob = get_observation(mdp)
            % Converts the current MuJoCo simulator state into a MDP state
            % qpos is a two-dimensional vector:
            %     qpos(1) - value from -1 to 1, describing location of cart
            %     qpos(2) - value from -\pi/2 to \pi/2, describing angle of pole
            ob = [mj('get', 'qpos'); ...
                  max(-10, min(10, mj('get', 'qvel'))); ...
                  max(-10, min(10, mj('get', 'qfrc_constraint')))];
        end

        function reward = get_reward(mdp, xpos_before, xpos_after, qpos_before, qpos_after)
            % Calculates the reward, where [xpos_before, qpos_before] is the previous state
            % and [xpos_after, qpos_after] describes the state after taking the action
            %
            % For the cartpole, the reward is based on the angle of the hinge.

            [hinge_joint_before, hinge_joint_after] = cartpole_mdp.get_hinge_joint(qpos_before, qpos_after);
            reward = cos(hinge_joint_after);
        end

        function done = check_if_done(mdp)
            % Returns true if the MDP is in an unrecoverable state. Otherwise returns false.
            qvel_qpos = [mj('get', 'qpos'); mj('get', 'qvel')];
            done = any(isinf(qvel_qpos)) || abs(qvel_qpos(1)) >= 1 || abs(qvel_qpos(2)) >= pi/3;
        end
    end

    methods (Static)
        function [pos_before, pos_after] = get_hinge_joint(qpos_before, qpos_after)
            pos_before = qpos_before(2);
            pos_after = qpos_after(2);
        end
    end
end
