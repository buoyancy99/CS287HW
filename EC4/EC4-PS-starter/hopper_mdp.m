classdef hopper_mdp < mujoco_mdp
    % Implements a MDP for the MuJoCo Hopper.
    % See the mujoco_mdp class for detailed comments on class properties and methods.

    properties
    end
    
    methods
        function mdp = hopper_mdp(model_file)
            mdp = mdp@mujoco_mdp(model_file);
            mdp.timestep = 0.02;
            mdp.frame_skip = 5;
        end

        function ob = get_observation(mdp)
            % Converts the current MuJoCo simulator state into a MDP state
            ob = [mj('get', 'qpos'); ...
                  max(-10, min(10, mj('get', 'qvel'))); ...
                  max(-10, min(10, mj('get', 'qfrc_constraint')))];
        end

        function reward = get_reward(mdp, xpos_before, xpos_after, qpos_before, qpos_after)
            % Calculates the reward, where [xpos_before, qpos_before] is the previous state
            % and [xpos_after, qpos_after] describes the state after taking the action
            %
            % Below are three ways of calculating the Hopper's reward based on
            % forward movement. In experiments, it seems reward_hip_joint and 
            % reward_hip_pos work better than reward_foot_pos.
            % Feel free to experiment with other reward functions too!

            [hip_joint_before, hip_joint_after] = hopper_mdp.get_hip_joint(qpos_before, qpos_after);
            reward_hip_joint = (hip_joint_after - hip_joint_before) / mdp.timestep;

            [foot_pos_before, foot_pos_after] = hopper_mdp.get_foot_pos(xpos_before, xpos_after);
            foot_pos_init = mdp.init_xpos(5,1);
            reward_foot_pos = foot_pos_after - foot_pos_init;

            [hip_pos_before, hip_pos_after] = hopper_mdp.get_hip_pos(xpos_before, xpos_after);
            reward_hip_pos = hip_pos_after;

            reward = reward_hip_joint;
        end

        function done = check_if_done(mdp)
            % Returns true if the MDP is in an unrecoverable state. Otherwise returns false.
            qvel_qpos = [mj('get', 'qpos'); mj('get', 'qvel')];
            done = any(isinf(qvel_qpos)) || any(abs(qvel_qpos(4:end)) > 100) || ...
                       qvel_qpos(1) <= 0.7 || abs(qvel_qpos(3)) >= 0.2;
        end
    end

    methods (Static)
        function [pos_before, pos_after] = get_hip_pos(xpos_before, xpos_after)
            pos_before = xpos_before(2,1);
            pos_after = xpos_after(2,1);
        end
        function [pos_before, pos_after] = get_foot_pos(xpos_before, xpos_after)
            pos_before = xpos_before(5,1);
            pos_after = xpos_after(5,1);
        end
        function [pos_before, pos_after] = get_hip_joint(qpos_before, qpos_after)
            pos_before = qpos_before(2);
            pos_after = qpos_after(2);
        end
    end
end
