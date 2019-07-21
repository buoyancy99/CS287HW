classdef walker_mdp < mujoco_mdp
    % Implements a MDP for the MuJoCo Walker.
    % See the mujoco_mdp class for detailed comments on class properties and methods.

    properties
    end
    
    methods
        function mdp = walker_mdp(model_file)
            mdp = mdp@mujoco_mdp(model_file);
            mdp.timestep = 0.02;
            mdp.frame_skip = 4;
        end

        function ob = get_observation(mdp)
            % Converts the current MuJoCo simulator state into a MDP state
            ob = [mj('get', 'qpos'); ...
                  sign(mj('get', 'qvel')); ...
                  sign(mj('get', 'qfrc_constraint'))];
        end

        function reward = get_reward(mdp, xpos_before, xpos_after, qpos_before, qpos_after)
            % Calculates the reward, where [xpos_before, qpos_before] is the previous state
            % and [xpos_after, qpos_after] describes the state after taking the action
            %
            % There are multiple ways of calculating the Walker's reward based on forward
            % movement. You can experiment with which one works the best!

            [pos_before, pos_after] = mujoco_mdp.get_minpos(xpos_before, xpos_after);
            %[pos_before, pos_after] = walker_mdp.get_hip_pos(xpos_before, xpos_after);
            %[pos_before, pos_after] = walker_mdp.get_feet_pos(xpos_before, xpos_after);
            reward = (pos_after - pos_before) / mdp.timestep + 0.1;
        end

        function done = check_if_done(mdp)
            % Returns true if the MDP is in an unrecoverable state. Otherwise returns false.
            qvel_qpos = [mj('get', 'qpos'); mj('get', 'qvel')];
            done = any(isinf(qvel_qpos)) || any(abs(qvel_qpos(4:end)) > 100) || ...
                       qvel_qpos(1) <= 0.7 || ...
                       abs(qvel_qpos(3)) >= 0.5;
        end
    end

    methods (Static)
        function [pos_before, pos_after] = get_hip_pos(xpos_before, xpos_after)
            pos_before = xpos_before(2,1);
            pos_after = xpos_after(2,1);
        end
        function [pos_before, pos_after] = get_feet_pos(xpos_before, xpos_after)
            pos_before = xpos_before(4,1) + xpos_before(7,1);
            pos_after = xpos_after(4,1) + xpos_after(7,1);
        end
    end
end
