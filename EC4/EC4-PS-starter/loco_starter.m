function [ ] = loco_starter( )
    % This function creates MDP and policy objects for a given problem, and
    % calls the following three policy search methods (which you will get to
    % implement the key parts of):
    %     1. Cross Entropy Method
    %     2. Covariance Matrix Adaptation
    %     3. Proximal Policy Optimization (PPO)
    %
    % Note: Uncomment the 'mj plot' line only if you are running MATLAB with
    % the -nodesktop flag turned on. Otherwise you'll get the following error:
    %     'X Error of failed request:  BadWindow (invalid Window parameter)'

    clc;
    clear all;
    clear global;
    
    % Add necessary packages in subdirectories to the MATLAB path. Assumes you
    % are running this script while in the parent folder.
    if isunix
        addpath('mujoco_linux')
    elseif ispc
        addpath('mujoco_windows')
    elseif ismac
        addpath('mujoco_osx')
    else
        disp('ERROR: Cannot recognize platform')
    end
    addpath('mjmex')
    addpath('matlab_nn')
    addpath('mtimesx')

    % Activate MuJoCo
    mj('activate', 'MUJOCO_LICENSE.TXT')

    % Construct Cartpole MDP and Cartpole Policy objects
    % Test your code on the Cartpole first, to make sure CEM, CMA, and PPO can
    % all quickly learn to balance the pole.
    % Report results on the Hopper though, not the Cartpole.
    mdp_pole = cartpole_mdp('mujoco_models/cartpole.xml');
    policy_pole = cartpole_policy(mdp_pole.ob_dim, ...
                                  mdp_pole.action_dim, mdp_pole.action_limits);

    % Uncomment 'mj plot' for MuJoCo visualization. Read the note at the top of this file.
    % mj plot
    cross_entropy(mdp_pole, policy_pole);  % YOURS to implement (see cross_entropy_starter.m)
    covariance_matrix_adaptation(mdp_pole, policy_pole);  % YOURS to implement (see covariance_matrix_adaptation_starter.m)
    policy_gradient_ppo(mdp_pole, policy_pole);  % YOURS to implement (see policy_gradient_ppo_starter.m)

    %%%%%%%%%%%%%%%%%%%%%

    % Construct Hopper MDP and Hopper Policy objects
    mdp_loco = hopper_mdp('mujoco_models/hopper.xml');
    policy_loco = hopper_policy(mdp_loco.ob_dim, ...
                                mdp_loco.action_dim, mdp_loco.action_limits);

    % Uncomment 'mj plot' for MuJoCo visualization. Read the note at the top of this file.
    % mj plot
    cross_entropy(mdp_loco, policy_loco);  % YOURS to implement (see cross_entropy_starter.m)
    covariance_matrix_adaptation(mdp_loco, policy_loco);  % YOURS to implement (see covariance_matrix_adaptation_starter.m)
    policy_gradient_ppo(mdp_loco, policy_loco);  % YOURS to implement (see policy_gradient_ppo_starter.m)
end
