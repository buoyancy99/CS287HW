classdef hopper_policy < mujoco_policy
    % Implements a neural network policy for the MuJoCo Walker.
    % See the mujoco_policy class for detailed comments on class properties and methods.

    properties
    end
    
    methods
        function policy = hopper_policy(input_dim, output_dim, action_limits)
            policy = policy@mujoco_policy(input_dim, output_dim, action_limits)
            policy.scale_factor = 100;
        end

        function net = construct_net(policy, input_dim, output_dim)
            % Create the neural net by defining each layer
            % Feel free to experiment with changing the number of layers, or
            % replacing some (or all) of the tanh layers with rectifier layers.
            layers = {};
            layers{1} = affine_layer(32);
            layers{2} = tanh_layer();
            layers{3} = affine_layer(32);
            layers{4} = tanh_layer();
            layers{5} = affine_layer(output_dim);
            layers{6} = euclidean_loss_layer();
            net = neural_network(layers, input_dim);
        end
    end
end
