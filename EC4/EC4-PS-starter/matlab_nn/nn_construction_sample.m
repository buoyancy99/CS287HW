
layers = {};

% the sequence of layers executed
% inputs to affine: doutput, w_2reg, w_1reg
layers{1} = affine_layer(15);  % SH
layers{2} = rectifier_layer(); % rectifier 
layers{3} = affine_layer(10);  % SH
layers{4} = rectifier_layer();
layers{5} = affine_layer(7);  % SH
% last layer has to be a loss to compute gradients
layers{6} = euclidean_loss_layer();

net = neural_network(layers, 7);  % SH

% input and bool for result = cell list of layers outputs
input = zeros(7,1);  % SH
result = net.forward(input, false); 

loss = net.loss(input, target, false);
% gradlosses is cell list of gradient
[loss, gradlosses] = net.forward_backward(input, target, false); 

% sample gradient descent update:
gradient_weight = 0.1;
for l = 1:size(layers, 2)
    if size(gradlosses{l}, 1) > 0   % SH
        new_paramvec = net.layers{l}.get_paramvec() - gradient_weight * gradlosses{l}; %SH
        net.layers{l}.set_paramvec(new_paramvec); %SH
    end
end

