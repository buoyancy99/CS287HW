classdef tanh_layer < layer

    properties
    end

    methods
        function layer = tanh_layer()
            layer.type = 'tanh';
        end

        function result = forward(e, input)
            exp_2input = exp(2 .* input);
            result = (exp_2input - 1) ./ (exp_2input + 1);
        end

        function result = forward_mat(e, input)
            result = e.forward(input);
        end

        function result = backward(e, input, gradoutput)
            result = {gradoutput .* (1 - (tanh(input) .^ 2)), []};
        end

        function result = backward_pg(e, input, gradoutput)
            result = {gradoutput * diag(1 - (tanh(input) .^ 2)), []};
        end

        function result = backward_pg_mat(e, input, gradoutput)
            % gradoutput should either be 2D (XxY) matrix or 3D (XxYxT) matrix
            % result will be a 3D matrix, XxZxT
            d_input = 1 - (tanh(input) .^ 2);
            d_input = reshape(d_input, [1,size(d_input,1),size(d_input,2)]);
            result = {bsxfun(@times, gradoutput, d_input), []};
        end

        function e = change_dimensions(e, Di)
            e.Di = Di;
            e.Do = Di;
            e.initialized = true;
        end

        function paramvec = get_paramvec(e)
            paramvec = [];
        end

        function e = set_paramvec(e, paramvec)
        
        end
    end
end
