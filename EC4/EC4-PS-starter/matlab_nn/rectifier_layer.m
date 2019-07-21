classdef rectifier_layer < layer
    %RECTIFIER_LAYER Summary of this class goes here
    %   Detailed explanation goes here
    
    properties
    end
    
    methods
        function layer = rectifier_layer()
            layer.type = 'rectifier';
        end;
        
        function result = forward(e, input)
            result = max(input, 0);
        end;

        function results = forward_mat(e, input)
            result = e.forward(input)
        end
        
        function result = backward(e, input, gradoutput)
            result = {gradoutput .* (input >= 0), []};
        end

        function result = backward_pg(e, input, gradoutput)
            result = {gradoutput * diag(input >= 0), []};
        end

        function result = backward_pg_mat(e, input, gradoutput)
            % gradoutput should either be 2D (XxY) matrix or 3D (XxYxT) matrix
            % result will be a 3D matrix, XxZxT
            d_input = input >= 0;
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

