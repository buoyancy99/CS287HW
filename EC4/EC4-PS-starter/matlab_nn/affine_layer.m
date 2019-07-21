classdef affine_layer < layer
    %AFFINE_LAYER Summary of this class goes here
    %   Detailed explanation goes here
    
    properties
        W
        b
    end
    
    methods
        function layer = affine_layer(Do)
            layer.Do = Do;
            zeros_out = zeros(Do * Do, 1);
            tallw = normrnd(zeros_out, 0.08 * ones(Do * Do, 1));
            layer.W = reshape(tallw, Do, Do);
            layer.b = normrnd(zeros(Do, 1), 0.08 * ones(Do, 1));
            layer.type = 'affine';
        end;
        
        function result = forward(e, input)
            result = e.W * input + e.b;
        end;

        function result = forward_mat(e, input)
            result = bsxfun(@plus, e.W * input, e.b);
        end;
        
        function result = backward(e, input, gradoutput)
            input_bar = e.W' * gradoutput;
            W_bar = gradoutput * input';
            [Do, Di] = size(e.W);
            b_bar = gradoutput; % TODO: fix if N > 1
            paramvec_bar = [reshape(W_bar, Do * Di, 1); b_bar];
            result = {input_bar, paramvec_bar};
        end

        function result = backward_pg(e, input, gradoutput)
            input_bar = gradoutput * e.W;
            [Do, Di] = size(e.W);
            b_bar = gradoutput;
            if size(input, 1) == 1
                input = input';
            end
            W_bar = reshape(repmat(input, [1, Do*Do])', [Do, Do*Di]) .* repmat(eye(Do), [1,Di]);
            W_bar = gradoutput * W_bar;
            parammatrix_bar = [W_bar, b_bar];  % output_dim x (numel(e.W) + length(e.b))
            result = {input_bar, parammatrix_bar};
        end  

        function result = backward_pg_mat(e, input, gradoutput)
            input_bar = mtimesx(gradoutput, e.W);
            [Do, Di] = size(e.W);
            T = size(input,2);
            if size(gradoutput, 3) ~= 1 && size(gradoutput, 3) ~= T
                error('Dimensions of gradoutput in backward pass are incorrect');
            end

            b_bar = gradoutput;
            if size(gradoutput, 3) == 1
                b_bar = repmat(b_bar, 1, 1, T);
            end

            if size(gradoutput, 3) > 1
                gradoutput = reshape(gradoutput, size(gradoutput,1), Do, 1, size(gradoutput,3));
            end

            input_4d = reshape(input, 1, 1, Di, T);
            W_bar = reshape(bsxfun(@times, gradoutput, input_4d), size(gradoutput,1), Do*Di, T);
            parammatrix_bar = [W_bar, b_bar];  % output_dim x (numel(e.W) + length(e.b)) x T
            result = {input_bar, parammatrix_bar};
        end
        
        function e = change_dimensions(e, Di)
            e.Di = Di;
            Do = e.Do;
            zeros_out = zeros(Do * Di, 1);
            tallw = normrnd(zeros_out, 0.08 * ones(Do * Di, 1));
            e.W = reshape(tallw, Do, Di);
            e.b = normrnd(zeros(Do, 1), 0.08 * ones(Do, 1));
            e.initialized = true;
        end
        
        function paramvec = get_paramvec(e)
            paramvec = [reshape(e.W, e.Di * e.Do, 1); e.b];
        end
        
        function e = set_paramvec(e, paramvec)
            e.W = reshape(paramvec(1 : e.Di * e.Do), e.Do, e.Di);
            e.b = paramvec(e.Di * e.Do + 1 : end);
        end

    end
    
end

