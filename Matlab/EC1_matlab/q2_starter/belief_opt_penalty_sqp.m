function [B, U] = belief_opt_penalty_sqp(B, U, model)

cfg = model.setup_sqp_params();

trust_box_size = cfg.initial_trust_box_size; % The trust region will be a box around the current iterate.
penalty_coeff = cfg.initial_penalty_coeff; % Coefficient of l1 penalties 

% TODO: The outer loop of the sqp algorithm, which repeatedly minimizes
% the merit function --- Calls minimize_merit_function defined below

% After this call, check to see if the
% constraints are satisfied.
% - If some constraint is violated, increase penalty_coeff by a factor of cfg.merit_coeff_increase_ratio
% You should also reset the trust region size to be larger than cfg.min_trust_box_size,
% which is used in the termination condition for the inner loop.
% - If all constraints are satisfied (which in code means if they are satisfied up to tolerance cfg.cnt_tolerance), we're done.

cvx_quiet(true);
cvx_solver SDPT3;

while true    
    
    % YOUR_CODE_HERE
    
end
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Given a belief trajectory and a set of controls, compute merit function value -- Eq. 6 in the problem set
function merit = compute_merit(B, U, model, penalty_coeff)

   % YOUR_CODE_HERE
   
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function [B, U, success] = minimize_merit_function(B, U, model, cfg, penalty_coeff, trust_box_size)

    success = true;
    sqp_iter = 1;
    
    bDim = model.bDim;
    uDim = model.uDim;
    T = model.T;
    
    while  true
        % In this loop, we repeatedly construct a linear approximation to
        % the nonlinear belief dynamics constraint
        fprintf('  sqp iter: %i\n', sqp_iter);
        
		% Compute merit value using the current iterate (trajectory and set of controls)
        merit = compute_merit(B, U, model, penalty_coeff);
        %fprintf('merit: %f\n',merit);
        
        % YOUR_CODE_HERE TO LINEARIZE THE BELIEF DYNAMICS CONSTRAINT (Eq. 5b)
		% Use the numerical_jac routine in q1_starter to compute the required Jacobians numerically
        
        while true 
            % This is the trust region loop
            % Using the approximations computed above, this loop shrinks
            % the trust region until the progress on the approximate merit
            % function is a sufficiently large fraction of the progress on
            % the exact merit function.
            
            fprintf('    trust region size: %.3g\n', trust_box_size);

            % YOUR_CODE_INSIDE cvx_begin and cvx_end BELOW
			
            cvx_begin 
                variables Bcvx(bDim, T) Ucvx(uDim, T-1);
                
                Bcvx(:,1) == B(:,1); % Constraint to ensure that the initial belief remains unchanged

                
                % YOUR_CODE_HERE
                
            cvx_end
            
            if strcmp(cvx_status,'Failed')
                fprintf('Failed to solve QP subproblem.\n');
                success = false;
                return;
            end
            
            model_merit = cvx_optval;
			
			% Compute merit value using the optimized trajectory and set of controls
            new_merit = compute_merit(Bcvx, Ucvx, model, penalty_coeff);
			
			% line search
            approx_merit_improve = merit - model_merit;
            exact_merit_improve = merit - new_merit;
            merit_improve_ratio = exact_merit_improve / approx_merit_improve;
                        
            info = struct('trust_box_size',trust_box_size);
            
            fprintf('      approx improve: %.3g. exact improve: %.3g. ratio: %.3g\n', approx_merit_improve, exact_merit_improve, merit_improve_ratio);
            if approx_merit_improve < -1e-5
                fprintf('Approximate merit function got worse (%.3e).\n',approx_merit_improve);
                fprintf('Either convexification is wrong to zeroth order, or you are in numerical trouble\n');
                success = false;
                return;
            elseif approx_merit_improve < cfg.min_approx_improve
                fprintf('Converged: y tolerance\n');
                B = Bcvx;
                U = Ucvx;
                plot_belief_trajectory(B, U, model);
                pause(0.01);
                return;
            elseif (exact_merit_improve < 1e-2) || (merit_improve_ratio < cfg.improve_ratio_threshold)
                trust_box_size = trust_box_size * cfg.trust_shrink_ratio;
            else
                trust_box_size = trust_box_size * cfg.trust_expand_ratio;
                B = Bcvx;
                U = Ucvx;
                plot_belief_trajectory(B, U, model);
                pause(0.01);
                break; % from trust region loop
            end
            
            if trust_box_size < cfg.min_trust_box_size
                fprintf('Converged: x tolerance\n');
                return;
            end
        end % tr
        sqp_iter = sqp_iter + 1;
    end % sqp

end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%