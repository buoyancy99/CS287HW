% NOT TO BE USED during SQP --- this is only used for verifying correctness of your implementation

% Compute foward simulated cost of belief trajectory given initial belief and set of control inputs (integrated forward)

% After we are done with the optimization, we use this function to evaluate the expected final cost of the trajectory
% Since there might be constraint violations at the end of the optimization, we forward integrate the beliefs using the set of controls obtained from the optimization to compute the final expected cost

function cost = compute_forward_simulated_cost(b1, U, model)

T = model.T;
belief_cost = zeros(1,T);
control_cost = zeros(1,T-1);

b_t = b1;
for t=1:T-1
    [~, SqrtSigma_t] = decompose_belief(b_t, model);
    belief_cost(t) = model.alpha_belief*sum(sum_square(SqrtSigma_t));
    control_cost(t) = model.alpha_control*sum_square(U(:,t));
    b_t = belief_dynamics(b_t, U(:,t), [], model);
end
[~, SqrtSigma_T] = decompose_belief(b_t, model);
belief_cost(T) = model.alpha_final_belief*sum(sum_square(SqrtSigma_T));

%fprintf('Belief cost: %f\n', sum(belief_cost(:)) );
%fprintf('Control cost: %f\n', sum(control_cost(:)) );
cost = sum(belief_cost(:)) + sum(control_cost(:));

end
