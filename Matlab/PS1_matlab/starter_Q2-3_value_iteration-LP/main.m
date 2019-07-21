p_noise=0.2; gamma=0.99;
gridworld_mdp = make_mdp_for_gridworld(p_noise, gamma);

precision = 1e-2;

[V, pi] = value_iteration(gridworld_mdp, precision);


[V2, pi2] = linear_programming(gridworld_mdp); %note: LP by default only gives V, do tiny bit of extra work after having found LP solution to find pi

% save gridworld_V_pi_V2_pi2.mat V pi V2 pi2
% --> load gridworld_V_pi_V2_pi2.mat to verify your results


queue_mdp = make_mdp_3queues([0.1, 0.2, 0.5], [1 1 1], 5, 0.9);
[V3, pi3] = value_iteration(queue_mdp, precision);
[V4, pi4] = linear_programming(queue_mdp);

% to verify your results:
%
% V4 for (2, 2, 2, a)  = 8.3870
% V4 for (0, 5, 2, a)  = 8.0874
% V4 for (5, 5, 0, c)  = 8.1466
 
