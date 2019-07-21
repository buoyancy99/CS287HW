close all;

lambda = .01; % TPS regularization coefficient

%%correspondences{1-3}.mat
%%traj{1-3}.mat
load correspondences1
load traj1

[A, B, c] = compute_warp_sol(S, T, lambda);%YOURS TO FILL IN

orig_fig = figure();
warp_fig = figure();

warp = make_warp(A, B, c, S);

figure(orig_fig);
hold on;
scatter(S(1,:), S(2,:), 'red');
scatter(traj(1,:), traj(2,:), 'cyan', 'x');
legend('S', 'traj');

figure(warp_fig);
hold on;
S_warped = warp_pts(S, warp);
warped_traj = warp_pts(traj, warp);
scatter(T(1,:), T(2,:), 50, 'green');
scatter(S_warped(1,:), S_warped(2,:), 50, 'red');
scatter(warped_traj(1,:), warped_traj(2,:), 50, 'cyan', 'x');
draw_grid([0 1], [1 0], warp, 5, orig_fig, warp_fig);
legend('T','f(S)','f(traj)');

%Saving results for traj1, correspondences1, with lambda = .01 
%so you can check your solution
%save('warped_traj1.mat', 'warped_traj')

