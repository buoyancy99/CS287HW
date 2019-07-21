close all;

lambda = .01; % TPS regularization coefficient

% scene correspondences
X_s = [1 1; 1 0; 0 0; 0 1]';
X_s_new = [1 1; 1 0; -1 0; 0 1]';

[A, B, c] = compute_warp(X_s, X_s_new, lambda);

orig_fig = figure();
warp_fig = figure();

warp = make_warp(A, B, c, X_s);

figure(orig_fig);
hold on;
scatter(X_s(1,:), X_s(2,:), 'red');
%scatter(X_g(1,:), X_g(2,:), 'cyan', 'x');
legend('x_i^{(S)}');%, 'x_t^{(G)}');

figure(warp_fig);
hold on;
X_s_warped = warp_pts(X_s, warp);
%X_g_warped = warp_pts(X_g, warp);
scatter(X_s_new(1,:), X_s_new(2,:), 50, 'green');
scatter(X_s_warped(1,:), X_s_warped(2,:), 50, 'red');
%scatter(X_g_warped(1,:), X_g_warped(2,:), 50, 'cyan', 'x');
draw_grid([0 1], [1 0], warp, 5, orig_fig, warp_fig);
legend('x_i^{(S)}\prime','f(x_i^{(S)})');%,'f(x_t^{(G)})');
