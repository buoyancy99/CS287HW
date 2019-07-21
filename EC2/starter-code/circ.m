function X = circ(arc_length, start_angle, n_pts, center, radius)
    theta = arc_length/n_pts;
    R = [cos(theta) sin(theta); -sin(theta) cos(theta)];
    start = radius*[cos(start_angle);sin(start_angle)];
    X = zeros(2, n_pts);
    X(:,1) = start;
    for i = 2:n_pts
        X(:,i) = R*X(:,i-1);
    end
    X = X + repmat(center, 1, n_pts);   
end