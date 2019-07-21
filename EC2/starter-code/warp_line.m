function warped = warp_line(start_pt, end_pt, warp, num_samples)
  %WARP_LINE Warps a line segment
  %
  % warp_line(start_pt, end_pt, warp, num_samples)
  %
  % Given the line segment defined by start_pt and end_pt, as well as a warping
  % function warp, returns a data matrix representing the warped curve, where
  % the columns of the matrix represent points along the curve.
  %
  % The warped curve is generated by sampling num_samples points along the
  % curve, then warping each of those points.

  X = linspace(start_pt(1), end_pt(1), num_samples);
  Y = linspace(start_pt(2), end_pt(2), num_samples);
  warped = zeros(2, num_samples);
  for i = 1:num_samples
    warped(:,i) = warp([X(i); Y(i)]);
  end
end
