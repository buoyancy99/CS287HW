% value iteration:

function [V, pi] = value_iteration(mdp, precision)

%IN: mdp, precision
%OUT: V, pi

% Recall: to obtain an estimate of the value function within accuracy of
% "precision" it suffices that one of the following conditions is met:
%   (i)  max(abs(V_new-V)) <= precision / (2*gamma/(1-gamma))
%   (ii) gamma^i * Rmax / (1-gamma) <= precision  -- with i the value
%   iteration count, and Rmax = max_{s,a,s'} | R(s,a,s') |

