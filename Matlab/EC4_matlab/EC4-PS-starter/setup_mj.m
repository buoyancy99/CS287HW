% Run this on your machine to install the included MuJoCo library and its MATLAB
% interface. You only need to run it once, not every time you re-start MATLAB.

if isunix
    addpath('mujoco_linux')
elseif ispc
    addpath('mujoco_windows')
elseif ismac
    addpath('mujoco_osx')
else
    disp('ERROR: Cannot recognize platform')
end

addpath('mjmex')

mjmex_make
