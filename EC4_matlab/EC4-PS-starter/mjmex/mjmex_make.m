% //-----------------------------------//
% //  This file is part of MuJoCo.     //
% //  Copyright 2009-2015 Roboti LLC.  //
% //-----------------------------------//
function mjmex_make(varargin)

% utility function
prepend = @(prefix,strings) cellfun(@(x)[prefix x],strings,'UniformOutput',false);

% resolve directories
mujoco_root = fileparts(which('mujoco.h'));
if isempty(mujoco_root)
   error('can''t find mujoco headers and library, please make sure they are in your path');
end
here        = pwd;
code_dir    = fileparts(mfilename('fullpath'));

% go to mujoco_root
if ~strcmp(here,mujoco_root)
	cd(mujoco_root);
end

% clear mj from memory, delete it
if exist('mj')==3 %#ok<EXIST>
    try
        mj('clear');
    catch err
       disp(err.message);
    end
    delete(which('mj'))
end

% resolve compiler
compiler = mex.getCompilerConfigurations('C','Selected');

% compiler optimizations
if nargin 
    options = ''; %if there any additional options (e.g. '-g'), don't optimize
else 
    if strcmp(compiler(1).Name,'Intel C++')
        options = 'COMPFLAGS=" /MP /Qopenmp /QxHost $COMPFLAGS"';
    else
        options = 'COMPFLAGS="/openmp /Ox $COMPFLAGS"';
    end
end

% cross-plarform directives
binary_target = [mujoco_root '/mj'];
cross_plat     = {'-D_MEX','-D_MJPRO_LIB','-output',binary_target,'-largeArrayDims'};

% include directories
include        = {['-I' mujoco_root]};

% mex files 
mex_compile = {[code_dir '/mjmex_main.cpp'], [code_dir '/mjmex_plot.cpp']};

% platform-specific additions
libraries = {['-L' mujoco_root]};
if ismac
    libraries = {libraries{:} , '-lmujoco', '-lglfw'};
    glfwname = [' -install_name="' mujoco_root '/libglfw.3.dylib" '];
    mujoconame = [' -install_name="' mujoco_root '/libmujoco.dylib" '];
    platform = {'-D_NIX', 'CXXFLAGS="\$CXXFLAGS -std=c++11 "' ,...
                'CFLAGS="\$CFLAGS"',...
                ['LDFLAGS="\$LDFLAGS' glfwname mujoconame '"']}; 
elseif  isunix
    libraries = {libraries{:} , '-lmujoco', '-lglfw'};
	platform = {'-D_NIX', 'CFLAGS="\$CFLAGS -fopenmp" CXXFLAGS="\$CXXFLAGS -fopenmp" LDFLAGS="\$LDFLAGS -fopenmp"'};
else
    libraries = {libraries{:}, which('mujoco.lib'), which('glfw3.lib')};
	platform = {'-DWIN32', 'CXXFLAGS="\$CXXFLAGS -std=c++11 "'};
end

% assemble compilation string
command = sprintf('%-50s...\n', varargin{:},...
    mex_compile{:},cross_plat{:},platform{:},include{:}, libraries{:}, options);
disp('          ====== mex string =======')
disp(command)
tic;
% compile and go back to the calling directory
try
	eval(['mex ' command]);
    
    if ismac % hack to fix OSX linking issues. surely there is a better way
        fromto = {['/usr/local/glfw/src/libglfw.3.dylib ' mujoco_root '/libglfw.3.dylib'],...
                   ['@executable_path/libmujoco.dylib ' mujoco_root '/libmujoco.dylib']};
        for i=1:length(fromto)
            [status,result] = unix(['install_name_tool -change ' fromto{i} ' mj.mexmaci64']);
            if status
                disp(result)
            end
        end
    end
    mj('version');
    
catch err
	%clc
	disp(err.message)
end

cd(here);
toc

