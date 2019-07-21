% //-----------------------------------//
% //  This file is part of MuJoCo.     //
% //  Copyright 2009-2015 Roboti LLC.  //
% //-----------------------------------//

modelfile = 'humanoid.xml';

%% license management
for i = 1:10
    mj('activate', which('mjkey.txt'));
    mj deactivate
end
disp('License test passed.') 
mj('activate', which('mjkey.txt'))


%% lots of  loading and clearing
for i = 1:100
    
    if mj('ismodel')
        mj clear
    else
        mj('load',which(modelfile))
    end
    
end


%% get, set, get, compare
mj clear
mj('load',which(modelfile));
m = mj('getmodel');
mj('setmodel',m)
m2 = mj('getmodel');
if ~isequal(m,m2)
    error('getmodel, setmodel, getmodel failed.')
else
   disp('test passed.') 
end


%% get, modify, set using 'setmodel', get, compare
mj clear
mj('load',which(modelfile));
m = mj('getmodel');
settablefields = mj('setmodelfield');
for i = 1:length(settablefields)
    field = settablefields{i};
    if isfield(m, field)
        m.(field) = m.(field) + .001*randn(size(m.(field)));
    end
end
mj('setmodel',m)
m2 = mj('getmodel');
if ~isequal(m,m2)
    error('get, modify, set using ''setmodel'', get, compare failed')
else
   disp('test passed.') 
end


%% get, modify, set using 'setmodelfield', get, compare
mj clear
mj('load',which(modelfile));
m = mj('getmodel');
settablefields = mj('setmodelfield');
for i = 1:length(settablefields)
    field = settablefields{i};
    if isfield(m, field)
        value = m.(field) + .001*randn(size(m.(field)));
        m.(field) = value;
        mj('setmodelfield',field,value)
    end
end
m2 = mj('getmodel');
if ~isequal(m,m2)
    error('get, modify, set using ''setmodelfield'', get, compare failed')
else
   disp('test passed.') 
end


%% get, modify, set compare option
option = mj('getoption');
disable_warmstart = 2^10;
option.disableflags = bitor(option.disableflags, disable_warmstart);
mj('setoption', option);
option2 = mj('getoption');
if ~isequal(option,option2)
    error('get, modify, set option failed')
else
   disp('test passed.') 
end


%% test multithreaded step
mj clear
mj('load',which(modelfile));

% disable warmstarts for exact comparison
option = mj('getoption');
disable_warmstart = 2^10;
option.disableflags = bitor(option.disableflags, disable_warmstart);
mj('setoption', option);

m = mj('getmodel');
N = 5000;
nx = m.nq + m.nv + m.na;
nu = m.nu;
X = randn(nx, N);
U = randn(nu, N);
Y = zeros(size(X));
tic
for i=1:N
    mj('set','qpos',X(1:m.nq,i),...
             'qvel',X(m.nq+(1:m.nv),i),...
             'act',X(m.nq+m.nv+(1:m.na),i),...
             'ctrl',U(:,i));
    mj step
    [q,v,a] = mj('get','qpos','qvel','act');
    Y(:,i) = [q;v;a];
end
tsingle = toc;

tic
Y2 = mj('step',X,U);
tthreaded = toc;

if norm(Y-Y2)
    error('mismatch between singlethreaded and multithreaded')
else
   disp('Parallel-step test passed.') 
end

fprintf('=== timing ===\nsingle thread: %5.2f ms/step\nmultithreaded: %5.2f ms/step\n',...
    1e3*tsingle/N,1e3*tthreaded/N)


%% Finalize
mj clear
mj deactivate
disp('all tests passed')