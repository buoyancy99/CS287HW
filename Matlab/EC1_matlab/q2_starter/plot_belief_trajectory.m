% Plot belief space trajectory given set of beliefs B and controls U
function plot_belief_trajectory(B, U, model)

xDim = model.xDim;
start = model.start;
goal = model.goal;
T = model.T;

model.plot_domain(B, model);

hold on;
axis equal;

Xt = zeros(xDim,T);

for t=1:T-1
    [Xt(:,t), SqrtSigma_t] = decompose_belief(B(:,t), model);
    Sigma_t = SqrtSigma_t*SqrtSigma_t;
    
    plot_cov(Xt(1:2,t),Sigma_t(1:2,1:2));
end
[Xt(:,T), SqrtSigma_T] = decompose_belief(B(:,T), model);
Sigma_T = SqrtSigma_T*SqrtSigma_T;

plot_cov(Xt(1:2,T),Sigma_T(1:2,1:2));

plot(Xt(1,:),Xt(2,:),'r-','linewidth',3);

for t=1:T
    plot(Xt(1,t),Xt(2,t),'ys','markersize',3,'linewidth',3);
end

plot(start(1),start(2),'cs','markersize',5,'linewidth',5);
plot(goal(1),goal(2),'cs','markersize',5,'linewidth',5);

hold off;
end

% Plot 2D ellipse corresponding to Sigma
function plot_cov(mu,Sigma)
t = -pi:.01:pi;
x = sin(t);
y = cos(t);

[vv,dd] = eigs(Sigma);
A = real((vv*sqrt(dd))');
%A = real((vv*dd)');
z = [x' y']*A;

%plot(mu(1),mu(2),'rX','markersize',10,'linewidth',3);
plot(z(:,1)+mu(1),z(:,2)+mu(2),'Color',[1 0.84314 0],'linewidth',4);
end