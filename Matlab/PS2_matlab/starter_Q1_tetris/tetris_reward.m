function r = tetris_reward(map)


% comment / uncomment appropriately to get your choice of reward

%reward = clearance from top:
phi = tetris_standard_22_features(map);
r = 20-phi(20);



%reward = 1 everywhere
%r = 1;