% Author: Pieter Abbeel pabbeel@cs.berkeley.edu www.cs.berkeley.edu/~pabbeel
% 2009/11/07, last mod: 2012/09/05

%  the data you obtain stored in a set of global variables from human play: 
%        * chosen_map_log: a log containing all the board configurations that
%             were chosen during human play
%             chosen_map_log{k} will contain the k'th board configuration
%             encountered during play
%        * options_map_log: a log containing all the board configurations
%             that were available as a next state during human play
%             options_map_log{k}{i}: i indexes over all board situations
%             that were available to the player as next states; the one the
%             player chose is stored in chosen_map_log{k}
%        -- this means that for Bellman back-ups / LP equations we 'd use
%        chosen_map_log{k} on the Left Hand Side, and
%        options_map_log{k+1}{i} on the Right Hand Side
%         Note: the human player can choose actions not considered
%         available to the computer player.  I.e., the computer player is
%         supposed to choose a translation and a rotation, and then simply
%         drop the block straight down.  As a consequence, it is possible
%         that the chosen_map_log{k} map is not equal to
%         options_map_log{k}{i} for any i.
%         Note2: when calling mtetris a 2nd, 3rd, etc. time, the new maps
%         will be appended to the back of the maps already stored in
%         options_map_log and chosen_map_log.
%
%
% if you have a mex compiler, start by running the following commands (it will ensure some calls will
% be executed as compiled c-code, rather than interpreted matlab code, this
% will speed up your runs)
%mex tetris_place_block.c
%mex tetris_standard_22_features.c
%mex tetris_2_features.c


% play:
global options_map_log chosen_map_log map_idx board_data% global variables recording tetris board situations, see above for details

board_data = init_board_data();

load tetris_game_log

% let's subsample states to be used in the back-up equation b/c states that are very similar are not that
% informative, and too many states makes things too slow

subsample_interval = 10;

k=1;
for b=1:length(chosen_map_log)
    if(mod(b,subsample_interval)==0)
        backup_maps{k} = chosen_map_log{b};
        k=k+1;
    end 
end

% we'll use the standard feature function
feature_f = @tetris_standard_22_features;


for b=1:length(backup_maps)
    backup_states_phi(:,b) = feature_f(backup_maps{b});
end


discount = .9;

% compute all next states, and the rewards associated with it:
for b=1:length(backup_maps)
    for block_idx=1:7
        i = 1;
        for rotation = 0:3
            for translation = 1:10
                [map1,game_over] = tetris_place_block(backup_maps{b}, block_idx, translation, rotation, board_data);
                phis_next{b}{block_idx}(:,i) = feature_f(map1);
                if(game_over)
                    R{b}{block_idx}(i,1) = 0; %MIN_REWARD/(1-discount);
                else
                    R{b}{block_idx}(i,1) = tetris_reward(map1); %tetris_reward(backup_maps{s});
                end
                i = i+1;
            end
        end
    end
end


% YOUR CODE
%  set up the approximate linear program and find the approximate value
%  function
	  

% 

board_data = init_board_data();

for i=1:20
    [scores(i), all_scores{i}, avg_num_blocks_placed(i)] = main_evaluate_theta(theta, feature_f, i+10, board_data);
end









return;



