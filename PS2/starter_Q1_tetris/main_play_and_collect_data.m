% Author: Pieter Abbeel pabbeel@cs.berkeley.edu www.cs.berkeley.edu/~pabbeel
% 2009/11/07, last mod: 2012/09/05

% This file is not necessarily intended to be starter code for any of the questions.
% Its sole purpose is to illustrate some of the functions provided to you.
% In particular, this file illustrates 
% (a) how you can run an instantiation of
%     tetris for human play 
%   Controls (use numpad):
%    [4] Left  [5] Spin  [6] Right
%              [2] Drop
%    alternatively,
%    [Q] Left  [W] Spin  [E] Right
%              [S] Drop
%
%    [Ctrl+P] Pause/Unpause
% (b) the data you obtain stored in a set of global variables from human play: 
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

 mtetris % repeat as many games as you like
% now options_map_log and chosen_map_log will contain data

