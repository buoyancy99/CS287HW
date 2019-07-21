% Author: Pieter Abbeel pabbeel@cs.berkeley.edu www.cs.berkeley.edu/~pabbeel
% 2009/11/07, last mod: 2012/09/05

% if you have a mex compiler, start by running the following commands (it will ensure some calls will
% be executed as compiled c-code, rather than interpreted matlab code, this
% will speed up your runs)
%mex tetris_place_block.c
%mex tetris_standard_22_features.c
%mex tetris_2_features.c
% let's have the computer play

board_data = init_board_data();

map = tetris_init_map(board_data);

[hmap, hfig] = tetris_init_draw(board_data); %inits drawing, we don't use hfig actually

game_over = 0;
max_i=1000;%max number of blocks we will let the computer play for
for i=1:max_i
    block_idx = ceil(rand*7);
    translation_action = ceil(rand*10); %choose from 1 to 10
    rotation_action = ceil(rand*4)-1; % choose from 0 to 3
    [map,game_over] = tetris_place_block(map, block_idx, translation_action, rotation_action, board_data);
    hmap = tetris_draw_now(hmap, map, board_data); %draws the current map; will slow things down, obviously
        %note: occasionally the "tetris_draw_now" fails and complains about an
    %invalide handle; not sure what the deal is; if you find out, let me
    %know; in the meantime, it really should not prevent you from cracking
    %your entire PS b/c most of the time you would not want to draw anyway
    %(b/c it takes time)
    if(game_over)
        break;
    end
    pause(.1); % allows us to watch it play (default policy is not all that clever and interesting to watch though!)
end
fprintf(1,['\n\nthe computer managed to place ' num2str(i) ' blocks onto the board\n\n']);

