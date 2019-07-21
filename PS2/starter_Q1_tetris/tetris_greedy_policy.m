function [tout, rout] = tetris_greedy_policy(map, block_idx, theta, feature_f, board_data)

best_val = -Inf;
tout=1; rout=0;

block_rotations{1} = 0:3;
block_rotations{2} = 0:1;
block_rotations{3} = 0:1;
block_rotations{4} = 0:1;
block_rotations{5} = 0;
block_rotations{6} = 0:3;
block_rotations{7} = 0:3;

for translation = 1:10
    for rotation = block_rotations{block_idx}
        [map1,game_over] = tetris_place_block(map, block_idx, translation, rotation,board_data);
        if(~game_over)
            phi_new = feature_f(map1);
            val = theta'*phi_new;
            if(val>best_val)
                best_val = val;
                tout = translation;
                rout = rotation;
            end
        end
    end
end