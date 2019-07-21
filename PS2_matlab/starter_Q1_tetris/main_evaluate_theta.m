% evaluate some theta we found

function [avg_reward_score, reward_scores, avg_blocks_placed, blocks_placed] = ...
    main_evaluate_theta(theta, feature_f, twisterseed, board_data)



total_score = 0;

for seed_idx=1:length(twisterseed)
    max_I = 10000;
    rand('twister', twisterseed(seed_idx));
    map = tetris_init_map(board_data);
    game_over = 0;
    reward_scores(seed_idx) = 0;
    block_idxs = ceil(rand(max_I,1)*7);%new blocks are uniformly at random one of the 7 blocks
    for i=1:max_I
        block_idx = block_idxs(i);
        [translation,rotation] = tetris_greedy_policy(map, block_idx, theta, feature_f, board_data);
        [map,game_over] = tetris_place_block(map, block_idx, translation, rotation, board_data);
        if(game_over)
            reward_scores(seed_idx) = reward_scores(seed_idx) + 0;
            break;
        end
        blocks_placed(seed_idx) = i;
        reward_scores(seed_idx) = reward_scores(seed_idx) + tetris_reward(map);
    end
    %    scores(seed_idx) = i;
end
avg_reward_score = mean(reward_scores);
avg_blocks_placed = mean(blocks_placed);




