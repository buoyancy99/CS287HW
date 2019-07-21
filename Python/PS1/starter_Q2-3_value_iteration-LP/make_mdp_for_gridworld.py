#mdp:
#T[action}(current_state, next_state) : transition model
#R(current_state, action, next_state) : reward
#gamma: discount factor
import numpy as np

def make_mdp_for_gridworld(p_noise, gamma):
    #  p_noise: probability that action fails

    # states:   
    #  7  8  9 10  
    #  4  x  5  6
    #  0  1  2  3
    
    #  11 = sink state

    # position x is occupied by a wall and can never be visited
    # Actions: N 0, E 1, S 2, W 3

    T = np.zeros((4,12,12))

    T[0,0,4] = 1 - p_noise
    T[0,0,1] = p_noise/2
    T[0,0,0] = p_noise/2
    T[1,0,1] = 1 - p_noise
    T[1,0,4] = p_noise/2
    T[1,0,0] = p_noise/2
    T[2,0,0] = 1 - p_noise + p_noise/2
    T[2,0,1] = p_noise/2
    T[3,0,0] = 1 - p_noise + p_noise/2
    T[3,0,4] = p_noise/2

    T[0,1,1] = 1 - p_noise
    T[0,1,0] = p_noise/2
    T[0,1,2] = p_noise/2
    T[1,1,2] = 1 - p_noise
    T[1,1,1] = p_noise
    T[2,1,1] = 1 - p_noise
    T[2,1,0] = p_noise/2
    T[2,1,2] = p_noise/2
    T[3,1,0] = 1 - p_noise
    T[3,1,1] = p_noise

    T[0,2,5] = 1 - p_noise
    T[0,2,1] = p_noise/2
    T[0,2,3] = p_noise/2
    T[1,2,3] = 1 - p_noise
    T[1,2,2] = p_noise/2
    T[1,2,5] = p_noise/2
    T[2,2,2] = 1 - p_noise
    T[2,2,1] = p_noise/2
    T[2,2,3] = p_noise/2
    T[3,2,1] = 1 - p_noise
    T[3,2,5] = p_noise/2
    T[3,2,2] = p_noise/2

    T[0,3,6] = 1 - p_noise
    T[0,3,2] = p_noise/2
    T[0,3,3] = p_noise/2
    T[1,3,3] = 1 - p_noise + p_noise/2
    T[1,3,6] = p_noise/2
    T[2,3,3] = 1 - p_noise + p_noise/2
    T[2,3,2] = p_noise/2
    T[3,3,2] = 1 - p_noise
    T[3,3,3] = p_noise/2
    T[3,3,6] = p_noise/2

    T[0,4,7] = 1 - p_noise
    T[0,4,4] = p_noise
    T[1,4,4] = 1 - p_noise
    T[1,4,7] = p_noise/2
    T[1,4,0] = p_noise/2
    T[2,4,0] = 1 - p_noise
    T[2,4,4] = p_noise
    T[3,4,4] = 1 - p_noise
    T[3,4,7] = p_noise/2
    T[3,4,0] = p_noise/2

    T[0,5,9] = 1 - p_noise
    T[0,5,5] = p_noise/2
    T[0,5,6] = p_noise/2
    T[1,5,6] = 1-p_noise
    T[1,5,9] = p_noise/2
    T[1,5,2] = p_noise/2
    T[2,5,2] = 1 - p_noise
    T[2,5,5] = p_noise/2
    T[2,5,6] = p_noise/2
    T[3,5,5] = 1 - p_noise
    T[3,5,2] = p_noise/2
    T[3,5,9] = p_noise/2


    T[0,6,11] = 1
    T[1,6,11] = 1
    T[2,6,11] = 1
    T[3,6,11] = 1

    T[0,7,7] = 1 - p_noise + p_noise/2
    T[0,7,8] = p_noise/2
    T[1,7,8] = 1 - p_noise
    T[1,7,7] = p_noise/2
    T[1,7,4] = p_noise/2
    T[2,7,4] = 1 - p_noise
    T[2,7,8] = p_noise/2
    T[2,7,7] = p_noise/2
    T[3,7,7] = 1 - p_noise + p_noise/2
    T[3,7,4] = p_noise/2

    T[0,8,8] = 1 - p_noise
    T[0,8,7] = p_noise/2
    T[0,8,9] = p_noise/2
    T[1,8,9] = 1 - p_noise
    T[1,8,8] = p_noise
    T[2,8,8] = 1 - p_noise
    T[2,8,9] = p_noise/2
    T[2,8,7] = p_noise/2
    T[3,8,7] = 1 - p_noise
    T[3,8,8] = p_noise

    T[0,9,9] = 1 - p_noise  
    T[0,9,8] = p_noise/2 
    T[0,9,10] = p_noise/2
    T[1,9,10] = 1 - p_noise 
    T[1,9,5] = p_noise/2 
    T[1,9,9] = p_noise/2
    T[2,9,5] = 1 - p_noise 
    T[2,9,8] = p_noise/2 
    T[2,9,10] = p_noise/2
    T[3,9,8] = 1 - p_noise 
    T[3,9,9] = p_noise/2 
    T[3,9,5] = p_noise/2

    T[0,10,11] = 1
    T[1,10,11] = 1
    T[2,10,11] = 1
    T[3,10,11] = 1

    T[0,11,11] = 1
    T[1,11,11] = 1
    T[2,11,11] = 1
    T[3,11,11] = 1


    R = np.zeros((4,12,12))

    R[0,10,11] = 1
    R[1,10,11] = 1
    R[2,10,11] = 1
    R[3,10,11] = 1
    R[0,6,11] = -1
    R[1,6,11] = -1
    R[2,6,11] = -1
    R[3,6,11] = -1


    gridworld_mdp = {}
    gridworld_mdp['T'] = T
    gridworld_mdp['R'] = R
    gridworld_mdp['gamma'] = gamma

    return gridworld_mdp
