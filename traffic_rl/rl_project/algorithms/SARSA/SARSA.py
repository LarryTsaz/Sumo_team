import numpy as np

def sarsa_update(Q, S, A, R_next, S_next, A_next, discount_factor, learning_rate):
    Q[S][A] = Q[S][A] + learning_rate * (R_next + discount_factor * Q[S_next][A_next] - Q[S][A])
    return Q[S][A]

def sarsa_algorithm(P, R, Policy_fn, discount_factor, learning_rate, Max_iteration):

    #initialize
    Q = np.zeros((len(P[0]), len(P)), dtype = float)
    iteration = 0
    # the initial state is 0
    current_state = 0
    # the initial action is random explore
    current_action = Policy_fn(Q[current_state])
    # store the history of every action value
    Q_history = {0:{0:[],1:[],2:[]},
                 1:{0:[],1:[],2:[]},
                 2:{0:[],1:[],2:[]}}

    #main loop
    while iteration < Max_iteration:
        iteration += 1
        
        #sampling next state
        S_next = np.random.choice(len(P[0]) , p = P[current_action][current_state])
        #sampling next action under policy
        A_next = Policy_fn(Q[S_next])
        #sampling reward under given current action, current state and next state
        R_mean = R[current_action][current_state][S_next]
        R_next = np.random.normal(R_mean, scale = 1)
        #sarsa update
        Q[current_state][current_action] = sarsa_update(Q, current_state , current_action, R_next, S_next, A_next, discount_factor, learning_rate)
        #store the action value
        Q_history[current_state][current_action].append(Q[current_state][current_action])

        current_state = S_next
        current_action = A_next

    return Q, Q_history