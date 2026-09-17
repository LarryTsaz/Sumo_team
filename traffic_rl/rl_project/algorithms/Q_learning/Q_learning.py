import numpy as np

def Q_update(Q, S, A, S_next, R_next, discount_factor, learning_rate):
    Q_new = Q[S][A] + learning_rate * ( R_next + discount_factor*np.max(Q[S_next]) - Q[S][A] )
    return Q_new

def Q_learning_algorithm(P, R, policy_fn, discount_factor, learning_rate, Max_iteration):
    
    #initialize
    n_states = len(P[0])
    n_actions = len(P)
    #iteration
    iteration = 0
    #Q-table
    Q = np.zeros((n_states, n_actions), dtype = float)
    # the initial state
    current_state = 0
    # the initial action under given policy function
    current_action = policy_fn(Q[current_state])
    #history
    Q_history = {s:{ a: [] for a in range(n_actions)} for s in range(n_states)}

    #Main loop
    while iteration < Max_iteration:
        iteration += 1
        #sampling next state
        S_next = np.random.choice(n_states, p = P[current_action][current_state])
        #sampling instant reward
        R_mean = R[current_action][current_state][S_next]
        R_next = np.random.normal(R_mean, scale = 1)
        #behavior policy
        A_next = policy_fn(Q[S_next])
        #Q_update
        Q_new = Q_update(Q, current_state, current_action, S_next, R_next, discount_factor, learning_rate)
        #update_Q-table
        Q[current_state][current_action] = Q_new
        #store action value
        Q_history[current_state][current_action].append(Q_new)
        #update current_state
        current_state = S_next
        #update current_action
        current_action = A_next
    
    return Q, Q_history