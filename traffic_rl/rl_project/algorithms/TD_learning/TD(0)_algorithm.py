import numpy as np

def TD_update_fn(V, S, R_next, S_next, discounted_factor, alpha):
    V_new = V[S] + alpha*(R_next+discounted_factor*V[S_next] - V[S])  #if S is at time t
    return V_new

def TD_algorithm_fn(P, R, discounted_factor, learning_rate_fn, Max_iteration = 10000):

    iteration = 0
    current_state = 0

    #-----------------------------------------------------------------------------------------
    Trajectory_of_S = [current_state]
    Trajectory_of_R = {0:[],1:[],2:[]}
    V_history = {0:[],1:[],2:[]}
    V = np.zeros(len(P), dtype = float) #[0, 0, 0]
    #-----------------------------------------------------------------------------------------
    
    while iteration < Max_iteration:
        iteration += 1

        #執行演算法--------------------------------------------------------------------------------------
        S_next = np.random.choice(len(P), p = P[current_state]) #抽樣s'
        R_next = R[current_state][S_next] #抽樣 r when s to s'
        alpha = learning_rate_fn(Trajectory_of_S, current_state)
        V_new = TD_update_fn(V, current_state, R_next, S_next, discounted_factor, alpha) #更新V
        V[current_state] = V_new #存入V(s)
        #--------------------------------------------------------------------------------------

        Trajectory_of_S.append(S_next)
        Trajectory_of_R[current_state].append(R_next)
        V_history[current_state].append(V_new)

        current_state = S_next #更新目前狀態
    return V, Trajectory_of_S, Trajectory_of_R, V_history

def TD_update(V, S, R_next, S_next, discounted_factor, alpha):
    V_new = V[S] + alpha*(R_next+discounted_factor*V[S_next] - V[S])  #if S is at time t
    return V_new

def TD_algorithm(P, R, discounted_factor, alpha, Max_iteration = 10000):

    iteration = 0
    current_state = 0

    #-----------------------------------------------------------------------------------------
    Trajectory_of_S = [current_state]
    Trajectory_of_R = {0:[],1:[],2:[]}
    V_history = {0:[],1:[],2:[]}
    V = np.zeros(len(P), dtype = float) #[0, 0, 0]
    #-----------------------------------------------------------------------------------------
    
    while iteration < Max_iteration:
        iteration += 1

        #執行演算法--------------------------------------------------------------------------------------
        S_next = np.random.choice(len(P), p = P[current_state]) #抽樣s'
        R_next = R[current_state][S_next] #抽樣 r when s to s'
        V_new = TD_update(V, current_state, R_next, S_next, discounted_factor, alpha) #更新V
        V[current_state] = V_new #存入V(s)
        #--------------------------------------------------------------------------------------

        Trajectory_of_S.append(S_next)
        Trajectory_of_R[current_state].append(R_next)
        V_history[current_state].append(V_new)

        current_state = S_next #更新目前狀態
    return V, Trajectory_of_S, Trajectory_of_R, V_history