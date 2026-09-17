import numpy as np
from .Q_table import Q_table

class Q_ensemble:
    def __init__(self, n_states, n_actions, ensemble_size, init_mean = 0.0, init_std = 0.1, seed = None):
        self.n_states = n_states
        self.n_actions = n_actions
        self.ensemble_size = ensemble_size # create M 張 Q-table

        # 建立 M 張 randomly initialize Q_table
        self.q_learners = [] # 存 M 個 Q_table class
        for i in range(ensemble_size):
            q_learner = Q_table(n_states, n_actions, init_mean, init_std, seed = None if seed is None else seed + i) # 一個q_learner是一個Q_table class
            self.q_learners.append(q_learner)

    def calculate_mean(self, state):

        q_values = []

        for q_learner in self.q_learners:
            q_values.append(q_learner.q_table[state]) #維度(M , |A|)

        # 轉乘array型式，可以用np.mean
        q_values = np.array(q_values)
        mean_q_values_over_M_tables = np.mean(q_values, axis = 0) #平均是除以 M 張表所以在維度0的地方

        return mean_q_values_over_M_tables

    def calculate_variance(self, state):

        q_values = []
        
        for q_learner in self.q_learners:
            q_values.append(q_learner.q_table[state]) #維度(M , |A|)
        
        # 轉乘array型式，可以用np.mean
        q_values = np.array(q_values)
        variance_q_value_over_M_tables = np.var(q_values, axis= 0 , ddof= 0) # 論文是除以M (Not M-1) 所以自由度=0

        return variance_q_value_over_M_tables

    def update_all(self, current_state, current_action, td_target, learning_rate):

        for q_learner in self.q_learners:
            q_learner.q_learning_update(current_state, current_action, td_target, learning_rate)
