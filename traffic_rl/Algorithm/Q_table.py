import numpy as np

class Q_table:
    def __init__(self, 
                 n_states, 
                 n_actions, 
                 init_mean = 0.0,
                 init_std = 0.1,
                 seed = None
                 ):
        """
        Create One Q-table

        Parameters
        -----------
        n_states : int state 數量

        n_actions : int action 數量

        init_mean : float Q-table initialization mean

        init_std : float Q-table initialization std

        seed : int or None random seed

        """

        self.n_states = n_states
        self.n_actions = n_actions

        # randomly initialize Q-table
        rng = np.random.default_rng(seed)
        self.q_table = rng.normal(init_mean, init_std, size = (n_states, n_actions))

    def q_learning_update(self, current_state, current_action, td_target, learning_rate):
        self.q_table[current_state, current_action] += learning_rate * ( td_target - self.q_table[current_state, current_action])
        return self.q_table[current_state, current_action]
