from traffic_rl.algorithm.OCBA import OCBA
import numpy as np

class Policy:
    def __init__(self):
        self.ocba = OCBA()
        
    def epsilon_OCBA_policy(self, mean_of_design, variance_of_design, epsilon, eta, rng):

        ocba = OCBA()

        n_actions = len(mean_of_design)

        apcs = ocba.calculate_APCS(mean_of_design, variance_of_design)

        if apcs <= eta :
            allocation_proportion = np.array(ocba.calculate_allocation_proportion(mean_of_design, variance_of_design))
            action_probability = (1- epsilon) * allocation_proportion + epsilon / n_actions
        else:
            action_probability = np.full(n_actions, epsilon / n_actions)
            action_probability[ocba.current_best(mean_of_design)] += 1 - epsilon

        action = rng.choice(n_actions, p = action_probability)

        return int(action)

    def epsilon_greedy_policy(self, mean_of_design, epsilon, rng):
        n_actions = len(mean_of_design)

        if rng.random() < epsilon:
            return int(rng.integers(n_actions))

        return int(np.argmax(mean_of_design))
                
        