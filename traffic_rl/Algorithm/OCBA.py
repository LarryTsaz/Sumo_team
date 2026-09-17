from scipy.stats import norm
import math
import numpy as np

class OCBA:
    def allocation(self, mean_of_design, variance_of_design, old_allocation, budget):
        
        n_designs = len(mean_of_design) # k = ?

        #除錯
        if n_designs == 0 :
            raise ValueError("mean_of_design 不能是空的")

        if len(variance_of_design) != n_designs:
            raise ValueError("mean_of_design 與 variance_of_design 長度不一")

        if len(old_allocation) != n_designs:
            raise ValueError("old_allocation 與 mean_of_design 長度不一")

        if budget < 0 :
            raise ValueError("budget 不可為負")

        if any(n < 0 for n in old_allocation):
            raise ValueError("old_allocation 不可為負")

        if any(var < 0 for var in variance_of_design):
            raise ValueError("variance 不可為負")

        #Initialize

        allocation = [0]* n_designs

        total_simulation_time = (sum(old_allocation) + budget)

        # find the best design

        current_optimal_design = self.current_best(mean_of_design)

        allocation_ratio = self.calculate_allocation_ratio(mean_of_design, variance_of_design)

        # if all of the ration is 0
        if sum(allocation_ratio) <= 1e-12 :
            additional_allocation = [0] * n_designs
            additional_allocation[current_optimal_design] = budget
            return additional_allocation

        # 開始根據allocation ratio 分配理論上的simulation_allocation budget
        is_run = [1] * n_designs
        temp_simulation_time = total_simulation_time

        while True:
            reallocate = False

            summation_of_allocation = 0.0
            for i in range(n_designs):
                if is_run[i] == 1 :
                    summation_of_allocation += allocation_ratio[i]

            if summation_of_allocation <= 1e-12 :
                break

            # calculate theoretic total allocation

            for i in range(n_designs):
                if is_run[i] == 1:
                    allocation[i] = int((allocation_ratio[i]/summation_of_allocation) * temp_simulation_time)

                if(is_run[i] == 1 and allocation[i] < old_allocation[i]):
                    reallocate = True
                    allocation[i] = old_allocation[i]
                    is_run[i] = 0

            if reallocate:
                temp_simulation_time = total_simulation_time

                for i in range(n_designs):
                    if is_run[i] == 0:
                        temp_simulation_time -= allocation[i]

            if not reallocate:
                break

        current_total_allocation = sum(allocation)

        remaining_simulation = total_simulation_time - current_total_allocation
        allocation[current_optimal_design] += remaining_simulation

        # calculate final additional_allocation
        final_additional_allocation = [0] * n_designs

        for i in range(n_designs):
            final_additional_allocation[i] = allocation[i] - old_allocation[i]

        return final_additional_allocation

    def current_best(self, data):
        return int(np.argmax(data))

    def calculate_allocation_ratio(self, mean_of_design, variance_of_design):
        
        n_designs = len(mean_of_design)

        allocation_ratio = [0.0] * n_designs

        mean_difference_with_best_design = [0.0] *n_designs

        current_optimal_design = self.current_best(mean_of_design)

        for i in range (n_designs):

            mean_difference_with_best_design[i] = (mean_of_design[current_optimal_design] - mean_of_design[i])

        # calculate non-best ratio
        # ratio_non_best = variance / delta^2

        sum_ratio_squared_over_variance = 0.0

        for i in range(n_designs):
            if i != current_optimal_design:
                allocation_ratio[i] = variance_of_design[i] / max( math.pow(mean_difference_with_best_design[i], 2), 1e-12 )
                sum_ratio_squared_over_variance += ( math.pow(allocation_ratio[i], 2)/ max( variance_of_design[i], 1e-12) )

        allocation_ratio[current_optimal_design] = math.sqrt( max( variance_of_design[current_optimal_design], 1e-12) * sum_ratio_squared_over_variance)

        return allocation_ratio

    def calculate_allocation_proportion(self, mean_of_design, variance_of_design):
        
        allocation_ratio = self.calculate_allocation_ratio(mean_of_design, variance_of_design)

        total_ratio = sum(allocation_ratio)

        allocation_proportion = [ratio/total_ratio for ratio in allocation_ratio]

        return allocation_proportion

    def calculate_APCS(self, mean_of_design, variance_of_design):
            """
            Parameters
            ----------
            mean_of_design : list[float] 每個 action 的 ensemble Q mean
    
            variance_of_design : list[float] 每個 action 的 ensemble Q variance
    
            Returns
            -------
            apcs : float Approximate Probability of Correct Selection / decision confidence
            """
            n_designs = len(mean_of_design)
    
            current_optimal_design = self.current_best(mean_of_design)
    
            sum_probability = 0.0
    
            for i in range(n_designs):
                if i != current_optimal_design:
                    # delta = mean_best - mean_i
                    delta = mean_of_design[current_optimal_design] - mean_of_design[i]
                    var_delta = variance_of_design[current_optimal_design] + variance_of_design[i]
    
                    probability = norm.cdf(-delta/max(math.sqrt(var_delta) , 1e-12))
                    sum_probability += probability
    
                apcs = 1 - sum_probability
    
            return apcs
