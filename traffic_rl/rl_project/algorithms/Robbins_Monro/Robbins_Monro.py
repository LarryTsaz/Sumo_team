import numpy as np

def Robbins_Monro_update(new_observation, theta_old, coefficient):
    theta_new = theta_old + coefficient*(new_observation - theta_old)
    return theta_new

# if coefficient is constant
def Robbins_Monro_c(mean, std, coefficient, Max_iterations = 10000):
     
    theta_old = 0
    iteration = 0
    sampling_list = []
     
    while iteration < Max_iterations:
        new_observation = np.random.normal(mean, std)
        theta_new = Robbins_Monro_update(new_observation, theta_old, coefficient)
        theta_old = theta_new

        sampling_list.append(theta_new)
        iteration += 1
   
    return theta_new, sampling_list

#if coefficient is function
def Robbins_Monro_i(mean, std, coeffecient_fn, Max_iterations):
    
    theta_old = 0
    sampling_list = []
    iteration = 0
    
    while iteration < Max_iterations:
        iteration += 1
        new_observation = np.random.normal(mean, std)

        coefficient = coeffecient_fn(iteration)
        theta_new = Robbins_Monro_update(new_observation, theta_old, coefficient)
        theta_old = theta_new

        sampling_list.append(theta_new)
    
    return theta_new, sampling_list

#list of function of coefficient
def coeffecient_fn1(n):
    return 1/(n+1)

def coeffecient_fn2(n):
    return 0.5/(n+1)

def coeffecient_fn3(n):
    return 1/((n+1) ** 0.7)