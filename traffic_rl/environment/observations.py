from abc import ABC, abstractmethod

import numpy as np
from gymnasium import spaces


class ObservationFunction(ABC):

    def __init__(self, ts):
        self.ts = ts

    @abstractmethod
    def __call__(self):
        pass

    @abstractmethod
    def observation_space(self):
        pass


class DefaultObservationFunction(ObservationFunction):

    def __call__(self):

        elapsed_times = self.ts.get_phase_elapsed_times() #觀測 Cp1、Cp2、Cp3、Cp4

        phase_queues = self.ts.get_phase_queues() #觀測 Qp1、Qp2、Qp3、Qp4

        # Python phase index 是 0,1,2,3
        # state 裡使用 p1,p2,p3,p4，所以 +1
        previous_phase = self.ts.previous_phase + 1 #觀測 t-1 時段是在哪個phase P1、P2、P3、P4

        observation = np.array(elapsed_times + phase_queues + [previous_phase], dtype=np.float32,)

        return observation


    def observation_space(self):

        #num_lanes = len(self.ts.lanes)
        num_phases = self.ts.num_green_phases
        #obs_dim = (self.ts.num_green_phases + num_lanes + num_lanes)
        low = np.array([0.0] * num_phases+ [0.0] * num_phases+ [1.0], dtype=np.float32,)
        high = np.array([np.inf] * num_phases + [np.inf] * num_phases + [float(num_phases)],dtype=np.float32,)

        #return spaces.Box(low=0.0, high=1.0, shape=(obs_dim,),dtype=np.float32,)
        return spaces.Box(low=low, high=high, dtype=np.float32)