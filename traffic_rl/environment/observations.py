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

        # 上一個執行的 Green phase
        phase_id = [
            1 if self.ts.green_phase == i else 0
            for i in range(self.ts.num_green_phases)
        ]

        # Incoming lane density
        density = self.ts.get_lanes_density()

        # Incoming lane queue
        queue = self.ts.get_lanes_queue()

        observation = np.array(
            phase_id + density + queue,
            dtype=np.float32,
        )

        return observation


    def observation_space(self):

        num_lanes = len(self.ts.lanes)

        obs_dim = (
            self.ts.num_green_phases
            + num_lanes
            + num_lanes
        )

        return spaces.Box(
            low=0.0,
            high=1.0,
            shape=(obs_dim,),
            dtype=np.float32,
        )