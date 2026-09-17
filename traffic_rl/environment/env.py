import os
import sys
from typing import Callable, Optional, Union

if "SUMO_HOME" in os.environ:
    tools = os.path.join(os.environ["SUMO_HOME"], "tools")
    sys.path.append(tools)
else:
    raise ImportError("Please declare the environment variable 'SUMO_HOME'")

import gymnasium as gym
import numpy as np
import sumolib
import traci

from .observations import DefaultObservationFunction, ObservationFunction
from .traffic_signal import TrafficSignal


class SumoEnvironment(gym.Env):

    """
    Single-agent SUMO environment.

    Decision structure:

        S_t
         ↓
        A_t
         ↓
        Green
         ↓
        Yellow
         ↓
        All Red
         ↓
        S_(t+1)

    Agent 只在 All Red 結束後做決策。
    """

    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        net_file: str,
        route_file: str,
        use_gui: bool = False,
        begin_time: int = 0,
        num_seconds: int = 3600,
        green_time: int = 90,
        yellow_time: int = 3,
        all_red_time: int = 2,
        reward_fn: Union[str, Callable] = "diff-waiting-time",
        observation_class: type[ObservationFunction] = DefaultObservationFunction,
        sumo_seed: Union[str, int] = "random",
        ts_id: Optional[str] = None,
        max_depart_delay: int = -1,
        waiting_time_memory: int = 1000,
        time_to_teleport: int = -1,
        sumo_warnings: bool = True,
    ):

        super().__init__()

        # ====================================================
        # SUMO files / GUI
        # ====================================================

        self.net_file = net_file
        self.route_file = route_file
        self.use_gui = use_gui

        self.sumo_binary = sumolib.checkBinary("sumo-gui" if self.use_gui else "sumo")

        # ====================================================
        # Simulation time
        # ====================================================

        self.begin_time = begin_time
        self.sim_max_time = begin_time + num_seconds

        # ====================================================
        # Traffic signal timing
        # ====================================================

        if green_time <= 0:
            raise ValueError("green_time must be > 0")

        if yellow_time < 0:
            raise ValueError("yellow_time must be >= 0")

        if all_red_time < 0:
            raise ValueError("all_red_time must be >= 0")

        self.green_time = green_time
        self.yellow_time = yellow_time
        self.all_red_time = all_red_time

        # 一個完整 decision epoch 的時間
        self.epoch_time = self.green_time + self.yellow_time + self.all_red_time

        # ====================================================
        # RL settings
        # ====================================================

        self.reward_fn = reward_fn
        self.observation_class = observation_class

        # ====================================================
        # SUMO settings
        # ====================================================

        self.sumo_seed = sumo_seed
        self.max_depart_delay = max_depart_delay
        self.waiting_time_memory = waiting_time_memory
        self.time_to_teleport = time_to_teleport
        self.sumo_warnings = sumo_warnings

        # ====================================================
        # Runtime variables
        # ====================================================

        self.sumo = None

        # TrafficSignal accumulated waiting time 會使用
        self.vehicles = {}

        # Evaluation counters
        self.num_arrived_vehicles = 0
        self.num_departed_vehicles = 0
        self.num_teleported_vehicles = 0

        # ====================================================
        # 暫時啟動 SUMO
        #
        # 用途：
        # 1. 找 traffic signal
        # 2. 建立 observation_space
        # 3. 建立 action_space
        # ====================================================

        traci.start([sumolib.checkBinary("sumo"), "-n", self.net_file])

        traffic_light_ids = list(traci.trafficlight.getIDList())

        if len(traffic_light_ids) == 0:
            traci.close()
            raise RuntimeError("No traffic light found in SUMO network.")

        # ====================================================
        # Traffic signal ID
        # ====================================================

        if ts_id is None:
            self.ts_id = traffic_light_ids[0]

        else:
            if ts_id not in traffic_light_ids:
                traci.close()
                raise ValueError(f"Traffic light '{ts_id}' does not exist.")

            self.ts_id = ts_id

        # ====================================================
        # 暫時建立 TrafficSignal
        #
        # 主要是為了取得：
        # observation_space
        # action_space
        # ====================================================

        self.traffic_signal = TrafficSignal(
            env=self,
            ts_id=self.ts_id,
            green_time=self.green_time,
            yellow_time=self.yellow_time,
            all_red_time=self.all_red_time,
            begin_time=self.begin_time,
            reward_fn=self.reward_fn,
            sumo=traci,
        )

        self.observation_space = self.traffic_signal.observation_space
        self.action_space = self.traffic_signal.action_space

        traci.close()

        self.sumo = None


    # ========================================================
    # Start SUMO
    # ========================================================

    def _start_simulation(self):

        sumo_cmd = [
            self.sumo_binary,
            "-n", self.net_file,
            "-r", self.route_file,
            "--max-depart-delay", str(self.max_depart_delay),
            "--waiting-time-memory", str(self.waiting_time_memory),
            "--time-to-teleport", str(self.time_to_teleport),
        ]

        if self.begin_time > 0:
            sumo_cmd.extend(["-b", str(self.begin_time)])

        if self.sumo_seed == "random":
            sumo_cmd.append("--random")
        else:
            sumo_cmd.extend(["--seed", str(self.sumo_seed)])

        if not self.sumo_warnings:
            sumo_cmd.append("--no-warnings")

        if self.use_gui:
            sumo_cmd.extend(["--start",])

        traci.start(sumo_cmd)

        self.sumo = traci


    # ========================================================
    # Reset
    # ========================================================

    def reset(self, seed: Optional[int] = None, options=None):

        super().reset(seed=seed)

        # 關閉上一個 episode
        if self.sumo is not None:
            self.close()

        # 更新 seed
        if seed is not None:
            self.sumo_seed = seed

        # 啟動新的 SUMO episode
        self._start_simulation()

        # 新 SUMO connection，所以重新建立 TrafficSignal
        self.traffic_signal = TrafficSignal(
            env=self,
            ts_id=self.ts_id,
            green_time=self.green_time,
            yellow_time=self.yellow_time,
            all_red_time=self.all_red_time,
            begin_time=self.begin_time,
            reward_fn=self.reward_fn,
            sumo=self.sumo,
        )

        # ====================================================
        # Reset statistics
        # ====================================================

        self.vehicles = {}

        self.num_arrived_vehicles = 0
        self.num_departed_vehicles = 0
        self.num_teleported_vehicles = 0

        # ====================================================
        # Initial All Red
        #
        # 讓 S0 和之後的 S1、S2...
        # 都是在 All Red 結束後觀察
        # ====================================================

        self.traffic_signal.set_all_red()

        self._run_for(self.all_red_time)

        # ====================================================
        # Observe S0
        # ====================================================

        observation = self.traffic_signal.compute_observation()

        # ====================================================
        # Initialize reward baseline
        #
        # 讓第一次 diff-waiting-time reward 變成：
        #
        # W(S0) - W(S1)
        # ====================================================

        self.traffic_signal.initialize_reward_baseline()

        # ====================================================
        # Evaluation info
        # ====================================================

        info = self._compute_info()

        return observation, info


    # ========================================================
    # Step
    # ========================================================

    def step(self, action: int):

        if not self.action_space.contains(action):
            raise ValueError(f"Invalid action: {action}")

        green_time = self.traffic_signal.get_green_duration(action)

        # 目前固定輪到的 phase
        self.traffic_signal.set_green()

        self._run_for(green_time)

        if self.sim_step < self.sim_max_time:
            self.traffic_signal.set_yellow()
            self._run_for(self.yellow_time)

        if self.sim_step < self.sim_max_time:
            self.traffic_signal.set_all_red()
            self._run_for(self.all_red_time)
        
        # 下一個 epoch 固定切到下一個 phase
        self.traffic_signal.next_phase()
        
        observation = self.traffic_signal.compute_observation()
        reward = self.traffic_signal.compute_reward()

        terminated = False
        truncated = self.sim_step >= self.sim_max_time

        info = self._compute_info()

        return observation, reward, terminated, truncated, info
    # ========================================================
    # Run SUMO for given duration
    # ========================================================

    def _run_for(self, duration: int):

        """
        讓目前 phase 持續 duration 秒。
        """

        target_time = min(self.sim_step + duration, self.sim_max_time)

        while self.sim_step < target_time:
            self._sumo_step()


    # ========================================================
    # One SUMO simulation step
    # ========================================================

    def _sumo_step(self):

        self.sumo.simulationStep()

        self.num_arrived_vehicles += self.sumo.simulation.getArrivedNumber()
        self.num_departed_vehicles += self.sumo.simulation.getDepartedNumber()
        self.num_teleported_vehicles += self.sumo.simulation.getEndingTeleportNumber()


    # ========================================================
    # Current SUMO time
    # ========================================================

    @property
    def sim_step(self):

        return self.sumo.simulation.getTime()


    # ========================================================
    # Evaluation metrics
    # ========================================================

    def _compute_info(self):

        vehicles = self.sumo.vehicle.getIDList()

        speeds = [self.sumo.vehicle.getSpeed(vehicle) for vehicle in vehicles]

        waiting_times = [self.sumo.vehicle.getWaitingTime(vehicle) for vehicle in vehicles]

        return {
            "step": self.sim_step,
            "total_running": len(vehicles),
            "total_stopped": sum(speed < 0.1 for speed in speeds),
            "total_arrived": self.num_arrived_vehicles,
            "total_departed": self.num_departed_vehicles,
            "total_teleported": self.num_teleported_vehicles,
            "total_waiting_time": sum(waiting_times),
            "mean_waiting_time": float(np.mean(waiting_times)) if vehicles else 0.0,
            "mean_speed": float(np.mean(speeds)) if vehicles else 0.0,
            "queue": self.traffic_signal.get_total_queued(),
            "accumulated_waiting_time": sum(self.traffic_signal.get_accumulated_waiting_time_per_lane()),
        }


    # ========================================================
    # Close SUMO
    # ========================================================

    def close(self):

        if self.sumo is not None:
            traci.close()
            self.sumo = None


    # ========================================================
    # Destructor
    # ========================================================

    def __del__(self):

        self.close()