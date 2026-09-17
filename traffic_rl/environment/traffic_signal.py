import os
import sys
from typing import Callable, List, Union


if "SUMO_HOME" in os.environ:
    tools = os.path.join(os.environ["SUMO_HOME"], "tools")
    sys.path.append(tools)
else:
    raise ImportError("Please declare the environment variable 'SUMO_HOME'")


from gymnasium import spaces


class TrafficSignal:

    MIN_GAP = 2.5

    def __init__(
        self,
        env,
        ts_id: str,
        green_time: int,
        yellow_time: int,
        all_red_time: int,
        begin_time: int,
        reward_fn: Union[str, Callable],
        sumo,
    ):

        self.id = ts_id
        self.env = env
        self.sumo = sumo

        self.green_time = green_time
        self.yellow_time = yellow_time
        self.all_red_time = all_red_time
        self.begin_time = begin_time

        # 最近一次執行的 Green action
        self.green_phase = 0

        self.last_ts_waiting_time = 0.0
        self.last_reward = None

        self.reward_fn = self._get_reward_fn_from_string(reward_fn)

        self._build_phases()

        # Incoming lanes
        self.lanes = list(
            dict.fromkeys(
                self.sumo.trafficlight.getControlledLanes(self.id)
            )
        )

        # Outgoing lanes
        controlled_links = self.sumo.trafficlight.getControlledLinks(self.id)

        self.out_lanes = [
            link[0][1]
            for link in controlled_links
            if link
        ]

        self.out_lanes = list(set(self.out_lanes))

        all_lanes = list(set(self.lanes + self.out_lanes))

        self.lanes_length = {
            lane: self.sumo.lane.getLength(lane)
            for lane in all_lanes
        }

        # Observation
        self.observation_fn = self.env.observation_class(self)
        self.observation_space = self.observation_fn.observation_space()

        # Action = choose Green phase
        self.green_durations = [30, 45, 60, 75, 90]
        self.action_space = spaces.Discrete(len(self.green_durations))


    # ========================================================
    # Build Green phases
    # ========================================================

    def _build_phases(self):

        programs = self.sumo.trafficlight.getAllProgramLogics(self.id)

        phases = programs[0].phases

        self.green_phases = []

        for phase in phases:

            state = phase.state

            has_yellow = "y" in state or "Y" in state

            all_stopped = all(
                signal in ("r", "R", "s")
                for signal in state
            )

            if not has_yellow and not all_stopped:
                self.green_phases.append(state)

        self.num_green_phases = len(self.green_phases)

        if self.num_green_phases == 0:
            raise RuntimeError(
                f"No green phase found for traffic light {self.id}."
            )

        state_length = len(self.green_phases[0])

        self.all_red_state = "r" * state_length


    # ========================================================
    # Green
    # ========================================================

    def set_green(self):

        green_state = self.green_phases[self.green_phase]

        self.sumo.trafficlight.setRedYellowGreenState(self.id, green_state)

    def get_green_duration(self, action: int):

        action = int(action)

        if action < 0 or action >= len(self.green_durations):
            raise ValueError(f"Invalid duration action: {action}")

        return self.green_durations[action]

    def next_phase(self):

        self.green_phase = (self.green_phase + 1) % self.num_green_phases

    # ========================================================
    # Yellow
    # ========================================================

    def set_yellow(self):

        current_green_state = self.green_phases[self.green_phase]

        yellow_state = ""

        for signal in current_green_state:

            if signal in ("G", "g"):
                yellow_state += "y"

            else:
                yellow_state += "r"

        self.sumo.trafficlight.setRedYellowGreenState(
            self.id,
            yellow_state,
        )


    # ========================================================
    # All Red
    # ========================================================

    def set_all_red(self):

        self.sumo.trafficlight.setRedYellowGreenState(
            self.id,
            self.all_red_state,
        )


    # ========================================================
    # Observation
    # ========================================================

    def compute_observation(self):

        return self.observation_fn()


    # ========================================================
    # Reward
    # ========================================================

    def compute_reward(self):

        self.last_reward = self.reward_fn(self)

        return self.last_reward


    def _get_reward_fn_from_string(self, reward_fn):

        if isinstance(reward_fn, str):

            if reward_fn not in self.reward_fns:
                raise NotImplementedError(
                    f"Reward function '{reward_fn}' is not implemented."
                )

            return self.reward_fns[reward_fn]

        return reward_fn


    def initialize_reward_baseline(self):

        self.last_ts_waiting_time = (
            sum(self.get_accumulated_waiting_time_per_lane()) / 100.0
        )


    # ========================================================
    # Rewards
    # ========================================================

    def _pressure_reward(self):

        return self.get_pressure()


    def _average_speed_reward(self):

        return self.get_average_speed()


    def _queue_reward(self):

        return -self.get_total_queued()


    def _co2_reward(self):

        return -self.get_total_co2()


    def _diff_waiting_time_reward(self):

        ts_wait = (
            sum(self.get_accumulated_waiting_time_per_lane()) / 100.0
        )

        reward = self.last_ts_waiting_time - ts_wait

        self.last_ts_waiting_time = ts_wait

        return reward


    # ========================================================
    # Waiting time
    # ========================================================

    def get_accumulated_waiting_time_per_lane(self) -> List[float]:

        wait_time_per_lane = []

        for lane in self.lanes:

            veh_list = self.sumo.lane.getLastStepVehicleIDs(lane)

            wait_time = 0.0

            for veh in veh_list:

                veh_lane = self.sumo.vehicle.getLaneID(veh)

                acc = self.sumo.vehicle.getAccumulatedWaitingTime(veh)

                if veh not in self.env.vehicles:

                    self.env.vehicles[veh] = {
                        veh_lane: acc
                    }

                else:

                    old_waiting_time = sum(
                        self.env.vehicles[veh][old_lane]
                        for old_lane in self.env.vehicles[veh]
                        if old_lane != veh_lane
                    )

                    self.env.vehicles[veh][veh_lane] = (
                        acc - old_waiting_time
                    )

                wait_time += self.env.vehicles[veh][veh_lane]

            wait_time_per_lane.append(wait_time)

        return wait_time_per_lane


    # ========================================================
    # Average normalized speed
    # ========================================================

    def get_average_speed(self) -> float:

        vehs = self._get_veh_list()

        if len(vehs) == 0:
            return 1.0

        avg_speed = 0.0

        for veh in vehs:

            allowed_speed = self.sumo.vehicle.getAllowedSpeed(veh)

            if allowed_speed > 0:

                avg_speed += (
                    self.sumo.vehicle.getSpeed(veh)
                    / allowed_speed
                )

        return avg_speed / len(vehs)


    # ========================================================
    # Pressure
    # ========================================================

    def get_pressure(self):

        outgoing = sum(
            self.sumo.lane.getLastStepVehicleNumber(lane)
            for lane in self.out_lanes
        )

        incoming = sum(
            self.sumo.lane.getLastStepVehicleNumber(lane)
            for lane in self.lanes
        )

        return outgoing - incoming


    # ========================================================
    # Density
    # ========================================================

    def get_out_lanes_density(self) -> List[float]:

        return self._get_density(self.out_lanes)


    def get_lanes_density(self) -> List[float]:

        return self._get_density(self.lanes)


    def _get_density(self, lanes):

        densities = []

        for lane in lanes:

            vehicle_number = (
                self.sumo.lane.getLastStepVehicleNumber(lane)
            )

            vehicle_length = (
                self.sumo.lane.getLastStepLength(lane)
            )

            capacity = (
                self.lanes_length[lane]
                / (self.MIN_GAP + vehicle_length)
            )

            if capacity <= 0:
                density = 0.0
            else:
                density = vehicle_number / capacity

            densities.append(min(1.0, density))

        return densities


    # ========================================================
    # Queue
    # ========================================================

    def get_lanes_queue(self) -> List[float]:

        queues = []

        for lane in self.lanes:

            halting_number = (
                self.sumo.lane.getLastStepHaltingNumber(lane)
            )

            vehicle_length = (
                self.sumo.lane.getLastStepLength(lane)
            )

            capacity = (
                self.lanes_length[lane]
                / (self.MIN_GAP + vehicle_length)
            )

            if capacity <= 0:
                queue = 0.0
            else:
                queue = halting_number / capacity

            queues.append(min(1.0, queue))

        return queues


    def get_total_queued(self) -> int:

        return sum(
            self.sumo.lane.getLastStepHaltingNumber(lane)
            for lane in self.lanes
        )


    # ========================================================
    # CO2
    # ========================================================

    def get_total_co2(self) -> float:

        return sum(
            self.sumo.lane.getCO2Emission(lane)
            for lane in self.lanes
        )


    # ========================================================
    # Vehicle list
    # ========================================================

    def _get_veh_list(self):

        veh_list = []

        for lane in self.lanes:
            veh_list += self.sumo.lane.getLastStepVehicleIDs(lane)

        return veh_list


    # ========================================================
    # Reward registration
    # ========================================================

    @classmethod
    def register_reward_fn(cls, fn: Callable):

        if fn.__name__ in cls.reward_fns:
            raise KeyError(
                f"Reward function {fn.__name__} already exists"
            )

        cls.reward_fns[fn.__name__] = fn


    reward_fns = {
        "diff-waiting-time": _diff_waiting_time_reward,
        "average-speed": _average_speed_reward,
        "queue": _queue_reward,
        "pressure": _pressure_reward,
        "co2": _co2_reward,
    }