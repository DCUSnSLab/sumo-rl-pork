"""This module contains the TrafficSignal class, which represents a traffic signal in the simulation."""
import os
import sys
from typing import Callable, List, Union

import sumo_rl
from enum import Enum
from collections import deque
from traci import TraCIException
import traci
if "SUMO_HOME" in os.environ:
    tools = os.path.join(os.environ["SUMO_HOME"], "tools")
    sys.path.append(tools)
else:
    raise ImportError("Please declare the environment variable 'SUMO_HOME'")
import numpy as np
from gymnasium import spaces
class Config_SUMO:
    # SUMO Configuration Files
    sumocfg_path = "../New_TestWay/test_cfg.sumocfg"
    # SUMO Scenario File Path
    scenario_path = "../New_TestWay"
    # SUMO Scenario File(.add.xml)
    scenario_file = "new_test.add.xml"

    sumoBinary = r'C:/Program Files (x86)/Eclipse/Sumo/bin/sumo-gui'

class Direction(Enum):
    SB = 0
    NB = 1
    EB = 2
    WB = 3

class InputStation(Enum):
    SB = '000000'
    NB = '010021'
    EB = '020018'
    WB = '030017'

def get_input_station_value(direction: Direction) -> str:
    # Direction의 name으로 InputStation을 찾아서 value를 반환
    return InputStation[direction.name].value

# Detector
class Detector:
    def __init__(self, id):
        self.id = id
        self.aux, self.bound, self.station_id, self.detector_id = self.parse_detector_id(id)
        self.minInterval = 30
        self.speed = 0
        self.volume = 0
        self.prevVehicles = tuple()
    def parse_detector_id(self, id):
        parts = id.split('_')
        if len(parts) != 2 or not parts[0].startswith("Det"):
            raise ValueError(f"Invalid detector ID format: {id}")
        det_info = parts[1]
        aux = det_info[0]
        bound = Direction(int(det_info[1]))
        station_id = det_info[0:6]
        detector_id = det_info[6:]
        return aux, bound, station_id, detector_id

    #update detection data by interval
    def update(self):
        vehicle_ids = traci.inductionloop.getLastStepVehicleIDs(self.id)
        #check duplicated vehicles
        dupvol = 0
        if self.prevVehicles is not None:
            for veh in vehicle_ids:
                if veh in self.prevVehicles:
                    dupvol += 1

        self.prevVehicles = vehicle_ids

        self.volume = traci.inductionloop.getLastStepVehicleNumber(self.id) - dupvol
        # if self.id == 'Det_02000000' or self.id == 'Det_02000001' or self.id == 'Det_12002604':
        #     print("%s -> v : %d" % (self.id, self.volume))
            #print('--- lsvid : ', vehicle_ids, type(vehicle_ids))

    def getVolume(self):
        return self.volume

    def getVehicles(self):
        return self.prevVehicles

class Station:
    def __init__(self, id, detectors):
        self.id = id
        self.dets = detectors
        self.direction = None if len(self.dets) == 0 else self.dets[0].bound
        self.volume = 0
        self.exitVolume = 0
        self.inputVeh = set()
        self.exitVeh = set()


    def update(self):
        self.volume = 0
        self.exitVolume = 0
        self.inputVeh = set()
        self.exitVeh = set()

        for det in self.dets:
            det.update()

            if det.aux == '1':
                self.exitVolume += det.getVolume()
                self.exitVeh.update(det.getVehicles())
            else:
                self.volume += det.getVolume()
                self.inputVeh.update(det.getVehicles())

        self.volume = self.volume if self.volume == 0 or self.volume < len(self.inputVeh) else len(self.inputVeh)
        self.exitVolume = self.exitVolume if self.exitVolume == 0 or self.exitVolume < len(self.exitVeh) else len(self.exitVeh)


    def getVolume(self):
        return self.volume

    def getExitVolume(self):
        return self.exitVolume

    def getVehicleData(self):
        return list(self.inputVeh), list(self.exitVeh)

    def getInputVehIds(self):
        return self.inputVeh

    def getExitVehIds(self):
        return self.exitVeh

class Section:
    def __init__(self, id, stations):
        self.id = id
        self.stations = stations
        self.direction = None if len(self.stations) == 0 else self.stations[0].direction
        self.section_co2_emission = 0
        self.section_volume = 0
        self.traffic_queue = 0
        self.section_vehicles = set()
        # self.stop_lane = self.StopLane_position()
        self.in_dilemmaZone=set()

    def reset(self):
        self.section_co2_emission = 0
        self.section_volume = 0
        self.traffic_queue = 0
        self.section_vehicles = set()
        self.in_dilemmaZone = set()

    def collect_data(self):
        # print("CD_sec_co2:", self.section_co2_emission, "CD_sec_vol: ", self.section_volume, "CD_queue :", self.traffic_queue)
        return self.section_co2_emission, self.section_volume, self.traffic_queue, list(self.section_vehicles)

    def update(self):
        self.section_co2_emission = 0
        self.section_volume = 0
        removal_veh = list()
        for i, station in enumerate(self.stations):
            station.update()
            if i == 0:
                self.section_volume += station.getVolume()
            if station.id == get_input_station_value(self.direction):
                self.traffic_queue += station.getVolume()
                self.section_vehicles.update(station.getInputVehIds())

            self.traffic_queue -= station.getExitVolume()
            self.section_vehicles.difference_update(station.getExitVehIds())

            # Ensure traffic_queue does not become negative
            if self.traffic_queue < 0:
                print(f"Warning: traffic_queue became negative: {self.traffic_queue}")
                self.traffic_queue = 0

        for vehicle in self.section_vehicles:
            try:
                if traci.vehicle.getCO2Emission(vehicle) >= 0:
                    self.section_co2_emission += traci.vehicle.getCO2Emission(vehicle) / 1000
            except TraCIException:
                removal_veh.append(vehicle)

        self.section_vehicles.difference_update(removal_veh)


class TrafficSignal:
    def __init__(
            self,
            env,
            ts_id: str,
            delta_time: int,
            yellow_time: int,
            min_green: int,
            max_green: int,
            begin_time: int,
            reward_fn: Union[str, Callable],
            section_objects,
            sumo
    ):
        self.id = ts_id
        self.env = env
        self.delta_time = delta_time
        self.yellow_time = yellow_time
        self.min_green = min_green
        self.max_green = max_green
        self.green_phase = 0
        self.is_yellow = False
        self.time_since_last_phase_change = 0
        self.next_action_time = begin_time
        self.last_measure = 0.0
        self.last_CO2 = 0.0
        self.last_reward = None
        self.reward_fn = reward_fn
        self.sumo = sumo
        self.config = Config_SUMO
        self.detectors = [Detector(detector_id) for detector_id in self.__get_detector_ids(self.config)]
        self.stations = {}
        self.sections = {}
        self.section_results = deque()
        self.total_results = deque()
        self.traffic_light_id = "TLS_0"
        self.total_co2_emission = 0
        self.total_volume = 0
        self.__get_station()
        self.reward = 0
        self.station_objects = {station_id: Station(station_id, detectors) for station_id, detectors in
                                self.stations.items()}
        self.__get_section()

        self.original_phase_durations = {}
        self.stepbySec = 1
        self.colDuration = 30  # seconds

        # 인스턴스 변수로 정의
        self.max_queue_capacities = {
            "0": 107,  # 섹션 0의 최대 대기 차량 수
            "1": 88,  # 섹션 1의 최대 대기 차량 수
            "2": 80,  # 섹션 2의 최대 대기 차량 수
            "3": 60,  # 섹션 3의 최대 대기 차량 수
        }
        self.max_CO2_emissions = {
            '0': 318,  # 섹션 0의 CO2 최대 배출량
            '1': 394,  # 섹션 1의 CO2 최대 배출량
            '2': 198,  # 섹션 2의 CO2 최대 배출량
            '3': 122,  # 섹션 3의 CO2 최대 배출량
        }
        if type(self.reward_fn) is str:
            if self.reward_fn in TrafficSignal.reward_fns.keys():
                self.reward_fn = TrafficSignal.reward_fns[self.reward_fn]
            else:
                raise NotImplementedError(f"Reward function {self.reward_fn} not implemented")

        self.observation_fn = self.env.observation_class(self)

        self._build_phases()

        self.section_objects = {section_id: Section(section_id, stations) for section_id, stations in self.sections.items()}
        self.lanes = list(
            dict.fromkeys(self.sumo.trafficlight.getControlledLanes(self.id))
        )  # Remove duplicates and keep order
        self.out_lanes = [link[0][1] for link in self.sumo.trafficlight.getControlledLinks(self.id) if link]
        self.out_lanes = list(set(self.out_lanes))
        self.lanes_length = {lane: self.sumo.lane.getLength(lane) for lane in self.lanes + self.out_lanes}

        self.observation_space = self.observation_fn.observation_space()
        self.action_space = spaces.Discrete(self.num_green_phases)
        self.section_objects = section_objects
    def __get_detector_ids(self, config):
        detector_ids = []
        with open(os.path.join(config.scenario_path, config.scenario_file), "r") as f:
            for line in f:
                if "inductionLoop" in line:
                    # parts = line.split('"')
                    # detector_ids.append(parts[1])
                    parts = line.split('"')
                    detector_id = parts[1]
                    # print(f"Found detector ID: {detector_id}")  # Debugging statement
                    detector_ids.append(detector_id)
        return detector_ids
    def __get_station(self):
        for detector in self.detectors:
            if detector.station_id not in self.stations:
                self.stations[detector.station_id] = []
            self.stations[detector.station_id].append(detector)

    def __get_section(self):
        for station_id in self.stations:
            section_id = station_id[1]
            if section_id not in self.sections:
                self.sections[section_id] = []
            self.sections[section_id].append(self.station_objects[station_id])

    def _build_phases(self):
        phases = self.sumo.trafficlight.getAllProgramLogics(self.id)[0].phases
        if self.env.fixed_ts:
            self.num_green_phases = len(phases) // 2  # Number of green phases == number of phases (green+yellow) divided by 2
            return


        self.green_phases = []
        self.yellow_dict = {}
        for phase in phases:
            state = phase.state
            if "y" not in state and (state.count("r") + state.count("s") != len(state)):
                self.green_phases.append(self.sumo.trafficlight.Phase(60, state))
        self.num_green_phases = len(self.green_phases)
        self.all_phases = self.green_phases.copy()

        for i, p1 in enumerate(self.green_phases):
            for j, p2 in enumerate(self.green_phases):
                if i == j:
                    continue
                yellow_state = ""
                for s in range(len(p1.state)):
                    if (p1.state[s] == "G" or p1.state[s] == "g") and (p2.state[s] == "r" or p2.state[s] == "s"):
                        yellow_state += "y"
                    else:
                        yellow_state += p1.state[s]
                self.yellow_dict[(i, j)] = len(self.all_phases)
                self.all_phases.append(self.sumo.trafficlight.Phase(self.yellow_time, yellow_state))

        programs = self.sumo.trafficlight.getAllProgramLogics(self.id)
        print("programs: ", programs)
        logic = programs[0]
        print("logic.type: ", logic.type)
        logic.type = 1

        logic.phases = self.all_phases
        self.sumo.trafficlight.setProgramLogic(self.id, logic)
        self.sumo.trafficlight.setRedYellowGreenState(self.id, self.all_phases[0].state)

    @property
    def time_to_act(self):
        """Returns True if the traffic signal should act in the current step."""
        return self.next_action_time == self.env.sim_step

    def update(self):
        """Updates the traffic signal state.

        If the traffic signal should act, it will set the next green phase and update the next action time.
        """
        self.time_since_last_phase_change += 1
        if self.is_yellow and self.time_since_last_phase_change == self.yellow_time:
            # self.sumo.trafficlight.setPhase(self.id, self.green_phase)
            self.sumo.trafficlight.setRedYellowGreenState(self.id, self.all_phases[self.green_phase].state)
            self.is_yellow = False


    def set_next_phase (self, new_phase: int):
        """Sets what will be the next green phase and sets yellow phase if the next phase is different than the current.

        Args:
            new_phase (int): Number between [0 ... num_green_phases]
        """
        new_phase = int(new_phase)
        if self.green_phase == new_phase or self.time_since_last_phase_change < self.yellow_time + self.min_green:
            # self.sumo.trafficlight.setPhase(self.id, self.green_phase)
            self.sumo.trafficlight.setRedYellowGreenState(self.id, self.all_phases[self.green_phase].state)
            self.next_action_time = self.env.sim_step + self.delta_time
        else:
            # self.sumo.trafficlight.setPhase(self.id, self.yellow_dict[(self.green_phase, new_phase)])  # turns yellow
            self.sumo.trafficlight.setRedYellowGreenState(
                self.id, self.all_phases[self.yellow_dict[(self.green_phase, new_phase)]].state
            )
            self.green_phase = new_phase
            self.next_action_time = self.env.sim_step + self.delta_time
            self.is_yellow = True
            self.time_since_last_phase_change = 0

    def compute_observation(self):
        """Computes the observation of the traffic signal."""
        return self.observation_fn()

    def compute_reward(self):
        """Computes the reward of the traffic signal."""
        self.last_reward = self.reward_fn(self)
        print(self.last_reward)
        return self.last_reward

    def _pressure_reward(self):
        return self.get_pressure()

    def _average_speed_reward(self):
        return self.get_average_speed()


    def _combined_reward_with_section(self, weight_CO2=0.6, weight_max_vehicles=0.4):
        # 각 세션에 대한 보상 저장
        section_rewards = {}
        section_CO2 = []
        section_queue =[]
        new_co2_reward = []
        new_queue_reward = []
        for section_id, section in self.section_objects.items():
            # 각 섹션의 데이터 수집
            section_co2_emission, section_volume, traffic_queue, section_vehicles = section.collect_data()
            # print(f"section_id: {section_id}, traffic_signal_traffic_queue: {traffic_queue}",)
            # 섹션별 CO2 배출량에 대한 보상 계산
            # max_CO2_emission = self.max_CO2_emissions.get(section_id)
            print(f"Section {section_id}- co2: {section_co2_emission}")
            section_CO2_tmp = max(0, 1 - section_co2_emission)
            section_CO2.append(section_CO2_tmp)
            # 섹션별 대기 큐에 대한 보상 계산
            max_queue_capacity = self.max_queue_capacities.get(section_id)
            print(f"Section {section_id}- traffic_queue: {traffic_queue}, max_queue_capacity: {max_queue_capacity}")
            normalized_queue = traffic_queue / max_queue_capacity
            section_queue_tmp = max(0, 1 - normalized_queue)
            section_queue.append(section_queue_tmp)

        new_co2_reward.append(section_CO2[2])
        new_co2_reward.append(section_CO2[3])
        new_co2_reward.append(section_CO2[0])
        new_co2_reward.append(section_CO2[1])
        new_queue_reward.append(section_queue[2])
        new_queue_reward.append(section_queue[3])
        new_queue_reward.append(section_queue[0])
        new_queue_reward.append(section_queue[1])
        print(f"new_co2: {new_co2_reward}, new_qeueu: {new_queue_reward}")
        section_reward = sum(weight_CO2 * co2 for co2 in new_co2_reward) + sum(weight_max_vehicles * queue for queue in new_queue_reward)

        for section_id, section in self.section_objects.items():
            # 가중치를 적용한 섹션별 보상 계산
            # section_reward = (weight_CO2 * section_CO2_reward)
            section_rewards[section_id] = section_reward

            # 디버깅용 출력
            # print(f"Section {section_id}: CO2 Reward: {section_CO2_reward}")
            print(f"Section {section_id}: CO2 Reward: {new_co2_reward}, Queue Reward: {new_queue_reward}, Combined Reward: {new_queue_reward}")

        # 가장 낮은 보상치를 가진 세션 식별
        worst_section_id = min(section_rewards, key=section_rewards.get)
        worst_section_reward = section_rewards[worst_section_id]

        # 보상 값 확인

        print(f"Section Rewards: {section_rewards}")
        print(f"Worst Section: {worst_section_id} with Reward: {worst_section_reward}")
        print("%#%" * 30)

        # 최악의 세션의 보상만 반환
        return worst_section_reward
    #
    # def _combined_reward_with_section(self, weight_CO2=0.6, weight_max_vehicles=0.4):
    #     # 각 세션에 대한 보상 저장
    #     section_rewards = {}
    #
    #     for section_id, section in self.section_objects.items():
    #         # 각 섹션의 데이터 수집
    #         section_co2_emission, section_volume, traffic_queue, section_vehicles = section.collect_data()
    #         # print(f"section_id: {section_id}, traffic_signal_traffic_queue: {traffic_queue}",)
    #         # 섹션별 CO2 배출량에 대한 보상 계산
    #         max_CO2_emission = self.max_CO2_emissions.get(int(section_id))
    #         print(f"Section {section_id}- co2: {section_co2_emission}, max_co2: {max_CO2_emission}")
    #         section_CO2_reward = max(0, 1 - section_co2_emission / max_CO2_emission)
    #
    #         # 섹션별 대기 큐에 대한 보상 계산
    #         max_queue_capacity = self.max_queue_capacities.get(int(section_id))
    #         print(f"Section {section_id}- traffic_queue: {traffic_queue}, max_queue_capacity: {max_queue_capacity}")
    #         normalized_queue = traffic_queue / max_queue_capacity
    #         section_queue_reward = max(0, 1 - normalized_queue)
    #
    #         # 가중치를 적용한 섹션별 보상 계산
    #         section_reward = (weight_CO2 * section_CO2_reward) + (weight_max_vehicles * section_queue_reward)
    #         section_rewards[section_id] = section_reward
    #
    #         # 디버깅용 출력
    #         print(f"Section {section_id}: CO2 Reward: {section_CO2_reward}, Queue Reward: {section_queue_reward}, Combined Reward: {section_reward}")
    #
    #     # 전체 시스템 보상 계산
    #     total_reward = sum(section_rewards.values()) / len(section_rewards)  # 평균 보상
    #
    #     # 보상 값 확인
    #     print(f"Section Rewards: {section_rewards}")
    #     print(f"Total System Reward: {total_reward}")
    #     print("%%%" * 30)
    #
    #     # 전체 시스템 보상 반환
    #     return total_reward
    #
    # def _observation_fn_default(self):
    #     from observations import DefaultObservationFunction, ObservationFunction
    #     self.queue = []
    #     try:
    #         # Phase ID (assuming green_phase is an integer and num_green_phases is the total number of phases)
    #         phase_id = [1 if self.green_phase == i else 0 for i in
    #                     range(min(self.num_green_phases, 15))]  # One-hot encoding
    #         min_green = [0 if self.time_since_last_phase_change < self.min_green + self.yellow_time else 1]
    #
    #         # Density calculation (assumes get_Section_density returns a list of densities)
    #         density = DefaultObservationFunction.get_Section_density()
    #
    #         if density is None:
    #             density = [0] * 4  # Replace with a list of zeros of the expected length
    #         print("density_traffic :", density)
    #         # CO2 Emission calculation
    #         co2_emissions = []
    #         for section_id, section in self.section_objects.items():
    #             section_co2_emission, _, _, _ = section.collect_data()
    #             co2_emissions.append(section_co2_emission)
    #         if len(co2_emissions) != 4:
    #             co2_emissions = [0] * 4  # Ensure the list has the correct length
    #
    #         # Queue calculation (fetching the latest value from deque)
    #         for section_id, section in self.section_objects.items():
    #             _, _, traffic_queue, _ = section.collect_data()
    #             self.queue = traffic_queue
    #         flattened_queue = self.queue if isinstance(self.queue, list) else list(self.queue)
    #
    #         # Combine all parts into one list
    #         observation = phase_id + min_green + density + co2_emissions + flattened_queue
    #
    #         # Ensure the observation has exactly 16 elements
    #         if len(observation) < 16:
    #             observation.extend([0] * (16 - len(observation)))
    #
    #         # Convert the observation to a numpy array
    #         observation = np.array(observation, dtype=np.float32)
    #         return observation
    #     except IndexError as e:
    #         print(f"IndexError encountered: {e}")
    #         return np.zeros(self.observation_space().shape, dtype=np.float32)

    # def _observation_fn_default(self):
    #     from observations import DefaultObservationFunction, ObservationFunction
    #     self.queue = []
    #     try:
    #         # Phase ID (assuming green_phase is an integer and num_green_phases is the total number of phases)
    #         phase_id = [1 if self.green_phase == i else 0 for i in
    #                     range(min(self.num_green_phases, 15))]  # One-hot encoding
    #         # print("phase_id: ", phase_id)
    #         min_green = [0 if self.time_since_last_phase_change < self.min_green + self.yellow_time else 1]
    #         # print("min_green: ", min_green)
    #
    #         # Density calculation (assumes get_Section_density returns a list of densities)
    #         density = DefaultObservationFunction.get_Section_density()
    #         if density is None:
    #             density = [0] * 4  # Replace with a list of zeros of the expected length
    #         # print("density:", density)
    #
    #         # Queue calculation (fetching the latest value from deque)
    #         for section_id, section in self.section_objects.items():
    #             section_co2_emission, section_volume, traffic_queue, section_vehicles = section.collect_data()
    #             self.queue = traffic_queue
    #             # print("self.queue: ", self.queue)
    #         # Assuming `self.queue` is a list, we need to flatten it if necessary.
    #         flattened_queue = self.queue if isinstance(self.queue, list) else list(self.queue)
    #
    #         # Create the observation array by flattening `self.queue`
    #         observation = np.array(phase_id + min_green + density + flattened_queue, dtype=np.float32)
    #
    #         # Create the observation array
    #         # observation = np.array(phase_id + min_green + density + [self.queue], dtype=np.float32)
    #         # print("observation = np.array(phase_id + min_green + density + flattened_queue, dtype=np.float32): ",
    #         #       observation)
    #         flattened_queue = self.queue if isinstance(self.queue, list) else list(self.queue)
    #
    #         # Combine all parts into one list
    #         observation = phase_id + min_green + density + flattened_queue
    #         # print("observation = phase_id + min_green + density + flattened_queue: ", observation)
    #
    #         # Ensure the observation has exactly 16 elements
    #         # If it has fewer than 16 elements, pad with zeros
    #         if len(observation) < 16:
    #             observation.extend([0] * (16 - len(observation)))
    #             # print("len(observation) < 16: ", observation)
    #         # Convert the observation to a numpy array
    #         observation = np.array(observation, dtype=np.float32)
    #         # print("np.array(observation, dtype=np.float32): ", observation)
    #         return observation
    #     except IndexError as e:
    #         print(f"IndexError encountered: {e}")
    #         return np.zeros(self.observation_space().shape, dtype=np.float32)


    def get_average_speed(self) -> float:
        """Returns the average speed normalized by the maximum allowed speed of the vehicles in the intersection.

        Obs: If there are no vehicles in the intersection, it returns 1.0.
        """
        avg_speed = 0.0
        vehs = self._get_veh_list()
        if len(vehs) == 0:
            return 1.0
        for v in vehs:
            avg_speed += self.sumo.vehicle.getSpeed(v) / self.sumo.vehicle.getAllowedSpeed(v)
        return avg_speed / len(vehs)

    def get_pressure(self):
        """Returns the pressure (#veh leaving - #veh approaching) of the intersection."""
        return sum(self.sumo.lane.getLastStepVehicleNumber(lane) for lane in self.out_lanes) - sum(
            self.sumo.lane.getLastStepVehicleNumber(lane) for lane in self.lanes
        )

    def _get_veh_list(self):
        veh_list = []
        for lane in self.lanes:
            veh_list += self.sumo.lane.getLastStepVehicleIDs(lane)
        return veh_list

    @classmethod
    def register_reward_fn(cls, fn: Callable):
        """Registers a reward function.

        Args:
            fn (Callable): The reward function to register.
        """
        if fn.__name__ in cls.reward_fns.keys():
            raise KeyError(f"Reward function {fn.__name__} already exists")

        cls.reward_fns[fn.__name__] = fn

    reward_fns = {
        "combined_reward": lambda env: env._combined_reward_with_section(weight_CO2=0.6, weight_max_vehicles=0.4),
    }