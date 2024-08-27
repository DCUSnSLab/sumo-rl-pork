from abc import abstractmethod
import numpy as np
from gymnasium import spaces
from typing import List
from traffic_signal import TrafficSignal, Section, Config_SUMO, Station, Detector

class ObservationFunction:
    """Abstract base class for observation functions."""

    def __init__(self, ts: TrafficSignal):
        """Initialize observation function."""
        self.ts = ts
        # 예시로 sections를 초기화하는 부분
        self.sections = self.ts.sections  # TrafficSignal에서 sections를 가져오는 가정
        # print(f"Sections: {self.sections}")  # 디버깅: sections가 올바르게 설정되었는지 확인
        self.section_objects = {section_id: Section(section_id, stations) for section_id, stations in self.sections.items()}
        # print(f"Section Objects: {self.section_objects}")  # 디버깅: section_objects가 올바르게 생성되었는지 확인
        self.max_queue_capacities = {
            "0": 107,  # 섹션 0의 최대 대기 차량 수
            "1": 88,  # 섹션 1의 최대 대기 차량 수
            "2": 80,  # 섹션 2의 최대 대기 차량 수
            "3": 60,  # 섹션 3의 최대 대기 차량 수
        }
        self.max_CO2_emissions = {
            '0': 318,  # 섹션 0의 CO2 최대 배출량
            '1': 594,  # 섹션 1의 CO2 최대 배출량
            '2': 598,  # 섹션 2의 CO2 최대 배출량
            '3': 522,  # 섹션 3의 CO2 최대 배출량
        }
    @abstractmethod
    def __call__(self):
        """Subclasses must override this method."""
        pass

    @abstractmethod
    def observation_space(self):
        """Subclasses must override this method."""
        pass

class DefaultObservationFunction(ObservationFunction):
    """Default observation function for traffic signals."""

    def __init__(self, ts: TrafficSignal):
        """Initialize default observation function."""
        super().__init__(ts)

    def __call__(self) -> np.ndarray:
        """Compute the observation for a given section."""
        self.queue = []
        queue_density = []
        co2_density = []
        new_co2 = []
        new_queue = []
        phase_id = [1 if self.ts.green_phase == i else 0 for i in range(min(self.ts.num_green_phases, 15))]  # One-hot encoding
        # min_green = [0 if self.ts.time_since_last_phase_change < self.ts.min_green + self.ts.yellow_time else 1]

        for section_id, section in self.ts.section_objects.items():
            section_co2_emission, _, traffic_queue, _ = section.collect_data()
            # print(f"section_id: {section_id}, obv_traffic_queue: {traffic_queue}, obv_co2: {section_co2_emission}")
            # print("density: ", self.density)
            # if isinstance(self.co2_emissions, list):
            #     self.co2_emissions.append(section_co2_emission)
            # else:
            #     self.co2_emissions = [section_co2_emission]

            # self.queue = [traffic_queue] if isinstance(traffic_queue, int) else traffic_queue
            max_CO2_emission = self.max_CO2_emissions.get(section_id)
            print(f"Section {section_id}- co2: {section_co2_emission}, max_co2: {max_CO2_emission}")
            co2_density.append(max(0, 1 - section_co2_emission))
            print(f"Section {section_id}- nomalized_co2: {co2_density}")


            max_queue_capacity = self.max_queue_capacities.get(section_id)
            print(f"Section {section_id}- traffic_queue: {traffic_queue}, max_queue_capacity: {max_queue_capacity}")
            normalized_queue = traffic_queue / max_queue_capacity
            queue_density.append(max(0, 1 - normalized_queue))
            print(f"Section {section_id}- nomalized_queue: {queue_density}")

        new_co2.append(co2_density[2])
        new_co2.append(co2_density[3])
        new_co2.append(co2_density[0])
        new_co2.append(co2_density[1])

        new_queue.append(queue_density[2])
        new_queue.append(queue_density[3])
        new_queue.append(queue_density[0])
        new_queue.append(queue_density[1])
        # print("co2_emissions:", self.co2_emissions)
        observation = phase_id + new_co2 + new_queue

        # observation = phase_id + min_green + co2_density +queue_density
        # observation = phase_id + min_green + self.co2_emissions + self.queue
        print("pre_observation: ", observation)
        if len(observation) < 16:
            observation.extend([0] * (16 - len(observation)))

        observation = np.array(observation, dtype=np.float32)
        print("observation: ", observation)
        return observation

    def observation_space(self) -> spaces.Box:
        """Return the observation space."""
        return spaces.Box(
            low=np.zeros(16, dtype=np.float32),
            high=np.ones(16, dtype=np.float32),
        )
