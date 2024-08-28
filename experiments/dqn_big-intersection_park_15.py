from stable_baselines3 import DQN
from stable_baselines3.common.callbacks import BaseCallback
from datetime import datetime
from experiments.Infra import Config_SUMO, Infra, SECTION_RESULT, TOTAL_RESULT
from experiments.RunSimulation import RunSimulation
from sumo_rl import SumoEnvironment
import numpy as np
from gymnasium import spaces
import traci
from sumo_rl import ObservationFunction, TrafficSignal

class CustomSumoEnvironment(SumoEnvironment):
    def __init__(self, simInfra, **kwargs):
        super().__init__(**kwargs)
        self._cust_step = 0
        self._cust_infra:Infra = simInfra

    def _compute_reward(self):
        # Get total CO2 emission in the current simulation step
        total_co2_emission = self._cust_infra.getTotalCO2mg()
        # for veh_id in traci.vehicle.getIDList():
        #     total_co2_emission += traci.vehicle.getCO2Emission(veh_id)
        print(f"reward : {-total_co2_emission}")
        # Calculate the reward as the negative of the CO2 emission
        reward = -total_co2_emission
        return reward

    def _get_system_info(self):
        vehicles = self.sumo.vehicle.getIDList()
        speeds = [self.sumo.vehicle.getSpeed(vehicle) for vehicle in vehicles]
        waiting_times = [self.sumo.vehicle.getWaitingTime(vehicle) for vehicle in vehicles]
        co2 = [self.sumo.vehicle.getCO2Emission(vehicle) for vehicle in vehicles]
        return {
            # In SUMO, a vehicle is considered halting if its speed is below 0.1 m/s
            "system_total_co2": sum(co2),
            "system_total_stopped": sum(int(speed < 0.1) for speed in speeds),
            "system_total_waiting_time": sum(waiting_times),
            "system_mean_waiting_time": 0.0 if len(vehicles) == 0 else np.mean(waiting_times),
            "system_mean_speed": 0.0 if len(vehicles) == 0 else np.mean(speeds),
        }

    def getCustInfra(self):
        return self._cust_infra

    def _sumo_step(self):
        self.sumo.simulationStep()
        self._cust_infra.update()
        self._cust_step += 1

class CO2ObservationFunction(ObservationFunction):
    """CO2-based observation function for traffic signals."""

    def __init__(self, ts: TrafficSignal):
        """Initialize CO2 observation function."""
        super().__init__(ts)

    def __call__(self) -> np.ndarray:
        self._custInfra: Infra = self.ts.env.getCustInfra()

        total_co2_emission = 0
        co2_emissions = []
        # for veh_id in traci.vehicle.getIDList():
        #     total_co2_emission += traci.vehicle.getCO2Emission(veh_id)
        total_co2_emission = self._custInfra.getTotalCO2mg()

        #for i, section in self._custInfra.getSections().items():
        #    co2_emissions.append(section.getCurrentCO2()*1000)

        """Return the CO2-based observation."""
        phase_id = [1 if self.ts.green_phase == i else 0 for i in range(self.ts.num_green_phases)]  # one-hot encoding
        min_green = [0 if self.ts.time_since_last_phase_change < self.ts.min_green + self.ts.yellow_time else 1]
        co2_emissions.append(total_co2_emission)
        print('total_emission: ',total_co2_emission, co2_emissions)
        # co2_emissions = self.ts.get_lanes_co2_emission()
        observation = np.array(phase_id + min_green + co2_emissions, dtype=np.float32)
        print("observation: ", observation)
        return observation

    def observation_space(self) -> spaces.Box:
        """Return the observation space."""
        return spaces.Box(
            low=np.zeros(self.ts.num_green_phases + 1 + 1, dtype=np.float32),
            high=np.ones(self.ts.num_green_phases + 1 + 1, dtype=np.float32),
        )

# 콜백 클래스를 정의하여 학습 결과를 매번 출력하도록 설정
class EveryStepCallback(BaseCallback):
    def __init__(self, verbose=0):
        super(EveryStepCallback, self).__init__(verbose)

    def _on_step(self) -> bool:
        if self.verbose > 0:
            print(f"Step: {self.num_timesteps}, Reward: {self.locals['rewards'][-1]}")
        return True

class LearningManager(RunSimulation):
    def __init__(self):
        super().__init__(Config_SUMO, 'RL_DQL', isExternalSignal=True)
        self.num_seconds = 11700
        self.start_time = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.env = CustomSumoEnvironment(
            net_file=self.config.scenario_file,
            single_agent=True,
            route_file=self.config.route_file,
            use_gui=True,
            num_seconds=self.num_seconds,
            yellow_time=4,
            min_green=5,
            max_green=120,
            sumo_seed=1,
            observation_class=CO2ObservationFunction,
            simInfra=self.getInfra()
        )

        self.model = DQN(
            env=self.env,
            policy="MlpPolicy",
            learning_rate=3e-3,
            learning_starts=0,
            buffer_size=50000,
            train_freq=1,
            target_update_interval=500,
            exploration_initial_eps=0.7,
            exploration_fraction=0.05,
            exploration_final_eps=0.3,
            verbose=1,
            device="cuda"
        )

        self.output_folder = f"outputs/dqn_{self.start_time}_numsec_{self.num_seconds}"

    def preinit(self):
        pass

    def run_simulation(self):
        total_timesteps = 100000
        timesteps_per_episode = 1000
        num_episodes = total_timesteps // timesteps_per_episode

        callback = EveryStepCallback(verbose=1)
        for episode in range(1, num_episodes + 1):
            self.env.reset()
            self.model.learn(total_timesteps=timesteps_per_episode, reset_num_timesteps=False, callback=callback)

            # 모델 파일을 생성한 폴더에 저장
            model_path = f"{self.output_folder}/dqn_model_episode_{episode}.zip"
            self.model.save(model_path)
            print(f"Episode {episode}: Model saved as {model_path}")

learnning = LearningManager()
learnning.run_simulation()