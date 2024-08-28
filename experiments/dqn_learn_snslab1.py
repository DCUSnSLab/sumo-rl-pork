import os
import sys
from datetime import datetime
import gymnasium as gym
import numpy as np
from gymnasium import spaces
import traci
from sumo_rl import ObservationFunction, TrafficSignal


if "SUMO_HOME" in os.environ:
    tools = os.path.join(os.environ["SUMO_HOME"], "tools")
    sys.path.append(tools)
else:
    sys.exit("Please declare the environment variable 'SUMO_HOME'")

import numpy as np
import traci
from stable_baselines3.dqn.dqn import DQN
from stable_baselines3.common.callbacks import BaseCallback
from sumo_rl import SumoEnvironment

class CO2ObservationFunction(ObservationFunction):
    """CO2-based observation function for traffic signals."""

    def __init__(self, ts: TrafficSignal):
        """Initialize CO2 observation function."""
        super().__init__(ts)

    def __call__(self) -> np.ndarray:
        total_co2_emission = 0
        co2_emissions = []
        for veh_id in traci.vehicle.getIDList():
            total_co2_emission += traci.vehicle.getCO2Emission(veh_id)
        """Return the CO2-based observation."""
        phase_id = [1 if self.ts.green_phase == i else 0 for i in range(self.ts.num_green_phases)]  # one-hot encoding
        min_green = [0 if self.ts.time_since_last_phase_change < self.ts.min_green + self.ts.yellow_time else 1]
        co2_emissions.append(total_co2_emission)
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

class CustomSumoEnvironment(SumoEnvironment):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.reward_fn = CustomSumoEnvironment._compute_reward
    def _compute_reward(self):
        # Get total CO2 emission in the current simulation step
        total_co2_emission = 0
        for veh_id in traci.vehicle.getIDList():
            total_co2_emission += traci.vehicle.getCO2Emission(veh_id)
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


# num_seconds 값을 정의
num_seconds = 3500

start_time = datetime.now().strftime("%Y%m%d_%H%M%S")

# 모델을 저장할 폴더 경로 생성
output_folder = f"outputs/dqn_{start_time}_numsec_{num_seconds}"
os.makedirs(output_folder, exist_ok=True)
env = CustomSumoEnvironment(
    net_file="test.net_mergy.xml",
    single_agent=True,
    route_file="generated_flows_pm.xml",
    out_csv_name=f"{output_folder}/dqn_{start_time}.csv",
    use_gui=True,
    num_seconds=11700,
    yellow_time=4,
    min_green=5,
    max_green=120,
    # reward_fn=CustomSumoEnvironment._compute_reward,
    observation_class= CO2ObservationFunction,
)
class EveryStepCallback(BaseCallback):
    def __init__(self, verbose=0):
        super(EveryStepCallback, self).__init__(verbose)

    def _on_step(self) -> bool:
        if self.verbose > 0:
            pass
            # print(f"Step: {self.num_timesteps}, Reward: {self.locals['rewards'][-1]}")
        return True


model = DQN(
    env=env,
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
total_timesteps = 100000
timesteps_per_episode = 1000
num_episodes = total_timesteps // timesteps_per_episode

callback = EveryStepCallback(verbose=1)
for episode in range(1, num_episodes + 1):
    env.reset()
    model.learn(total_timesteps=timesteps_per_episode, reset_num_timesteps=False, callback=callback)

    # 모델 파일을 생성한 폴더에 저장
    model_path = f"{output_folder}/dqn_model_episode_{episode}.zip"
    model.save(model_path)
    print(f"Episode {episode}: Model saved as {model_path}")

