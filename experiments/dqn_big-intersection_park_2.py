import os
import sys

import gymnasium as gym

if "SUMO_HOME" in os.environ:
    tools = os.path.join(os.environ["SUMO_HOME"], "tools")
    sys.path.append(tools)
else:
    sys.exit("Please declare the environment variable 'SUMO_HOME'")

import numpy as np
import traci
from stable_baselines3.dqn.dqn import DQN

from sumo_rl import SumoEnvironment


class CustomSumoEnvironment(SumoEnvironment):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def _compute_reward(self):
        # Get total CO2 emission in the current simulation step
        total_co2_emission = 0
        for veh_id in traci.vehicle.getIDList():
            total_co2_emission += traci.vehicle.getCO2Emission(veh_id)

        # Calculate the reward as the negative of the CO2 emission
        reward = -total_co2_emission
        return reward


env = CustomSumoEnvironment(
    net_file="test.net_mergy.xml",
    single_agent=True,
    route_file="generated_flows_am.xml",
    out_csv_name="outputs/big-intersection/dqn",
    use_gui=True,
    num_seconds=2500,
    yellow_time=4,
    min_green=5,
    max_green=60
)

model = DQN(
    env=env,
    policy="MlpPolicy",
    learning_rate=3e-3,
    learning_starts=0,
    buffer_size=50000,
    train_freq=1,
    target_update_interval=500,
    exploration_fraction=0.05,
    exploration_final_eps=0.01,
    verbose=1,
    device="cuda"
)

total_timesteps = 100000
timesteps_per_episode = 1000
num_episodes = total_timesteps // timesteps_per_episode

for episode in range(1, num_episodes + 1):
    env.reset()
    model.learn(total_timesteps=timesteps_per_episode, reset_num_timesteps=False)
    model_path = f"dqn_model_episode_{episode}.zip"
    model.save(model_path)
    print(f"Episode {episode}: Model saved as {model_path}")
