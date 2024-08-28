import traci
from stable_baselines3 import DQN
from stable_baselines3.common.callbacks import BaseCallback

# 모델 로드
model = DQN.load("dqn_model_episode_72.zip")

import os
from sumo_rl import SumoEnvironment

class CustomSumoEnvironment(SumoEnvironment):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def _compute_reward(self):
        total_co2_emission = 0
        for veh_id in traci.vehicle.getIDList():
            total_co2_emission += traci.vehicle.getCO2Emission(veh_id)

        reward = -total_co2_emission
        return reward


# 콜백 클래스를 정의하여 학습 결과를 매번 출력하도록 설정
class EveryStepCallback(BaseCallback):
    def __init__(self, verbose=0):
        super(EveryStepCallback, self).__init__(verbose)

    def _on_step(self) -> bool:
        if self.verbose > 0:
            print(f"Step: {self.num_timesteps}, Reward: {self.locals['rewards'][-1]}")
        return True

# SUMO 환경 설정
# env = SumoEnvironment(
#     net_file='path_to_net_file.net.xml',
#     route_file='path_to_route_file.rou.xml',
#     use_gui=False,
#     reward_fn=lambda state: -state['CO2_emission'],
#     sumo_binary="sumo"
# )

env = CustomSumoEnvironment(
    net_file="test.net_mergy.xml",
    single_agent=True,
    route_file="generated_flows_pm.xml",
    use_gui=True,
    yellow_time=4,
    min_green=5,
    max_green=120
)

obs, _ = env.reset()
done = False
total_reward = 0

while not done:
    action, _states = model.predict(obs,state=None, deterministic=False)
    obs, reward, done, truncated, info = env.step(action)
    total_reward += reward

print(f"Total CO2 Emission Reward: {total_reward}")
