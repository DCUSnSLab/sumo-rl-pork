import os
import sys
from datetime import datetime
import gymnasium as gym
from stable_baselines3.dqn.dqn import DQN
from stable_baselines3.common.callbacks import BaseCallback  # 콜백 클래스 임포트
import traci
from sumo_rl import SumoEnvironment

if "SUMO_HOME" in os.environ:
    tools = os.path.join(os.environ["SUMO_HOME"], "tools")
    sys.path.append(tools)
else:
    sys.exit("Please declare the environment variable 'SUMO_HOME'")


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


# 현재 시스템 시간을 가져와 시작 시간으로 저장
start_time = datetime.now().strftime("%Y%m%d_%H%M%S")

# num_seconds 값을 정의
num_seconds = 11000

# 모델을 저장할 폴더 경로 생성
output_folder = f"outputs/dqn_{start_time}_numsec_{num_seconds}"
os.makedirs(output_folder, exist_ok=True)

# out_csv_name에 시스템 시간을 반영
env = CustomSumoEnvironment(
    net_file="test.net_mergy.xml",
    single_agent=True,
    route_file="generated_flows_am.xml",
    out_csv_name=f"{output_folder}/dqn_{start_time}.csv",
    use_gui=True,
    num_seconds=num_seconds,
    yellow_time=4,
    min_green=5,
    max_green=60
)

model = DQN(
    env=env,
    policy="MlpPolicy",
    learning_rate=2e-3,
    learning_starts=0,
    buffer_size=50000,
    train_freq=1,
    target_update_interval=500,
    exploration_fraction=0.05,
    exploration_final_eps=0.01,
    verbose=1,
    device="cpu"
)

total_timesteps = 100000
timesteps_per_episode = 1000
num_episodes = total_timesteps // timesteps_per_episode

# 콜백 인스턴스 생성
callback = EveryStepCallback(verbose=1)

for episode in range(1, num_episodes + 1):
    env.reset()
    model.learn(total_timesteps=timesteps_per_episode, reset_num_timesteps=False, callback=callback)
    model.act()
    # 모델 파일을 생성한 폴더에 저장
    model_path = f"{output_folder}/dqn_model_episode_{episode}.zip"
    model.save(model_path)

    print(f"Episode {episode}: Model saved as {model_path}")
