import os
import sys
from datetime import datetime
import gymnasium as gym
from stable_baselines3.dqn.dqn import DQN
import traci
from sumo_rl import SumoEnvironment

if "SUMO_HOME" in os.environ:
    tools = os.path.join(os.environ["SUMO_HOME"], "tools")
    sys.path.append(tools)
else:
    sys.exit("Please declare the environment variable 'SUMO_HOME'")

# 학습된 모델 로드
model_path = "path_to_your_model/dqn_model_episode_10.zip"
model = DQN.load(model_path)

# SUMO 환경 설정 (학습에 사용한 동일한 환경 설정 필요)
env = SumoEnvironment(
    net_file="test.net_mergy.xml",
    single_agent=True,
    route_file="generated_flows_am.xml",
    use_gui=True,
    num_seconds=2500,
    yellow_time=4,
    min_green=5,
    max_green=60
)

# 환경을 리셋하여 초기 상태를 가져옴
obs = env.reset()

# 임의의 상태로 수동 설정 (옵션)
# 예: 특정 관측 값으로 설정하는 경우
# obs = [0.5, 0.2, 0.1, 0.3]  # 실제 상태 공간에 맞는 값이어야 함

# 모델을 사용하여 주어진 상태에 대한 행동(action) 예측
action, _states = model.predict(obs, deterministic=True)

# 추론 결과 출력
print(f"Predicted action: {action}")

# 환경에서 예측된 행동을 수행하고 결과를 확인
obs, reward, done, info = env.step(action)

print(f"Observation after action: {obs}")
print(f"Reward: {reward}")
print(f"Done: {done}")
print(f"Info: {info}")
