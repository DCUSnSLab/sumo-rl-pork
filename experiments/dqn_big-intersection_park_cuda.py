import gymnasium as gym
import sumo_rl
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from collections import deque
import random
import traci  # SUMO TraCI API


# Q-네트워크 정의
class QNetwork(nn.Module):
    def __init__(self, state_size, action_size):
        super(QNetwork, self).__init__()
        self.fc1 = nn.Linear(state_size, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, action_size)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x


# 하이퍼파라미터 설정
state_size = env.observation_space.shape[0]
action_size = env.action_space.n
q_network = QNetwork(state_size, action_size)
target_network = QNetwork(state_size, action_size)
target_network.load_state_dict(q_network.state_dict())
optimizer = optim.Adam(q_network.parameters(), lr=1e-3)
criterion = nn.MSELoss()

replay_buffer = deque(maxlen=2000)
batch_size = 64
gamma = 0.99
epsilon = 1.0
epsilon_decay = 0.995
epsilon_min = 0.01
target_update_freq = 1000


# 리플레이 버퍼에서 무작위 샘플링
def sample_experiences(batch_size):
    experiences = random.sample(replay_buffer, batch_size)
    states, actions, rewards, next_states, dones = zip(*experiences)
    return np.array(states), actions, rewards, np.array(next_states), dones


# 탄소 배출량 계산 (가상의 함수 예시)
def calculate_carbon_emission():
    total_emission = 0.0
    for vehicle in env.vehicles:
        total_emission += env.get_vehicle_emission(vehicle)
    return total_emission


# Q-네트워크 학습
def train_q_network():
    if len(replay_buffer) < batch_size:
        return

    states, actions, rewards, next_states, dones = sample_experiences(batch_size)

    states = torch.FloatTensor(states)
    actions = torch.LongTensor(actions)
    rewards = torch.FloatTensor(rewards)
    next_states = torch.FloatTensor(next_states)
    dones = torch.FloatTensor(dones)

    q_values = q_network(states).gather(1, actions.unsqueeze(1)).squeeze(1)
    next_q_values = target_network(next_states).max(1)[0]
    target_q_values = rewards + gamma * next_q_values * (1 - dones)

    loss = criterion(q_values, target_q_values)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()


# SUMO 환경 설정
env = gym.make('sumo-rl-v0',
               net_file='test.net_mergy.xml',
               route_file='generated_flows_pm.xml',
               out_csv_name='path_to_output.csv',
               use_gui=True,
               num_seconds=3000)


# 신호등 색상별 최소 시간 설정 함수
def set_traffic_light_min_times(tls_id, min_green=5, min_red=5):
    traci.trafficlight.setPhaseDuration(tls_id, min_green)
    traci.trafficlight.setPhaseDuration(tls_id, min_red)


# 에피소드 결과 저장 함수
def save_episode_results(episode, total_reward, filename='results.txt'):
    with open(filename, 'a') as file:
        file.write(f"Episode {episode + 1}: Total Reward: {total_reward}\n")


# 에피소드 루프
num_episodes = 5
for episode in range(num_episodes):
    state, info = env.reset()  # 에피소드 시작 시 환경 초기화
    set_traffic_light_min_times("0", min_green=10, min_red=5)  # 신호등 최소 시간 설정
    set_traffic_light_min_times("1", min_green=10, min_red=5)  # 신호등 최소 시간 설정
    set_traffic_light_min_times("2", min_green=10, min_red=5)  # 신호등 최소 시간 설정
    set_traffic_light_min_times("3", min_green=10, min_red=5)  # 신호등 최소 시간 설정
    done = False
    total_reward = 0

    while not done:
        if np.random.rand() <= epsilon:
            action = env.action_space.sample()
        else:
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            q_values = q_network(state_tensor)
            action = torch.argmax(q_values).item()

        next_state, reward, terminated, truncated, info = env.step(action)

        # 탄소 배출량을 보상으로 사용
        carbon_emission = calculate_carbon_emission()
        reward = -carbon_emission  # 배출량이 많을수록 패널티

        done = terminated or truncated

        replay_buffer.append((state, action, reward, next_state, done))
        state = next_state
        total_reward += reward

        train_q_network()  # Q-네트워크 학습

        if done:
            epsilon = max(epsilon_min, epsilon_decay * epsilon)
            save_episode_results(episode, total_reward)  # 에피소드 결과 저장
            print(f"Episode {episode + 1}: Total Reward: {total_reward}")

    # 일정 주기마다 타겟 네트워크 업데이트
    if episode % target_update_freq == 0:
        target_network.load_state_dict(q_network.state_dict())

env.close()
