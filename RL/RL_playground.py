import gym
import torch
import torch.nn as nn
import torch.optim as optim
import random
import numpy as np
from collections import deque


# 定义Q网络
class QNetwork(nn.Module):
    def __init__(self, state_size, action_size):
        super(QNetwork, self).__init__()
        self.fc1 = nn.Linear(state_size, 24)
        self.fc2 = nn.Linear(24, 24)
        self.fc3 = nn.Linear(24, action_size)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x


# 超参数
state_size = 4
action_size = 2
learning_rate = 0.001
gamma = 0.99
epsilon_start = 1.0
epsilon_end = 0.01
epsilon_decay = 0.995
buffer_size = 100000
batch_size = 64
target_update = 10
episodes = 500


# 经验回放缓冲区
class ReplayBuffer:
    def __init__(self, buffer_size):
        self.buffer = deque(maxlen=buffer_size)

    def add(self, experience):
        self.buffer.append(experience)

    def sample(self, batch_size):
        return random.sample(self.buffer, batch_size)

    def __len__(self):
        return len(self.buffer)


# 创建环境
env = gym.make('CartPole-v1')

# 初始化Q网络和目标网络
q_network = QNetwork(state_size, action_size)
target_network = QNetwork(state_size, action_size)
target_network.load_state_dict(q_network.state_dict())
target_network.eval()

optimizer = optim.Adam(q_network.parameters(), lr=learning_rate)
replay_buffer = ReplayBuffer(buffer_size)

epsilon = epsilon_start

# 训练DQN
for episode in range(episodes):
    state = env.reset()
    state = torch.FloatTensor(state).unsqueeze(0)
    total_reward = 0

    for t in range(1000):
        # 根据epsilon-greedy策略选择动作
        if random.random() < epsilon:
            action = env.action_space.sample()
        else:
            with torch.no_grad():
                action = q_network(state).argmax().item()

        next_state, reward, done, _ = env.step(action)
        next_state = torch.FloatTensor(next_state).unsqueeze(0)
        total_reward += reward

        # 将经验添加到回放缓冲区
        replay_buffer.add((state, action, reward, next_state, done))
        state = next_state

        if done:
            break

        # 从缓冲区中采样并训练Q网络
        if len(replay_buffer) > batch_size:
            batch = replay_buffer.sample(batch_size)
            states, actions, rewards, next_states, dones = zip(*batch)

            states = torch.cat(states)
            actions = torch.tensor(actions).unsqueeze(1)
            rewards = torch.tensor(rewards).unsqueeze(1)
            next_states = torch.cat(next_states)
            dones = torch.tensor(dones).unsqueeze(1)

            current_q_values = q_network(states).gather(1, actions)
            max_next_q_values = target_network(next_states).max(1)[0].unsqueeze(1)
            expected_q_values = rewards + (gamma * max_next_q_values * (1 - dones))

            loss = nn.MSELoss()(current_q_values, expected_q_values)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    # 减少epsilon
    epsilon = max(epsilon_end, epsilon_decay * epsilon)

    # 每隔几个episode更新目标网络
    if episode % target_update == 0:
        target_network.load_state_dict(q_network.state_dict())

    print(f"Episode {episode}, Total Reward: {total_reward}")

env.close()
