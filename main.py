import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import random
from tqdm import tqdm
import gym
import numpy as np
import matplotlib.pyplot as plt


class QNetwork(nn.Module):
    def __init__(self, state_size, action_size):
        super(QNetwork, self).__init__()
        self.fc1 = nn.Linear(state_size, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, action_size)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        return self.fc3(x)
    
class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)
    
    def add(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))
    
    def sample(self, batch_size):
        return random.sample(self.buffer, batch_size)
    
    def size(self):
        return len(self.buffer)
    

def result_save(data, graph_name):
    vis_x = list(range(len(data)))
    vis_y = data
    plt.plot(vis_x, vis_y)
    plt.title(f"{graph_name} per Episode")
    plt.xlabel("Episode")
    plt.ylabel(f"{graph_name}")
    plt.grid(True)
    plt.savefig(f"./results/{graph_name}.jpg")
    plt.clf()

    
EPISODES = 2000
LEARNING_RATE = 5e-4
REPLAY_BUFFER_CAPACITY = 10000
BATCH_SIZE = 64
GAMMA = 0.999
TARGET_UPDATE_FREQUENCY = 5

epsilon = 1.0
epsilon_decay = 0.9999
min_epsilon = 0.001

env = gym.make("LunarLander-v2", render_mode=None)
state_size = env.observation_space.shape[0]
action_size = env.action_space.n

q_network = QNetwork(state_size=state_size, action_size=action_size)
optimizer = optim.Adam(q_network.parameters(), lr=LEARNING_RATE)

target_network = QNetwork(state_size=state_size, action_size=action_size)
target_network.load_state_dict(q_network.state_dict())
target_network.eval()

replay_buffer = ReplayBuffer(REPLAY_BUFFER_CAPACITY)

result_rewards = []
epsilons = []
buff_size_log = []
best_reward = float('-inf')

for episode in tqdm(range(EPISODES), desc="EPISODE TRAINING"):
    state, _ = env.reset()
    done = False
    total_reward = 0

    while not done:
        state_tensor = torch.FloatTensor(state)
        if random.random() < epsilon:
            action = env.action_space.sample()
        else:
            with torch.no_grad():
                q_values = q_network(state_tensor)
            action = torch.argmax(q_values).item()
        
        epsilon = max(min_epsilon, epsilon * epsilon_decay)

        next_state, reward, done, truncated, info = env.step(action)
        done = done or truncated
        total_reward += reward

        replay_buffer.add(state, action, reward, next_state, done)
        state = next_state
        
        if replay_buffer.size() >= BATCH_SIZE:
            batch = replay_buffer.sample(batch_size=BATCH_SIZE)
            states, actions, rewards, next_states, dones = zip(*batch)
            states = torch.FloatTensor(np.array(states))
            actions = torch.LongTensor(actions).unsqueeze(1)
            rewards = torch.FloatTensor(rewards)
            next_states = torch.FloatTensor(np.array(next_states))
            dones = torch.FloatTensor(dones)
            current_q = q_network(states).gather(1, actions)

            with torch.no_grad():
                max_next_q = target_network(next_states).max(1)[0]
            expected_q = rewards + (GAMMA * max_next_q * (1 - dones)) 

            loss = nn.MSELoss()(current_q.squeeze(), expected_q.detach())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    if episode % TARGET_UPDATE_FREQUENCY == 0:
        target_network.load_state_dict(q_network.state_dict())

    if best_reward <= total_reward and total_reward > 0:
        if episode > int(EPISODES / 2):
            torch.save(q_network.state_dict(), f"./results/models/best_reward_model{episode:05}.pth")
        best_reward = total_reward

    result_rewards.append(total_reward)
    epsilons.append(epsilon)
    buff_size_log.append(replay_buffer.size())

    if episode % 10 == 0:
        result_save(result_rewards, "Reward")
        result_save(epsilons, "Epsilon")
        result_save(buff_size_log, "Buffer Size")

env.close()