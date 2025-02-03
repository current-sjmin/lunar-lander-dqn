import torch
import torch.nn as nn
import gym


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

env = gym.make("LunarLander-v2", render_mode='human')
state_size = env.observation_space.shape[0]
action_size = env.action_space.n

loaded_model = QNetwork(state_size=state_size, action_size=action_size)
loaded_model.load_state_dict(torch.load("./results/models/best_reward_model4.pth"))
loaded_model.eval()

idx = 0
while True:
    state, data_type = env.reset()
    done = False
    total_reward = 0
    while not done:
        state_tensor = torch.FloatTensor(state).unsqueeze(0)

        with torch.no_grad():
            q_values = loaded_model(state_tensor)
        action = torch.argmax(q_values).item()
        next_state, reward, done, truncated, info = env.step(action=action)
        done = done or truncated
        total_reward += reward
        state = next_state
    print(f"EPISODE / SCORE : {idx+1} / {total_reward:.2f}")
    idx += 1
