import random
from collections import deque

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim


class DQNNetwork(nn.Module):

    def __init__(self, state_dim: int, action_dim: int):
        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, action_dim),
        )

    def forward(self, x):
        return self.network(x)


class ReplayBuffer:

    def __init__(self, capacity: int = 10000):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size: int):
        batch = random.sample(self.buffer, batch_size)

        states, actions, rewards, next_states, dones = zip(*batch)

        states = np.array(states, dtype=np.float32)
        actions = np.array(actions, dtype=np.int64)
        rewards = np.array(rewards, dtype=np.float32)
        next_states = np.array(next_states, dtype=np.float32)
        dones = np.array(dones, dtype=np.float32)

        return states, actions, rewards, next_states, dones

    def __len__(self):
        return len(self.buffer)


class DQNAgent:

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        lr: float = 3e-4,
        gamma: float = 0.99,
        epsilon_start: float = 1.0,
        epsilon_end: float = 0.05,
        epsilon_decay: float = 0.995,
        batch_size: int = 128,
        buffer_size: int = 10000,
        target_update_interval: int = 100,
        device: str = None,
    ):

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"

        self.device = torch.device(device)

        self.state_dim = state_dim
        self.action_dim = action_dim

        self.gamma = gamma
        self.batch_size = batch_size
        self.target_update_interval = target_update_interval

        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay

        self.policy_net = DQNNetwork(state_dim, action_dim).to(self.device)
        self.target_net = DQNNetwork(state_dim, action_dim).to(self.device)

        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.AdamW(self.policy_net.parameters(), lr=lr)

        self.replay_buffer = ReplayBuffer(buffer_size)

        self.learn_step = 0


    def select_action(self, state, training: bool = True): 

        if training and random.random() < self.epsilon:
            return random.randrange(self.action_dim)

        state_tensor = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)

        with torch.no_grad():
            q_values = self.policy_net(state_tensor)

        action = torch.argmax(q_values, dim=1).item()

        return action


    def store_transition(self, state, action, reward, next_state, done):

        self.replay_buffer.push(
            state,
            action,
            reward,
            next_state,
            done,
        )


    def train_step(self):

        if len(self.replay_buffer) < self.batch_size:
            return None

        states, actions, rewards, next_states, dones = self.replay_buffer.sample(self.batch_size)

        states = torch.tensor(states, dtype=torch.float32, device=self.device)
        actions = torch.tensor(actions, dtype=torch.long, device=self.device).unsqueeze(1)
        rewards = torch.tensor(rewards, dtype=torch.float32, device=self.device).unsqueeze(1)
        next_states = torch.tensor(next_states, dtype=torch.float32, device=self.device)
        dones = torch.tensor(dones, dtype=torch.float32, device=self.device).unsqueeze(1)

        q_values = self.policy_net(states).gather(1, actions)

        with torch.no_grad():

            next_q_values = self.target_net(next_states).max(dim=1, keepdim=True)[0]

            target_q_values = rewards + self.gamma * (1 - dones) * next_q_values

        loss = nn.functional.smooth_l1_loss(q_values, target_q_values)

        self.optimizer.zero_grad()

        loss.backward()

        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), max_norm=10.0)

        self.optimizer.step()

        self.learn_step += 1

        if self.learn_step % self.target_update_interval == 0:
            self.update_target_network()

        return loss.item()


    def update_target_network(self):

        self.target_net.load_state_dict(self.policy_net.state_dict())


    def decay_epsilon(self):

        self.epsilon = max(
            self.epsilon_end,
            self.epsilon * self.epsilon_decay,
        )


    def save(self, path: str):

        torch.save(
            {
                "policy_net": self.policy_net.state_dict(),
                "target_net": self.target_net.state_dict(),
                "optimizer": self.optimizer.state_dict(),
                "epsilon": self.epsilon,
            },
            path,
        )


    def load(self, path: str):

        checkpoint = torch.load(path, map_location=self.device)

        self.policy_net.load_state_dict(checkpoint["policy_net"])

        self.target_net.load_state_dict(checkpoint["target_net"])

        self.optimizer.load_state_dict(checkpoint["optimizer"])

        self.epsilon = checkpoint["epsilon"]