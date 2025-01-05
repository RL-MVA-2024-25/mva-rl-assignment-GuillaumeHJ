from gymnasium.wrappers import TimeLimit
from env_hiv import HIVPatient
import torch
import torch.nn as nn
import random
import numpy as np
import matplotlib.pyplot as plt
import os

env = TimeLimit(
    env=HIVPatient(domain_randomization=False), max_episode_steps=200
)  # The time wrapper limits the number of steps in an episode at 200.
# Now is the floor is yours to implement the agent and train it.

state_dim = env.observation_space.shape[0]
n_action = env.action_space.n 
nb_neurons=24


class ReplayBuffer:
    def __init__(self, capacity, device):
        self.capacity = int(capacity) # capacity of the buffer
        self.data = []
        self.index = 0 # index of the next cell to be filled
        self.device = device
    def append(self, s, a, r, s_, d):
        if len(self.data) < self.capacity:
            self.data.append(None)
        self.data[self.index] = (s, a, r, s_, d)
        self.index = (self.index + 1) % self.capacity
    def sample(self, batch_size):
        batch = random.sample(self.data, batch_size)
        return list(map(lambda x:torch.Tensor(np.array(x)).to(self.device), list(zip(*batch))))
    def __len__(self):
        return len(self.data)
    
def greedy_action(network, state):
    device = "cuda" if next(network.parameters()).is_cuda else "cpu"
    with torch.no_grad():
        if not isinstance(state, torch.Tensor):
            state = torch.tensor(state, device=device, dtype=torch.float32)  # Convert state to tensor and move to devic
        Q = network(state.unsqueeze(0))
        return torch.argmax(Q).item()

class ProjectAgent:

    def __init__(self, config=None, model=None):

         # Define the device first
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # Use GPU if available

        if config is None:
            self.config = {
                'nb_actions': env.action_space.n,
                'learning_rate': 0.001,
                'gamma': 0.95,
                'buffer_size': 1000000,
                'epsilon_min': 0.01,
                'epsilon_max': 1.,
                'epsilon_decay_period': 1000,
                'epsilon_delay_decay': 20,
                'batch_size': 20
                }
        else:
            self.config = config

        if model is None:
            self.model = torch.nn.Sequential(nn.Linear(state_dim, nb_neurons),
                          nn.ReLU(),
                          nn.Linear(nb_neurons, nb_neurons),
                          nn.ReLU(), 
                          nn.Linear(nb_neurons, n_action)
                          ).to(self.device)
        else:
            self.model = model.to(self.device)  # Ensure model is on the same device
    

        self.gamma = self.config['gamma']
        self.batch_size = self.config['batch_size']
        self.nb_actions = self.config['nb_actions']
        self.memory = ReplayBuffer(self.config['buffer_size'], self.device)
        self.epsilon_max = self.config['epsilon_max']
        self.epsilon_min = self.config['epsilon_min']
        self.epsilon_stop = self.config['epsilon_decay_period']
        self.epsilon_delay = self.config['epsilon_delay_decay']
        self.epsilon_step = (self.epsilon_max-self.epsilon_min)/self.epsilon_stop
        self.criterion = torch.nn.MSELoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.config['learning_rate'])


    def gradient_step(self):
        if len(self.memory) > self.batch_size:
            X, A, R, Y, D = self.memory.sample(self.batch_size)
            X, A, R, Y, D = X.to(self.device), A.to(self.device), R.to(self.device), Y.to(self.device), D.to(self.device)
            QYmax = self.model(Y).max(1)[0].detach()
            update = torch.addcmul(R, 1-D, QYmax, value=self.gamma)
            QXA = self.model(X).gather(1, A.to(torch.long).unsqueeze(1))
            loss = self.criterion(QXA, update.unsqueeze(1))
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step() 
    
    def train(self, env, max_episode):
        episode_return = []
        episode = 0
        episode_cum_reward = 0
        state, _ = env.reset()
        state = torch.tensor(state, device=self.device, dtype=torch.float32)  # Move state to device
        epsilon = self.epsilon_max
        step = 0

        while episode < max_episode:
            # update epsilon
            if step > self.epsilon_delay:
                epsilon = max(self.epsilon_min, epsilon-self.epsilon_step)

            # select epsilon-greedy action
            if np.random.rand() < epsilon:
                action = env.action_space.sample()
            else:
                action = greedy_action(self.model, state)

            # step
            next_state, reward, done, trunc, _ = env.step(action)
            next_state = torch.tensor(next_state, device=self.device, dtype=torch.float32)  # Move next_state to device
            self.memory.append(state, action, reward, next_state, done)
            episode_cum_reward += reward

            # train
            self.gradient_step()

            # next transition
            step += 1
            if done:
                episode += 1
                print("Episode ", '{:3d}'.format(episode), 
                      ", epsilon ", '{:6.2f}'.format(epsilon), 
                      ", batch size ", '{:5d}'.format(len(self.memory)), 
                      ", episode return ", '{:4.1f}'.format(episode_cum_reward),
                      sep='')
                state, _ = env.reset()
                state = torch.tensor(state, device=self.device, dtype=torch.float32)  # Move reset state to device
                episode_return.append(episode_cum_reward)
                episode_cum_reward = 0
            else:
                state = next_state

        return episode_return
    

    def act(self, observation, use_random=False):
        action = greedy_action(self.model, observation)
        return action

    def save(self, path):
        torch.save(self.model.state_dict(), os.path.join(path, "saved_models"))

    def load(self):
         self.model.load_state_dict(torch.load("saved_models.pth", map_location=torch.device('cpu')))
