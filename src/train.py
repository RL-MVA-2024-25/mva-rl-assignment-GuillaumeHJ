from gymnasium.wrappers import TimeLimit
from env_hiv import HIVPatient
from evaluate import evaluate_HIV
import torch
import torch.nn as nn
import random
import numpy as np
import matplotlib.pyplot as plt
import os
from tqdm import tqdm
from copy import deepcopy

env = TimeLimit(
    env=HIVPatient(domain_randomization=False), max_episode_steps=200
)  # The time wrapper limits the number of steps in an episode at 200.
# Now is the floor is yours to implement the agent and train it.

class ReplayBuffer:
    def __init__(self, capacity, device):
        self.capacity = capacity
        self.data = []
        self.index = 0
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

hidden_layer_dim = 256
inner_dim = 128

DQN = torch.nn.Sequential(
            torch.nn.Linear(env.observation_space.shape[0], hidden_layer_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_layer_dim, hidden_layer_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_layer_dim, inner_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(inner_dim, inner_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(inner_dim, hidden_layer_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_layer_dim, hidden_layer_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_layer_dim, env.action_space.n)
)


# ProjectAgent class to define agent logic
# ProjectAgent class to define agent logic
class ProjectAgent:
    def __init__(self):
        # Configuration for the agent
        self.nb_actions = env.action_space.n
        self.learning_rate = 0.0001
        self.gamma = 0.95
        self.buffer_size = 50000
        self.epsilon_min = 0.01
        self.epsilon_max = 1.0
        self.epsilon_decay_period = 50000
        self.epsilon_delay_decay = 1000
        self.epsilon_delay = 100
        self.epsilon_step = (self.epsilon_max - self.epsilon_min) / self.epsilon_decay_period
        self.batch_size = 1024
        self.gradient_steps = 10
        self.update_target_strategy = 'replace'
        self.update_target_freq = 500
        self.update_target_tau = 0.01
        self.criterion = torch.nn.SmoothL1Loss()
        self.fine_tuning = False

        # Linking first the models to GPU device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        if not self.fine_tuning:
            self.model = DQN.to(self.device)
        self.target_model = deepcopy(self.model).to(self.device)
        self.memory = ReplayBuffer(self.buffer_size, self.device)

        # Choice of optimizers for both networks : Adams
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.learning_rate)

    def act(self, observation, use_random=False):
        # Acting greedily towards Q
        with torch.no_grad():  # Disable gradient computation for action selection
            Q = self.model(torch.Tensor(observation).unsqueeze(0).to(self.device))
            return torch.argmax(Q).item()


    def save(self, path):
        self.path = path + "final_dqn.pt"
        torch.save(self.model.state_dict(), self.path)

    def load(self):
        current_path = os.getcwd()
        self.path = current_path + "/final_dqn_deeper_network_more_gradient_steps.pt"
        self.model = DQN.to(self.device)
        self.model.load_state_dict(torch.load(self.path, map_location=self.device))
        self.model.eval()

    def train(self, max_episode):
        previous_val = 0
        episode_return = []
        episode = 0
        episode_cum_reward = 0
        state, _ = env.reset()
        epsilon = self.epsilon_max
        step = 0

        while episode < max_episode:
            # Decay epsilon after a certain number of steps
            if step > self.epsilon_delay:
                epsilon = max(self.epsilon_min, epsilon - self.epsilon_step)

            # Action selection (epsilon-greedy)
            if np.random.rand() < epsilon:
                action = env.action_space.sample()  # Exploration
            else:
                action = self.act(state)  # Exploitation

            # Take action, observe result
            next_state, reward, done, trunc, _ = env.step(action)
            self.memory.append(state, action, reward, next_state, done)
            episode_cum_reward += reward

            # Train the model
            for _ in range(self.gradient_steps):
                if len(self.memory) > self.batch_size:
                    X, A, R, Y, D = self.memory.sample(self.batch_size)
                    QYmax = self.target_model(Y).max(1)[0].detach()
                    update = torch.addcmul(R, 1 - D, QYmax, value=self.gamma)
                    QXA = self.model(X).gather(1, A.to(torch.long).unsqueeze(1))
                    loss = self.criterion(QXA, update.unsqueeze(1))
                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()

            # Update target model periodically
            if self.update_target_strategy == 'replace':
                if step % self.update_target_freq == 0:
                    self.target_model.load_state_dict(self.model.state_dict())

            step += 1
            if done or trunc:
                episode += 1
                val_score = evaluate_HIV(agent=self, nb_episode=1)

                # Print training progress
                print(f"Episode {episode:3d} | "
                      f"Epsilon {epsilon:6.2f} | "
                      f"Batch Size {len(self.memory):5d} | "
                      f"Episode Return {episode_cum_reward:.2e} | "
                      f"Evaluation Score {val_score:.2e}")
                state, _ = env.reset()

                # Save model if evaluation score improves
                if val_score > previous_val:
                    previous_val = val_score
                    self.best_model = deepcopy(self.model).to(self.device)
                    path = os.getcwd()
                    self.save(path)
                episode_return.append(episode_cum_reward)
                episode_cum_reward = 0
            else:
                state = next_state

        # Load the best model and save it
        self.model.load_state_dict(self.best_model.state_dict())
        path = os.getcwd()
        self.save(path)
        return episode_return

def seed_everything(seed: int = 42):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.cuda.manual_seed_all(seed)

