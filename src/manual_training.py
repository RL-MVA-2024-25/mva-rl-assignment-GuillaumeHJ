from evaluate import evaluate_HIV, evaluate_HIV_population
from train import ProjectAgent, env  # Replace DummyAgent with your agent implementation
import random
import os
from pathlib import Path
import numpy as np
import torch
import matplotlib as plt

def seed_everything(seed: int = 42):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.cuda.manual_seed_all(seed)


file = Path("score.txt")
if not file.is_file():
    seed_everything(seed=42)
    # Initialization of the agent. Replace DummyAgent with your custom agent implementation.
    agent = ProjectAgent()

    scores = agent.train(env, 200)
    plt.plot(scores)

    # Evaluate agent and write score.
    score_agent: float = evaluate_HIV(agent=agent, nb_episode=5)
    score_agent_dr: float = evaluate_HIV_population(agent=agent, nb_episode=20)
    with open(file="score.txt", mode="w") as f:
        f.write(f"{score_agent}\n{score_agent_dr}")
