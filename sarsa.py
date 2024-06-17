import random

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import gymnasium as gym

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(4, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, 2)
    
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

class SarsaAgent:
    def __init__(self):
        self.memory = []
        self.learning_rate = 0.0005
        self.gamma = 0.99
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.net = Net().to(self.device)
        self.criterion = nn.MSELoss()
        self.optimizer = optim.Adam(self.net.parameters(), lr=self.learning_rate)

    def sample_action(self, s, epsilon):
        with torch.no_grad():
            q_values: torch.Tensor = self.net(s.to(self.device))
            # random.random: [0, 1)
            if random.random() > epsilon:
                return q_values.argmax().item()
            else:
                # masking by p
                return np.random.choice(range(len(q_values))).item()

    def put_data(self, transition):
        self.memory.append(transition)

    def get_data(self):
        s_lst = []
        a_lst = []
        r_lst = []
        s_prime_lst = []
        a_prime_lst = []
        done_mask_lst = []
        for transition in self.memory:
            s, a, r, s_prime, a_prime, done_mask = transition
            s_lst.append(s)
            a_lst.append([a])
            r_lst.append([r])
            s_prime_lst.append(s_prime)
            a_prime_lst.append([a_prime])
            done_mask_lst.append([done_mask])
        s_lst = torch.tensor(s_lst, dtype=torch.float).to(self.device)
        a_lst = torch.tensor(a_lst).to(self.device)
        r_lst = torch.tensor(r_lst).to(self.device)
        s_prime_lst = torch.tensor(s_prime_lst, dtype=torch.float).to(self.device)
        a_prime_lst = torch.tensor(a_prime_lst).to(self.device)
        done_mask_lst = torch.tensor(done_mask_lst).to(self.device)
        self.memory = []
        return s_lst, a_lst, r_lst, s_prime_lst, a_prime_lst, done_mask_lst

    def get_input(self, s, a):
        q = self.net(s)
        q_a = q.gather(-1, a)
        return q_a

    def get_target(self, r, s_prime, a_prime, done_mask):
        with torch.no_grad():
            q_prime = self.net(s_prime)
            q_a_prime = q_prime.gather(-1, a_prime)
            target = r + self.gamma * done_mask * q_a_prime
            return target

    def train(self):
        s, a, r, s_prime, a_prime, done_mask = self.get_data()
        input = self.get_input(s, a)
        target = self.get_target(r, s_prime, a_prime, done_mask)
        loss = self.criterion(input, target)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss.item()

def main():
    env = gym.make("CartPole-v1")
    agent = SarsaAgent()

    print_interval = 20
    score = 0.0

    for n_epi in range(10_000):
        epsilon = max(0.05, 1.0 - 0.95 * (n_epi / 10_000)) #Linear annealing from 100% to 5% that takes 10,000 episodes
        s, _ = env.reset()
        s = s.tolist()
        done = False
        while not done:
            a = agent.sample_action(torch.Tensor(s), epsilon)
            s_prime, r, done, truncated, info = env.step(a)
            s_prime = s_prime.tolist()
            done_mask = 0.0 if done else 1.0
            if agent.memory:
                agent.memory[-1][-2] = a
            # last placeholder a_prime will be ignored by done_mask
            agent.put_data([s, a, r, s_prime, 0, done_mask])
            s = s_prime
            score += r
            if done:
                break
        agent.train()
        if n_epi % print_interval == 0 and n_epi != 0:
            print(f"n_episode :{n_epi}, score : {score / print_interval:.1f}, eps : {epsilon * 100:.1f}%")
            score = 0.0
    env.close()

if __name__ == '__main__':
    main()
